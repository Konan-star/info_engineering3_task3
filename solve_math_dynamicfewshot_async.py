import os
import json
import asyncio
import time
import re
from collections import Counter

from openai import AsyncOpenAI

# --- Dynamic few-shot 用（ローカル類似度計算） ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Self-consistency の票割れ防止（式の正規化） ---
try:
    import sympy as sp
except ImportError:
    sp = None


# --- 定数設定 ---
API_KEY_FILE = "API.txt"
# 実験用設定
TRAIN_FILE = "math_train80.jsonl"
TEST_FILE = "math_val20.jsonl"
OUTPUT_FILE = "val20_preds_async.jsonl"
MODEL = "gpt-4o-mini"

# Dynamic few-shot: 類似例を何個入れるか
DYNAMIC_FEW_SHOT_K = 7

# Self-consistency: 何本生成して多数決するか
SC_SAMPLES = 15
SC_TEMPERATURE = 0.7

# 生成トークン上限（推論過程を含むので多めに）
MAX_TOKENS = 1500

# 並列実行数の制限（レート制限対策）
MAX_CONCURRENT = 8  # 同時に処理する問題数


def read_api_key(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"APIキーファイル '{file_path}' が見つかりません。"
            "ファイルを作成し、OpenAI APIキーを記述してください。"
        )
    with open(file_path, "r") as f:
        api_key = f.read().strip()
    if not api_key:
        raise ValueError(f"'{file_path}' が空です。APIキーを記述してください。")
    return api_key


def load_jsonl(file_path: str):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_jsonl(file_path: str, data):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"結果を '{file_path}' に正常に書き込みました。")


def clean_answer(answer: str) -> str:
    """
    抽出した答えから余計な記号や文章を除去する。
    """
    if not answer:
        return ""

    s = answer.strip()

    # Unicode数学記号をLaTeX形式に変換
    s = s.replace("π", "\\pi")
    s = s.replace("α", "\\alpha")
    s = s.replace("β", "\\beta")
    s = s.replace("θ", "\\theta")
    s = s.replace("√", "\\sqrt")

    # 余計な文を除去
    s = re.sub(r'^(?:Therefore|Thus|So|Hence|Therefore,|Thus,|So,|Hence,|The answer is|The answer|Answer is|Answer)\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^(?:which gives|which is|which equals|equals|is)\s*', '', s, flags=re.IGNORECASE)

    # LaTeX記号の除去（ただし数式は保持）
    # \text{...} を除去
    s = re.sub(r'\\text\{[^}]*\}', '', s)

    # 余計な説明文を除去（括弧内の説明など）
    s = re.sub(r'\\quad\s*\([^)]*\)', '', s)  # \quad (説明) を除去
    s = re.sub(r'\s*\([^)]*not[^)]*\)', '', s, flags=re.IGNORECASE)  # (not ...) を除去
    s = re.sub(r'\s*\([^)]*\binteger\b[^)]*\)', '', s, flags=re.IGNORECASE)  # (integer) を除去
    s = re.sub(r'\s*\([^)]*\)', '', s)  # 残りの括弧内説明も除去

    # \quad や余計なLaTeXコマンドを除去
    s = re.sub(r'\\quad\b', '', s)

    # 末尾の余計な記号や不完全なLaTeXコマンドを除去
    s = re.sub(r'\\\\+$', '', s)  # 末尾の \\ を除去
    # 注意: 単独の \ は削除しない（\pi などの有効なコマンドを保持）
    s = re.sub(r'\s*\\\)\s*$', '', s)  # 末尾の \) を除去（正しくエスケープ）
    s = re.sub(r'\s*\.\s*$', '', s)  # 末尾の . を除去
    s = re.sub(r'\*+$', '', s)  # 末尾の * を除去（** など）

    # 末尾の孤立した } のみを除去（\frac{...} などの一部でない場合）
    # { と } の数をカウントして、余分な } がある場合のみ除去
    open_count = s.count('{')
    close_count = s.count('}')
    if close_count > open_count:
        # 余分な } を末尾から除去
        for _ in range(close_count - open_count):
            s = re.sub(r'\}\s*$', '', s, count=1)

    # 末尾の不完全なLaTeXコマンド（\leq, \geq, \neq など）を除去
    # 注意: \pi, \alpha などの数学定数は除去しない
    s = re.sub(r'\s*\\(?:leq|geq|le|ge|neq|lt|gt|times|cdot|pm|mp)\s*$', '', s, flags=re.IGNORECASE)

    # 文章っぽいものを除去（長すぎる、単語が多すぎる）
    words = s.split()
    if len(words) > 10:  # 単語が多すぎる場合は文章と判断
        # 数値や式だけを抽出
        numbers = re.findall(r'[\d\.\+\-\*/\(\)]+|\\frac\{[^}]+\}\{[^}]+\}|\\sqrt\{[^}]+\}', s)
        if numbers:
            s = numbers[-1]  # 最後の数値や式を使用

    # 不完全な式や文章を除去
    # 英字だけ（LaTeXコマンドを除いた後）の場合は除去
    test_s = re.sub(r'\\[a-zA-Z]+\{?[^}]*\}?', '', s)
    if re.match(r'^[A-Za-z\s]+$', test_s):
        return ""

    # 末尾に不完全なLaTeXコマンドがまだ残っている場合は除去
    # ただし、数学定数（\pi, \alpha, \betaなど）は保持する
    if re.search(r'\\[a-zA-Z]+\s*$', s):
        # 数学定数以外の不完全なコマンドを除去
        if not re.search(r'\\(?:pi|alpha|beta|gamma|delta|theta|phi|sigma|tau|omega|lambda|mu|nu|rho|epsilon|zeta|eta|kappa|chi|psi)\s*$', s, flags=re.IGNORECASE):
            # 不完全なコマンドの前の部分を抽出
            s = re.sub(r'\s*\\[a-zA-Z]+\s*$', '', s)

    return s.strip()


def extract_final_answer(text: str):
    """
    推論過程から最終的な答えを抽出する。
    """
    if text is None:
        return None
    s = str(text)

    # 1) FINAL: を優先（\text{FINAL:}も含む）
    # まず \text{...} 内のFINALパターンを探す
    m = re.search(r"\\text\{FINAL:\s*\}\s*([^\n\]]+)", s, flags=re.IGNORECASE)
    if m:
        answer = m.group(1).strip()
        return clean_answer(answer)

    # 通常のFINAL:パターン
    m = re.search(r"FINAL:\s*(.+)", s, flags=re.IGNORECASE)
    if m:
        answer = m.group(1).strip()
        return clean_answer(answer)

    # 2) \boxed{...}
    m = re.search(r"\\boxed\{([^}]*)\}", s)
    if m:
        answer = m.group(1).strip()
        return clean_answer(answer)

    # 3) 最後の数値や式を探す
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        # 最後の数行を逆順で見る
        for line in reversed(lines[-5:]):
            # = の後の数値や式を探す
            m = re.search(r'=\s*([^\n=]+?)(?:\.|$|\n|\\text|\\quad)', line)
            if m:
                candidate = m.group(1).strip()
                # 数値や式らしいものを抽出
                if candidate and len(candidate) < 100:
                    cleaned = clean_answer(candidate)
                    if cleaned and not re.match(r'^[A-Za-z\s]+$', cleaned):  # 英字だけの文章でない
                        return cleaned

            # 数値や式パターンを直接探す（より具体的に）
            # LaTeX式を優先し、次に数値（複雑な式から単純な数値の順）
            complete_patterns = [
                r'\\frac\{[^}]+\}\{[^}]+\}',  # 分数（最優先）
                r'\\sqrt\{[^}]+\}',  # 平方根
                r'\d+\.\d+',  # 小数
                r'(?<![a-zA-Z\^])\d+(?![a-zA-Z])',  # 独立した整数（変数や演算子に囲まれていない）
            ]
            for pattern in complete_patterns:
                matches = re.findall(pattern, line)
                if matches:
                    candidate = matches[-1]  # 最後のマッチ
                    cleaned = clean_answer(candidate)
                    # 不完全な式でないことを確認
                    if cleaned and not re.match(r'^[A-Za-z\s]+$', cleaned) and not re.search(r'\\[a-zA-Z]+\s*$', cleaned):
                        return cleaned

        # 4) 最後の非空行（最後の手段）
        last_line = lines[-1]
        cleaned = clean_answer(last_line)
        if cleaned:
            return cleaned

    return None


def normalize_answer(ans: str) -> str:
    """
    Self-consistency の多数決用に答案を正規化。
    """
    if ans is None:
        return ""
    s = ans.strip()

    # よくある余計なプレフィックス除去（モデルがたまに付ける）
    s = re.sub(r"^(答え|Answer)\s*[:：]\s*", "", s, flags=re.IGNORECASE)

    # 空白正規化
    s = re.sub(r"\s+", "", s)

    # sympy で式として扱えるなら簡約（票割れを減らす）
    if sp is not None and s != "":
        try:
            expr = sp.sympify(s, evaluate=False)
            expr_simplified = sp.simplify(expr)
            s2 = str(expr_simplified)
            s2 = re.sub(r"\s+", "", s2)
            return s2
        except (sp.SympifyError, ValueError, SyntaxError, TypeError, Exception) as e:
            pass

    return s


def build_retriever(train_data):
    """
    train_data の problem を TF-IDF ベクトル化して、検索器を作る。
    """
    train_problems = [item["problem"] for item in train_data]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=50000
    )
    train_matrix = vectorizer.fit_transform(train_problems)
    return vectorizer, train_matrix


def retrieve_few_shots(problem_text, train_data, vectorizer, train_matrix, k: int, problem_type=None):
    """
    テスト問題に対して、train_data から類似度上位k件を返す。
    problem_typeが指定されている場合は、同じtypeの問題を優先する。
    """
    q = vectorizer.transform([problem_text])
    sims = cosine_similarity(q, train_matrix).ravel()

    if problem_type:
        # 同じtypeの問題のインデックスを取得
        same_type_indices = [i for i, item in enumerate(train_data) if item.get("type") == problem_type]
        other_type_indices = [i for i, item in enumerate(train_data) if item.get("type") != problem_type]

        # 同じtypeの中で類似度上位を選択
        same_type_sims = [(i, sims[i]) for i in same_type_indices]
        same_type_sims.sort(key=lambda x: x[1], reverse=True)

        # k個選ぶ（同じtypeが足りない場合は他のtypeからも選ぶ）
        selected_indices = [i for i, _ in same_type_sims[:k]]

        if len(selected_indices) < k:
            # 他のtypeから残りを選択
            other_type_sims = [(i, sims[i]) for i in other_type_indices]
            other_type_sims.sort(key=lambda x: x[1], reverse=True)
            remaining = k - len(selected_indices)
            selected_indices.extend([i for i, _ in other_type_sims[:remaining]])

        return [train_data[i] for i in selected_indices]
    else:
        # typeが指定されていない場合は従来通り
        top_idx = sims.argsort()[::-1][:k]
        return [train_data[i] for i in top_idx]


def make_messages(system_prompt: str, few_shots, problem_text: str):
    messages = [{"role": "system", "content": system_prompt}]
    for ex in few_shots:
        messages.append({"role": "user", "content": f"問題:\n{ex['problem']}"})
        # 解法と答えを組み合わせてfew-shot例として提示
        solution_text = ex.get("solution", "")
        answer_text = str(ex.get("answer", ""))
        if solution_text:
            assistant_content = f"{solution_text}\n\n答え: {answer_text}"
        else:
            assistant_content = answer_text
        messages.append({"role": "assistant", "content": assistant_content})
    messages.append({"role": "user", "content": f"問題:\n{problem_text}"})
    return messages


def verify_answer(problem_text: str, answer: str) -> bool:
    """
    答えが問題の条件を満たすか簡易検算。
    """
    if not answer or answer.strip() == "":
        return False

    # 答えから数値を抽出（FINAL: などのプレフィックス除去済みを想定）
    answer_clean = normalize_answer(answer)

    # sympyで数値として解釈できるか試す
    if sp is not None:
        try:
            expr = sp.sympify(answer_clean, evaluate=False)
            # 数値として評価できるか
            if expr.is_number:
                num_val = float(expr.evalf())

                # 問題文から条件を推測
                problem_lower = problem_text.lower()

                # 整数条件チェック（「何個」「何人」「何枚」など）
                integer_keywords = ["how many", "何個", "何人", "何枚", "何本", "何台", "何回", "何点", "何歳"]
                if any(kw in problem_lower for kw in integer_keywords):
                    if not expr.is_integer and not (isinstance(num_val, (int, float)) and num_val.is_integer()):
                        # 整数でない場合は、非常に近い整数かチェック
                        if abs(num_val - round(num_val)) > 1e-6:
                            return False

                # 正の数条件チェック（「何歳」「何個」など）
                positive_keywords = ["how many", "何個", "何人", "何歳", "何枚", "何本", "何台", "何回", "何点", "area", "面積", "perimeter", "周長"]
                if any(kw in problem_lower for kw in positive_keywords):
                    if num_val < 0:
                        return False

                # 極端に大きい値は不自然（10000以上は警告的だが、許容）
                # 極端に小さい負の値も不自然
                if num_val < -10000:
                    return False

                return True
        except (sp.SympifyError, ValueError, SyntaxError, TypeError, Exception) as e:
            pass

    # sympyで解釈できない場合でも、基本的な形式チェック
    # 空でなければ一旦許容（検算できないだけ）
    return len(answer_clean) > 0


def majority_vote(candidates, problem_text: str = None):
    """
    candidates: list[str] 生の答案（推論過程を含む可能性がある）
    problem_text: 問題文（検算用、オプション）
    """
    if not candidates:
        return ""

    # 各候補から最終的な答えを抽出
    extracted_answers = []
    for c in candidates:
        try:
            extracted = extract_final_answer(c)
            if extracted is None:
                extracted = c  # 抽出できなければ生のテキストを使用
            extracted_answers.append(extracted)
        except Exception:
            extracted_answers.append(c)

    # 検算をパスしたものだけをフィルタリング
    original_count = len(extracted_answers)
    if problem_text:
        verified_answers = []
        verified_raw = []
        for ans, raw in zip(extracted_answers, candidates):
            try:
                if verify_answer(problem_text, ans):
                    verified_answers.append(ans)
                    verified_raw.append(raw)
            except Exception:
                verified_answers.append(ans)
                verified_raw.append(raw)

        verified_count = len(verified_answers)
        if verified_answers:
            extracted_answers = verified_answers
            candidates = verified_raw

    # 抽出した答えを正規化して多数決
    normed = []
    for x in extracted_answers:
        try:
            normed.append(normalize_answer(x))
        except Exception:
            normed.append(str(x) if x else "")
    counts = Counter(normed)
    best_norm, best_cnt = counts.most_common(1)[0]

    # tie-break: 同率がある場合、最初に現れたものを採用
    tied = [k for k, v in counts.items() if v == best_cnt]
    if len(tied) > 1:
        for n, raw in zip(normed, extracted_answers):
            if n in tied:
                return raw.strip()

    # 代表として best_norm と一致する最初の raw を返す
    for n, raw in zip(normed, extracted_answers):
        if n == best_norm:
            return raw.strip()

    return extracted_answers[0].strip() if extracted_answers else ""


async def process_one_problem(
    client: AsyncOpenAI,
    test_item: dict,
    train_data: list,
    vectorizer,
    train_matrix,
    system_prompt: str,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int
):
    """1つの問題を非同期で処理"""
    async with semaphore:  # 並列数を制限
        problem_id = test_item["id"]
        problem_text = test_item["problem"]
        problem_type = test_item.get("type")

        print(f"[{index+1}/{total}] 問題ID {problem_id} ({problem_type}) を処理中...")

        # Few-shot検索
        few_shots = retrieve_few_shots(
            problem_text, train_data, vectorizer, train_matrix, DYNAMIC_FEW_SHOT_K, problem_type
        )

        same_type_count = sum(1 for fs in few_shots if fs.get("type") == problem_type)
        few_shot_types = [fs.get("type", "Unknown") for fs in few_shots]
        print(f"  Few-shot例: {same_type_count}/{len(few_shots)}個が同じtype ({', '.join(few_shot_types)})")

        messages = make_messages(system_prompt, few_shots, problem_text)

        try:
            # 非同期でAPI呼び出し
            response = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=SC_TEMPERATURE,
                max_tokens=MAX_TOKENS,
                n=SC_SAMPLES
            )

            candidates = []
            for ch in response.choices:
                if ch.message and ch.message.content:
                    candidates.append(ch.message.content.strip())

            # 多数決
            final_answer = majority_vote(candidates, problem_text)

            # 全候補から抽出した答えを表示
            extracted_candidates = [extract_final_answer(c) or c[:30] + '...' for c in candidates]
            print(f"  最終答え: {final_answer}")
            print(f"  全{len(candidates)}個の候補: {extracted_candidates}")

            return {
                "id": problem_id,
                "prediction": final_answer
            }

        except Exception as e:
            print(f"問題ID {problem_id} の処理中にエラーが発生しました: {e}")
            return {
                "id": problem_id,
                "prediction": f"ERROR: {e}"
            }


async def main():
    try:
        api_key = read_api_key(API_KEY_FILE)
        client = AsyncOpenAI(api_key=api_key)

        train_data = load_jsonl(TRAIN_FILE)
        test_data = load_jsonl(TEST_FILE)

        # --- Dynamic Few-shot 検索器を構築 ---
        vectorizer, train_matrix = build_retriever(train_data)

        system_prompt = (
            "あなたは数学の専門家です。与えられた問題を正確に解いてください。\n\n"
            "重要なルール:\n"
            "- 内部では段階的に推論し、計算や場合分けを丁寧に行ってよい。\n"
            "- 最後に必ず「FINAL: [答え]」の形式で最終的な答えを明示すること。\n"
            "- 答えは数値または簡約した式で返すこと。\n"
            "- 条件（定義域、整数条件、正の条件など）を必ず満たすことを確認すること。\n"
            "- 最後に必ず自己検算（代入・境界・符号・次元など）を行い、矛盾があれば修正してから出力すること。\n\n"
            "出力形式:\n"
            "- 推論過程を書いてもよいが、最後に必ず「FINAL: [答え]」を1行で出力すること\n"
            "- FINAL: の後には答えそのものだけを書く（単位や余計な説明は不要）\n"
            "- 例: FINAL: 24 または FINAL: \\frac{3}{2}"
        )

        # セマフォで並列数を制限
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        # 全問題を非同期で処理
        total_problems = len(test_data)
        print(f"全{total_problems}問を最大{MAX_CONCURRENT}並列で処理開始...\n")

        start_time = time.time()

        tasks = [
            process_one_problem(
                client, test_item, train_data, vectorizer, train_matrix,
                system_prompt, semaphore, i, total_problems
            )
            for i, test_item in enumerate(test_data)
        ]

        predictions = await asyncio.gather(*tasks)

        elapsed_time = time.time() - start_time

        write_jsonl(OUTPUT_FILE, predictions)

        print(f"\n処理完了: {elapsed_time:.1f}秒 (平均: {elapsed_time/total_problems:.1f}秒/問)")

    except (FileNotFoundError, ValueError) as e:
        print(f"エラー: {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
