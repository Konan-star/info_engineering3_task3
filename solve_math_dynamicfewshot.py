import os
import json
import time
import re
from collections import Counter

from openai import OpenAI

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
OUTPUT_FILE = "val20_preds.jsonl"
MODEL = "gpt-4o-mini"

# Dynamic few-shot: 類似例を何個入れるか
DYNAMIC_FEW_SHOT_K = 5

# Self-consistency: 何本生成して多数決するか
SC_SAMPLES = 9
SC_TEMPERATURE = 0.7

# 生成トークン上限（推論過程を含むので多めに）
MAX_TOKENS = 1500


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
    s = re.sub(r'\\+$', '', s)  # 末尾の \ を除去（単独）
    s = re.sub(r'\s*\\\)\s*$', '', s)  # 末尾の \) を除去（正しくエスケープ）
    s = re.sub(r'\s*\.\s*$', '', s)  # 末尾の . を除去
    s = re.sub(r'\*+$', '', s)  # 末尾の * を除去（** など）

    # 末尾の不完全なLaTeXコマンド（\leq, \geq, \neq など）を除去
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
    if re.search(r'\\[a-zA-Z]+\s*$', s):
        # 不完全なコマンドの前の部分を抽出
        s = re.sub(r'\s*\\[a-zA-Z]+\s*$', '', s)

    return s.strip()


def extract_final_answer(text: str):
    """
    推論過程から最終的な答えを抽出する。
    - FINAL: を優先
    - \boxed{...} を次に優先
    - 最後の数値や式を探す（推論過程の最後の答えらしい部分）
    - 最後の非空行を最後の手段として使用
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
                r'[0-9\+\-\*/\(\)\^]+',  # 数式（^を含む）
                r'\d+',  # 整数（最後）
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
    - 前後空白除去
    - 余計な空白や改行を潰す
    - 可能なら sympy で簡約して canonical に寄せる（例: 2/4 -> 1/2）
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
            # sympyのパースエラーを防ぐため、安全にパース
            expr = sp.sympify(s, evaluate=False)
            expr_simplified = sp.simplify(expr)
            # sympyの出力を文字列化（空白なし）
            s2 = str(expr_simplified)
            s2 = re.sub(r"\s+", "", s2)
            return s2
        except (sp.SympifyError, ValueError, SyntaxError, TypeError, Exception) as e:
            # パースエラーは無視して元の文字列を返す
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
    - 数値として解釈できるか
    - 整数条件（「何個」「何人」など）
    - 正の数条件（「何歳」「何個」など）
    - 基本的な範囲チェック
    """
    if not answer or answer.strip() == "":
        return False
    
    # 答えから数値を抽出（FINAL: などのプレフィックス除去済みを想定）
    answer_clean = normalize_answer(answer)
    
    # sympyで数値として解釈できるか試す
    if sp is not None:
        try:
            # sympyのパースエラーを防ぐため、安全にパース
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
            # パースエラーは無視して次のチェックに進む
            pass
    
    # sympyで解釈できない場合でも、基本的な形式チェック
    # 空でなければ一旦許容（検算できないだけ）
    return len(answer_clean) > 0


def majority_vote(candidates, problem_text: str = None):
    """
    candidates: list[str] 生の答案（推論過程を含む可能性がある）
    problem_text: 問題文（検算用、オプション）
    1) 各候補から最終的な答えを抽出
    2) 検算をパスしたものだけを対象にする（problem_textがある場合）
    3) 正規化して投票
    4) 最頻値を返す（同率なら「最初に出たもの」を返す）
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
            # 抽出エラーが発生した場合は生のテキストを使用
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
                # 検算エラーが発生した場合は検算をスキップして候補に含める
                verified_answers.append(ans)
                verified_raw.append(raw)
        
        verified_count = len(verified_answers)
        if verified_answers:
            extracted_answers = verified_answers
            candidates = verified_raw
            if verified_count < original_count:
                print(f"  検算: {original_count}個中{verified_count}個を採用（{original_count - verified_count}個を除外）")
        else:
            print(f"  検算: 全ての候補が検算をパスしませんでした。全候補で多数決します。")

    # 抽出した答えを正規化して多数決
    normed = []
    for x in extracted_answers:
        try:
            normed.append(normalize_answer(x))
        except Exception:
            # 正規化エラーが発生した場合は元の文字列を使用
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


def main():
    try:
        api_key = read_api_key(API_KEY_FILE)
        client = OpenAI(api_key=api_key)

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

        predictions = []
        total_problems = len(test_data)

        for i, test_item in enumerate(test_data):
            problem_id = test_item["id"]
            problem_text = test_item["problem"]
            problem_type = test_item.get("type")

            print(f"問題ID {problem_id} ({problem_type}) を処理中... ({i + 1}/{total_problems})")

            # --- Dynamic Few-shot: この問題に近い例を毎回選ぶ（同じtypeを優先） ---
            few_shots = retrieve_few_shots(
                problem_text, train_data, vectorizer, train_matrix, DYNAMIC_FEW_SHOT_K, problem_type
            )

            # Few-shot例の情報を表示
            same_type_count = sum(1 for fs in few_shots if fs.get("type") == problem_type)
            few_shot_types = [fs.get("type", "Unknown") for fs in few_shots]
            print(f"  Few-shot例: {same_type_count}/{len(few_shots)}個が同じtype ({', '.join(few_shot_types)})")

            messages = make_messages(system_prompt, few_shots, problem_text)

            try:
                # --- Self-consistency: n本生成して多数決 ---
                response = client.chat.completions.create(
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

                # majority_vote内で既に答えを抽出している
                final_answer = majority_vote(candidates, problem_text)

                # 全候補から抽出した答えを表示
                extracted_candidates = [extract_final_answer(c) or c[:30] + '...' for c in candidates]
                print(f"  最終答え: {final_answer}")
                print(f"  全{len(candidates)}個の候補: {extracted_candidates}")

                # score_math_test.pyは"prediction"フィールドを期待している
                # 生の予測を保存（score_math_test.pyがextract_final_answerを使う）
                # ただし、majority_voteで既に抽出されているので、それを保存
                predictions.append({
                    "id": problem_id,
                    "prediction": final_answer
                })

            except Exception as e:
                print(f"問題ID {problem_id} の処理中にエラーが発生しました: {e}")
                predictions.append({"id": problem_id, "prediction": f"ERROR: {e}"})
                time.sleep(5)

        write_jsonl(OUTPUT_FILE, predictions)

    except (FileNotFoundError, ValueError) as e:
        print(f"エラー: {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


if __name__ == "__main__":
    main()
