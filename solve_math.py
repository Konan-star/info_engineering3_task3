import os
import json
from openai import OpenAI
import time

# --- 定数設定 ---
API_KEY_FILE = "API.txt"
TRAIN_FILE = "math_level12_easy_train100_student_with_answer_solution.jsonl"
TEST_FILE = "math_level12_easy_test100_student.jsonl"
OUTPUT_FILE = "team_a_preds.jsonl"
MODEL = "gpt-4o-mini"
# few-shotで学習させるサンプル数
FEW_SHOT_COUNT = 5

def read_api_key(file_path):
    """
    ファイルからAPIキーを読み込みます。
    ファイルが存在しない、または空の場合はエラーを発生させます。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"APIキーファイル '{file_path}' が見つかりません。"
            "ファイルを作成し、OpenAI APIキーを記述してください。"
        )
    with open(file_path, 'r') as f:
        api_key = f.read().strip()
    if not api_key:
        raise ValueError(f"'{file_path}' が空です。APIキーを記述してください。")
    return api_key

def load_jsonl(file_path):
    """
    JSONLファイルを読み込み、辞書のリストとして返します。
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(file_path, data):
    """
    データのリストをJSONLファイルに書き込みます。
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            # ensure_ascii=False で日本語の文字化けを防ぐ
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"結果を '{file_path}' に正常に書き込みました。")

def main():
    """
    メインの処理を実行します。
    1. APIキーとデータを読み込みます。
    2. few-shotプロンプトを作成します。
    3. テストデータの各問題についてAPIを呼び出し、解答を生成します。
    4. 結果を指定されたファイル形式で保存します。
    """
    try:
        # 1. APIキーの読み込みとOpenAIクライアントの初期化
        api_key = read_api_key(API_KEY_FILE)
        client = OpenAI(api_key=api_key)

        # 2. 学習データとテストデータの読み込み
        train_data = load_jsonl(TRAIN_FILE)
        test_data = load_jsonl(TEST_FILE)

        # 3. few-shotプロンプトの準備
        system_prompt = (
            "あなたは数学の専門家です。与えられた数学の問題を解き、最終的な答えだけを返してください。"
            "思考過程、解説、単位、追加のテキストは一切含めないでください。数値または単純化された式のみを回答してください。"
        )

        few_shot_examples = []
        for item in train_data[:FEW_SHOT_COUNT]:
            few_shot_examples.append({"role": "user", "content": f"問題:\n{item['problem']}"})
            few_shot_examples.append({"role": "assistant", "content": str(item['answer'])})

        # 4. 各テスト問題の処理
        predictions = []
        total_problems = len(test_data)
        for i, test_item in enumerate(test_data):
            problem_id = test_item["id"]
            problem_text = test_item["problem"]
            
            print(f"問題ID {problem_id} を処理中... ({i + 1}/{total_problems})")

            messages = [
                {"role": "system", "content": system_prompt},
                *few_shot_examples,
                {"role": "user", "content": f"問題:\n{problem_text}"}
            ]

            try:
                # 5. OpenAI API呼び出し
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0,  # 再現性のある結果を得るために温度を0に設定
                    max_tokens=100, # 解答は短いはずなので、トークン数を制限
                    n=1,
                    stop=None
                )
                answer = response.choices[0].message.content.strip()
                predictions.append({"id": problem_id, "answer": answer})

            except Exception as e:
                print(f"問題ID {problem_id} の処理中にエラーが発生しました: {e}")
                # エラーが発生した場合でも、処理を続行し、エラーがあったことを記録
                predictions.append({"id": problem_id, "answer": f"ERROR: {e}"})
                # APIのレート制限に達した場合などを考慮して少し待機
                time.sleep(5)


        # 6. 結果をファイルに書き込む
        write_jsonl(OUTPUT_FILE, predictions)

    except (FileNotFoundError, ValueError) as e:
        print(f"エラー: {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
