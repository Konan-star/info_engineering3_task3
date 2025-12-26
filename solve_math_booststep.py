import json
import re
import time
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# =========================
# 設定
# =========================
TRAIN_FILE = "math_train80.jsonl"
TEST_FILE = "noanswer_val20.jsonl"
OUTPUT_FILE = "val20_preds.jsonl"
MODEL = "gpt-4o-mini"
API_KEY_FILE = "API.txt"

STEP_SIM_THRESHOLD = 0.3

# =========================
# utilis.pyに対応する部分
# =========================
def read_api_key(path):
    with open(path) as f:
        return f.read().strip()

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def extract_answer(text):
    m = re.search(r"FINAL:\s*(.+)", text) # FINALの後にある空白を無視して、後の文章をとってくる
    return m.group(1).strip() if m else "" # mがあれば前後の空白を消して返し、なければ" "を返す

def construct_message(problem, solution, example, has_example, first_step):
    if first_step:
        prompt = "You are a professional math problem solver. I will give you a math problem. And I will give you another one with its first step which you can refer to. Please output only the first step(fewer than 4000 tokens) to the first problem, starting with 'Step 1:'."
    else:
        prompt = "You are a professional math problem solver. I will give you a math problem and part of its solution. And you need to only output the next step of the solution(fewer than 4000 tokens), starting with 'Step $i$:', where $i$ is the step number. In case you don't know how to derive the correct content, an example with 'Key Step' will be given. You need to learn how 'Key Step' is derived, and implement similar strategy in your derivation procedure. If you think that the final step is derived, output the final answer in the format:'FINAL: [answer]'. The answer must be a single number or a simplified mathematical expression. Do not include explanations after FINAL."
    messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": problem}
        ]
    if solution:
            messages.append({"role": "user", "content": solution})
    if has_example:
            messages.append({"role": "user", "content": example})
    return messages

def save_result(result: dict, output_file: str):
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

# =========================
# example.pyに対応する部分
# =========================
def convert_example(question, step_num):
    """
    steps リストを使って、
    指定された step_num だけを Key Step として強調する
    """
    steps = question["steps"]
    example = 'Problem: ' + question['problem'] + '\n'
    for i, s in enumerate(steps): # stepsリストの番号iと中身s
        if i == step_num:
            example += f'Key Step {i+1}: {s}\n'
        else:
            example += f'Step {i+1}: {s}\n'
    return example

def construct_example_bank(file_path):
    """
    solution から step リストを作る
    各 step をベクトル化して、似た step を検索できる状態にする
    """
    id1 = 0 # 問題番号
    id2 = 0 # ステップ番号
    data_example = [] # 問題丸ごと
    example_step = [] # 全問題の全 step を一列に並べたリスト
    problem2step = {} # step → どの問題の何番目のstepかの対応表

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            question = json.loads(line)
            steps = [s.strip() for s in question["solution"].split("\n") if s.strip()] # solution を改行で分割して steps にする
            data_example.append({
                "problem": question["problem"],
                "steps": steps
            })
            for i, s in enumerate(steps):
                example_step.append(s)
                problem2step[id2] = (id1, i)
                id2 += 1
            id1 += 1
        vectorizer = TfidfVectorizer() # step をベクトル化
        tfidf_matrix = vectorizer.fit_transform(example_step)
        embeddings = tfidf_matrix.toarray()  # numpy array に変換
        
        return vectorizer, tfidf_matrix, embeddings, problem2step, data_example

def retrieve_step(key, vectorizer_step, example_step_embeddings, problem2step, example_data, thrsd=0.7):
    """
    現在の解答の途中（key）に最も似ている過去のステップを検索し、類似度が閾値以上なら参考例として返す
    """
    new_step_embedding = vectorizer_step.transform([key]).toarray() # 今回の key をベクトルに変換
    similarities = cosine_similarity(new_step_embedding, example_step_embeddings).flatten() # key のベクトルと全ステップベクトルのコサイン類似度を計算

    max_similarity = similarities.max() # 最も類似しているステップの類似度
    example_num = problem2step[similarities.argmax()][0] # 最も類似しているステップの問題番号
    example_step = problem2step[similarities.argmax()][1] # 最も類似しているステップのステップ番号

    if max_similarity >= thrsd: # 類似度が閾値以上なら、そのステップを参考例として採用
        has_example = True
        example = 'Example: ' + convert_example(example_data[example_num], example_step)
        print("Example for current step: " + str(example_step))
    else:
        has_example = False
        example = ""
        print("No Example for this step")

    return has_example, example

# =========================
# reasoning.pyに対応する部分
# =========================
def first_try(client, problem, pre_solution, first_step, amc=None):
    example = ""
    has_example = False
    messages = construct_message(problem, pre_solution, example, has_example, first_step)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
        max_tokens=5000,
        top_p=1
    )
    
    first_try_reasoning = response.choices[0].message.content
    print("first_try for this step: " + first_try_reasoning)
    return first_try_reasoning

def solve(client, whole_problem, vectorizer_step, example_step_embeddings, problem2step, example_data, thrsd=0.7):
    """
    ステップごとの推論:
    1. first_try で現在ステップを生成
    2. 類似ステップを retrieve_step で参照
    3. 参照例を使って最終ステップ生成
    """
    problem = 'Problem: ' + whole_problem['problem']
    total_solution = ""
    
    step_num = 0
    max_step = 20
    first_step = True
    
    while True:
        step_num += 1
        print(f"Step-{step_num}: first_try reasoning")
  
        # まず最初の推論ステップを生成
        first_try_reasoning = first_try(client, problem, total_solution, first_step)
        print(f"first_try result: {first_try_reasoning}")    
        
        # 類似ステップを探す
        has_example, example_step = retrieve_step(first_try_reasoning, vectorizer_step, example_step_embeddings, problem2step, example_data, thrsd)
        print(f"Example found: {has_example}")     
        
        # 最終ステップ生成
        new_message = construct_message(problem, total_solution, example_step, has_example, first_step)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=new_message,
            temperature=0,
            max_tokens=5000,
            top_p=1
        )
        final_reasoning = response.choices[0].message.content
        print(f"final reasoning: {final_reasoning}")
        
        pre_solution = total_solution
        total_solution += final_reasoning
        first_step = False     
        
        # 終了条件
        if 'FINAL:' in total_solution:
            break    
        # 行き詰まった場合はループ終了（0-shot 部分は削除）
        if step_num > max_step or final_reasoning in pre_solution:
            print("Reached max steps or repetition detected. Stopping step-by-step reasoning.")
            break
    
    answer = extract_answer(total_solution)
    return total_solution, answer

# =========================
# Main
# =========================
def main():
    client = OpenAI(api_key=read_api_key(API_KEY_FILE))
    vectorizer, tfidf_matrix, embeddings, problem2step, data_example = construct_example_bank(TRAIN_FILE)
    test_data = load_jsonl(TEST_FILE)
    for problem_item in test_data:
        print(f"Solving ID {problem_item['id']}...")     
        # ステップごと推論
        total_solution, answer = solve(
            client, 
            problem_item, 
            vectorizer, 
            embeddings,
            problem2step, 
            data_example, 
            thrsd=STEP_SIM_THRESHOLD
        )
        # predictionを必ず"FINAL: ..."形式にする
        if answer:
            prediction = f"FINAL: {answer}"
        else:
            # もし抽出できなければ空の文字列
            prediction = "FINAL: "
        save_result({
            "id": problem_item["id"],
            "prediction": prediction
        }, OUTPUT_FILE)
        time.sleep(1)

if __name__ == "__main__":
    main()