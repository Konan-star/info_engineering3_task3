#!/usr/bin/env python3
"""
全5パターンのfoldでsolve_math_dynamicfewshot.pyを実行して評価
"""
import os
import json
import subprocess
import time
from collections import Counter

def load_jsonl(file_path: str):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def score_predictions(gold_file, pred_file):
    """予測結果をスコアリング"""
    gold_data = load_jsonl(gold_file)
    pred_data = load_jsonl(pred_file)

    gold_by_id = {item['id']: item['answer'] for item in gold_data}
    pred_by_id = {item['id']: item['prediction'] for item in pred_data}

    correct = 0
    total = 0
    wrong_cases = []

    for problem_id, gold_answer in gold_by_id.items():
        total += 1
        pred_answer = pred_by_id.get(problem_id, '')

        if str(gold_answer).strip() == str(pred_answer).strip():
            correct += 1
        else:
            wrong_cases.append({
                'id': problem_id,
                'gold': gold_answer,
                'pred': pred_answer
            })

    accuracy = correct / total * 100 if total > 0 else 0
    return correct, total, accuracy, wrong_cases

# 全foldの設定
folds = [
    {"name": "Fold 1", "train": "math_train80.jsonl", "val": "math_val20.jsonl", "pred": "val20_preds.jsonl"},
    {"name": "Fold 2", "train": "math_train80_fold2.jsonl", "val": "math_val20_fold2.jsonl", "pred": "val20_preds_fold2.jsonl"},
    {"name": "Fold 3", "train": "math_train80_fold3.jsonl", "val": "math_val20_fold3.jsonl", "pred": "val20_preds_fold3.jsonl"},
    {"name": "Fold 4", "train": "math_train80_fold4.jsonl", "val": "math_val20_fold4.jsonl", "pred": "val20_preds_fold4.jsonl"},
    {"name": "Fold 5", "train": "math_train80_fold5.jsonl", "val": "math_val20_fold5.jsonl", "pred": "val20_preds_fold5.jsonl"},
]

# solve_math_dynamicfewshot.pyを一時的に書き換えて各foldを実行
original_script = "solve_math_dynamicfewshot.py"

# オリジナルのスクリプトを読み込み
with open(original_script, "r", encoding="utf-8") as f:
    original_content = f.read()

all_results = []

print("=" * 80)
print("全5パターンのfoldで評価を実行")
print("=" * 80)

for fold in folds:
    print(f"\n{'=' * 80}")
    print(f"{fold['name']}: {fold['val']}")
    print(f"{'=' * 80}")

    # スクリプトのファイル名を書き換え
    modified_content = original_content
    modified_content = modified_content.replace('TRAIN_FILE = "math_train80.jsonl"', f'TRAIN_FILE = "{fold["train"]}"')
    modified_content = modified_content.replace('TEST_FILE = "math_val20.jsonl"', f'TEST_FILE = "{fold["val"]}"')
    modified_content = modified_content.replace('OUTPUT_FILE = "val20_preds.jsonl"', f'OUTPUT_FILE = "{fold["pred"]}"')

    # 一時スクリプトを作成
    temp_script = f"temp_solve_{fold['name'].replace(' ', '_')}.py"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(modified_content)

    # 実行
    print(f"実行中... (出力は抑制されます)")
    start_time = time.time()
    result = subprocess.run(
        ["python", temp_script],
        capture_output=True,
        text=True
    )
    elapsed_time = time.time() - start_time

    if result.returncode != 0:
        print(f"エラーが発生しました:")
        print(result.stderr)
        continue

    # 一時スクリプトを削除
    os.remove(temp_script)

    # スコアリング
    correct, total, accuracy, wrong_cases = score_predictions(fold["val"], fold["pred"])

    print(f"\n結果:")
    print(f"  正解数: {correct}/{total} ({accuracy:.1f}%)")
    print(f"  実行時間: {elapsed_time:.1f}秒")

    if wrong_cases:
        print(f"  不正解の問題 ({len(wrong_cases)}個): ", end="")
        print(", ".join([f"ID={w['id']}" for w in wrong_cases[:5]]))
        if len(wrong_cases) > 5:
            print(f"    ... 他{len(wrong_cases)-5}個")

    all_results.append({
        "fold": fold["name"],
        "correct": correct,
        "total": total,
        "accuracy": accuracy
    })

# 全体の結果をまとめ
print(f"\n{'=' * 80}")
print("全体の結果")
print(f"{'=' * 80}")
print(f"{'Fold':<10} | {'正解数':<10} | {'精度':<10}")
print("-" * 40)
for r in all_results:
    print(f"{r['fold']:<10} | {r['correct']}/{r['total']:<7} | {r['accuracy']:5.1f}%")

avg_accuracy = sum([r['accuracy'] for r in all_results]) / len(all_results)
print("-" * 40)
print(f"{'平均':<10} | {'':<10} | {avg_accuracy:5.1f}%")

print(f"\n各foldの予測結果ファイル:")
for fold in folds:
    print(f"  {fold['pred']}")
