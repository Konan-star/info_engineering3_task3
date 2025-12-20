from sklearn.model_selection import train_test_split
from solve_math import load_jsonl, write_jsonl

# 読み込むファイルの名前を登録
INPUT_FILE = "math_level12_easy_train100_student_with_answer_solution.jsonl"

# 作成するファイルの名前を登録
TRAIN_FILE = "math_train80.jsonl"
VAL_FILE   = "math_val20.jsonl"

# 分割の設定
TRAIN_RATIO = 0.8 # trainに80問、valに20問
SEED = 42 # 再現用

# JSONLファイルを読み込む
data = load_jsonl(INPUT_FILE)

# 分割後もlevelの分布が大体同じになるようにしたいので
# stratify 用に level のリストを作る
level_list = [d["level"] for d in data]
train_data, val_data = train_test_split(
    data,
    train_size=TRAIN_RATIO,
    random_state=SEED,
    stratify=level_list # 分布の指定をしたくないときは、この行をコメントアウト
)

# JSONLファイルに書き込む
write_jsonl(TRAIN_FILE, train_data)
write_jsonl(VAL_FILE, val_data)