import pandas as pd
from bank import load_bank

df = pd.read_csv("results/eval_bank.csv")
bank = {t["id"]: t for t in load_bank("data/tasks.jsonl")}
for _, r in df[df.type == "B"].iterrows():
    print(r.model, r.task_id, repr(r.parsed), "| эталон:", repr(bank[r.task_id]["эталон"]), "| correct:", r.correct)
