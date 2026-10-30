# Локально: gigachat, yandexgpt, foreign. local_qwen выполняется в Colab.
import json, os, time
from bank import load_bank
from clients import ask
from config import MODELS, BANK_PATH, LOGS_DIR
from prompts import build_prompt

SKIP_MODELS = {"local_qwen"}
TEMP = 0.0

def completed_pairs() -> set:
    done = set()
    if not os.path.isdir(LOGS_DIR):
        return done
    for fname in os.listdir(LOGS_DIR):
        if not fname.endswith(".jsonl"):
            continue
        mk = fname[:-len(".jsonl")]
        with open(os.path.join(LOGS_DIR, fname), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                if e.get("task_id") is not None:
                    done.add((mk, e["task_id"], e.get("n_shots")))
    return done

def main(lang="ru", temperature=TEMP):
    os.makedirs(LOGS_DIR, exist_ok=True)
    done = completed_pairs()
    tasks = load_bank(BANK_PATH)
    print(f"Задач: {len(tasks)}, уже выполнено записей: {len(done)}")
    for model_key in MODELS:
        if model_key in SKIP_MODELS:
            print(f"=== {model_key}: пропуск (выполняется в Colab) ===")
            continue
        print(f"\n=== {model_key} ===")
        n_done, n_err = 0, 0
        for task in tasks:
            ttype = task["тип"].upper()
            if ttype == "E":
                continue
            if (model_key, task["id"], None) in done:
                continue
            prompt = build_prompt(task, lang=lang)
            try:
                ask(model_key, prompt, temperature=temperature,
                    task_id=task["id"], task_type=ttype)
                n_done += 1
            except Exception as e:
                n_err += 1
                print(f"  {task['id']} ОШИБКА {type(e).__name__}: {e}")
            time.sleep(0.5)
        print(f"[банк] {model_key}: новых {n_done}, ошибок {n_err}")
    print("При ошибках перезапустите скрипт — выполненные задачи будут пропущены.")

if __name__ == "__main__":
    main()
