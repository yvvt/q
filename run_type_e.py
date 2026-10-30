# Локально: тип E для gigachat, yandexgpt, foreign. local_qwen — в Colab.
import json, os, time
from clients import ask
from config import MODELS, LOGS_DIR
from prompts import build_type_e_prompt
from type_e_data import load_ai4i, make_split, examples_for, row_text, LABEL

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

def fmt_eta(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 60}м {seconds % 60:02d}с"

def main(temperature=TEMP):
    os.makedirs(LOGS_DIR, exist_ok=True)
    done = completed_pairs()
    df = load_ai4i()
    split = make_split(df)  # создаст results/type_e_split.json — этот файл потом в Colab
    for model_key in MODELS:
        if model_key in SKIP_MODELS:
            print(f"=== {model_key}: пропуск (выполняется в Colab) ===")
            continue
        for n in (0, 5, 20):
            examples = examples_for(df, split, n)
            rows = split["test_idx"]
            todo = [i for i in rows if (model_key, f"E_n{n}_{i}", n) not in done]
            done_n = len(rows) - len(todo)
            if not todo:
                print(f"[{model_key}] n={n}: все {len(rows)} уже выполнены")
                continue
            print(f"[{model_key}] n={n}: выполнено {done_n}/{len(rows)}, осталось {len(todo)}")
            n_ok, n_err = 0, 0
            t0 = time.perf_counter()
            for k, i in enumerate(todo, 1):
                tid = f"E_n{n}_{i}"
                prompt = build_type_e_prompt(row_text(df.loc[i]), examples)
                try:
                    ask(model_key, prompt, temperature=temperature,
                        task_id=tid, task_type="E",
                        meta={"n_shots": n, "row_idx": int(i),
                              "y_true": int(df.loc[i, LABEL])})
                    n_ok += 1
                except Exception as e:
                    n_err += 1
                    print(f"  {tid} ОШИБКА {type(e).__name__}: {e}")
                time.sleep(0.3)
                if k % 10 == 0 or k == len(todo):
                    el = time.perf_counter() - t0
                    per = el / k
                    print(f"  [{model_key} n={n}] {k}/{len(todo)} "
                          f"(всего {done_n + k}/{len(rows)}), "
                          f"{per:.1f} с/запрос, прошло {fmt_eta(el)}, "
                          f"осталось ~{fmt_eta(per * (len(todo) - k))}, ошибок {n_err}")
            print(f"[{model_key}] n={n}: новых {n_ok}, ошибок {n_err}")
    print("При ошибках перезапустите скрипт — выполненные задачи будут пропущены.")

if __name__ == "__main__":
    main()
