import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score

from bank import load_bank
from config import (MODELS, LOGS_DIR, RESULTS_DIR, BANK_PATH,
                    ANSWER_PATTERN, CONFIDENCE_PATTERN, RELATIVE_TOLERANCE)

##TRUE_WORDS = {"верно", "истина", "истинно", "да", "правда", "правильно", "верное утверждение"}
##FALSE_WORDS = {"неверно", "ложь", "ложно", "нет", "неправильно", "ложное утверждение"}
##UNSOLVABLE = {"нельзя определить", "недостаточно данных", "невозможно определить",
##              "данных недостаточно", "нельзя"}
NUM_TOKEN = r"-?\d+(?:[.,]\d+)?\s*%?"

_EXACT_TRUE  = {"верно", "истина", "истинно", "да", "правда", "правильно", "true", "yes", "1"}
_EXACT_FALSE = {"неверно", "неправильно", "ложь", "ложно", "false", "no", "0"}
_SUB_FALSE   = ("неверно", "неправильно", "ложь", "ложно", "false")   # отрицания ищем первыми:
_SUB_TRUE    = ("верно", "правильно", "истин", "true")   


# ---------- чтение логов ----------

def load_logs(model_key: str) -> list[dict]:
    path = os.path.join(LOGS_DIR, f"{model_key}.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------- парсинг ответа ----------

def _last(pattern: str, text: str):
    found = re.findall(pattern, text or "", flags=re.IGNORECASE)
    return found[-1].strip() if found else None


def parse_answer(answer: str):
    return _last(ANSWER_PATTERN, answer)


def parse_confidence(answer: str):
    v = _last(CONFIDENCE_PATTERN, answer)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def normalize_number(s) -> float | None:
    """'16%' -> 0.16, '0,161' -> 0.161, '0.161' -> 0.161."""
    if s is None:
        return None
    s = str(s).strip().strip('"\'').rstrip(".")
    sign = -1 if s.startswith(("-", "−")) else 1
    s = s.lstrip("-−+")
    percent = s.endswith("%")
    if percent:
        s = s[:-1]
    s = s.replace(",", ".").replace(" ", "").replace("\u00a0", "")
    try:
        v = float(s)
    except ValueError:
        return None
    return sign * (v / 100 if percent else v)


def parse_class(parsed) -> int | None:
    """Класс 0/1 из поля после ОТВЕТ:."""
    if parsed is None:
        return None
    s = parsed.strip()
    if s in {"0", "1"}:
        return int(s)
    m = re.search(r"\b([01])\b", s)  # 'класс 1' -> 1; '10' не матчится
    return int(m.group(1)) if m else None


# ---------- проверки ----------

def check_a(parsed, reference, tol: float = RELATIVE_TOLERANCE) -> bool:
    """Числовой ответ с допуском. Поддерживает эталон из нескольких чисел (список или ';' )."""
    if parsed is None:
        return False
    exp_raw = reference if isinstance(reference, list) else [str(reference)]
    if len(exp_raw) == 1 and ";" in exp_raw[0]:
        exp_raw = exp_raw[0].split(";")
    exp = [normalize_number(x) for x in exp_raw]
    if any(x is None for x in exp):
        return False
    if len(exp) == 1:
        got = [normalize_number(parsed)]
    else:
        got = [normalize_number(x) for x in re.findall(NUM_TOKEN, parsed)]
        if len(got) != len(exp):
            return False
    for g, x in zip(got, exp):
        if g is None:
            return False
        if x == 0:
            if abs(g) > tol:
                return False
        elif abs(g - x) / abs(x) > tol:
            return False
    return True


def _verdict(s):
    if s is None: return None
    if isinstance(s, bool): return s
    t = str(s).strip().lower().strip(".!()\"'»« \xa0\t")
    if t in _EXACT_TRUE:  return True
    if t in _EXACT_FALSE: return False
    first = re.split(r"[\s,;:—()-]+", t, maxsplit=1)[0].strip(".!()\"'»«")
    if first in _EXACT_TRUE:  return True
    if first in _EXACT_FALSE: return False
    for k in _SUB_FALSE:
        if k in t: return False
    for k in _SUB_TRUE:
        if k in t: return True
    return None

def check_b(parsed, reference) -> bool:
    p, r = _verdict(parsed), _verdict(reference)
    return p is not None and r is not None and p == r


##def check_f(parsed, reference, tol: float = RELATIVE_TOLERANCE) -> bool:
##    """Ловушки: либо число, либо 'нельзя определить'."""
##    if parsed is None:
##        return False
##    ref = str(reference).strip().lower()
##    if any(k in ref for k in UNSOLVABLE):
##        p = parsed.strip().lower()
##        return any(k in p for k in UNSOLVABLE)
##    return check_a(parsed, reference, tol)


# ---------- сбор результатов ----------

def evaluate_bank() -> pd.DataFrame:
    tasks = {t["id"]: t for t in load_bank(BANK_PATH)}
    rows = []
    for model_key in MODELS:
        for e in load_logs(model_key):
            if e.get("task_type") == "E" or e.get("task_id") is None:
                continue
            task = tasks.get(e.get("task_id"))
            if task is None:
                continue
            t = task["тип"].upper()
            parsed = parse_answer(e["answer"])
            if t == "A":
                correct = check_a(parsed, task["эталон"], task.get("допуск", RELATIVE_TOLERANCE))
            elif t == "B":
                correct = check_b(parsed, task["эталон"])
##            elif t == "F":
##                correct = check_f(parsed, task["эталон"], task.get("допуск", RELATIVE_TOLERANCE))
            else:
                correct = None  # C,F и D оцениваются вручную
            rows.append({
                "model": model_key,
                "task_id": task["id"],
                "type": t,
                "parsed": parsed,
                "correct": correct,
                "answered": parsed is not None,
                "confidence": parse_confidence(e["answer"]),
                "total_tokens": e.get("total_tokens"),
                "elapsed_seconds": e.get("elapsed_seconds"),
            })
    return pd.DataFrame(rows)


def evaluate_type_e() -> pd.DataFrame:
    rows = []
    for model_key in MODELS:
        for e in load_logs(model_key):
            if e.get("task_type") != "E":
                continue
            pred = parse_class(parse_answer(e["answer"]))
            rows.append({
                "model": model_key,
                "n_shots": e.get("n_shots"),
                "row_idx": e.get("row_idx"),
                "y_true": e.get("y_true"),
                "y_pred": pred,
                "answered": pred is not None,
                "confidence": parse_confidence(e["answer"]),
                "total_tokens": e.get("total_tokens"),
                "elapsed_seconds": e.get("elapsed_seconds"),
            })
    return pd.DataFrame(rows)


# ---------- сводные таблицы ----------

def bank_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (m, t), g in df.groupby(["model", "type"]):
        correct = g["correct"].fillna(False).astype(bool)
        n_ans = int(g["answered"].sum())
        out.append({
            "model": m, "type": t, "n": len(g), "answered": n_ans,
            "acc_all": round(correct.mean(), 3),
            "acc_answered": round(correct.sum() / n_ans, 3) if n_ans else float("nan"),
        })
    return pd.DataFrame(out)


def type_e_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (m, n), g in df.groupby(["model", "n_shots"]):
        answered = g[g["y_pred"].notna()]
        y_t = g["y_true"].to_numpy(int)
        const_acc = float((y_t == 0).mean())  # константа 'всегда класс 0' на том же тесте
        row = {
            "model": m, "n_shots": n, "n_test": len(g), "n_pos_test": int(y_t.sum()),
            "const_acc_always0": round(const_acc, 3),
            "format_fail_rate": round(1 - len(answered) / len(g), 3),
        }
        if len(answered):
            yt = answered["y_true"].to_numpy(int)
            yp = answered["y_pred"].to_numpy(int)
            row["acc_all"] = round(float((yp == yt).sum()) / len(g), 3)
            row["f1_rare"] = round(float(f1_score(yt, yp, pos_label=1, zero_division=0)), 3)
            try:
                row["balanced_acc"] = round(float(balanced_accuracy_score(yt, yp)), 3)
            except ValueError:
                row["balanced_acc"] = float("nan")
        out.append(row)
    return pd.DataFrame(out)


def bootstrap_ci(correct: pd.Series, n_boot: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    accs = np.array([
        correct.to_numpy()[rng.integers(0, len(correct), len(correct))].mean()
        for _ in range(n_boot)
    ])
    return correct.mean(), np.percentile(accs, 2.5), np.percentile(accs, 97.5)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    bank_df = evaluate_bank()
    e_df = evaluate_type_e()
    bank_df.to_csv(os.path.join(RESULTS_DIR, "eval_bank.csv"), index=False)
    e_df.to_csv(os.path.join(RESULTS_DIR, "eval_type_e.csv"), index=False)

    bs, es = bank_summary(bank_df), type_e_summary(e_df)
    bs.to_csv(os.path.join(RESULTS_DIR, "summary_bank.csv"), index=False)
    es.to_csv(os.path.join(RESULTS_DIR, "summary_type_e.csv"), index=False)
    print(bs.to_string(index=False))
    print()
    print(es.to_string(index=False))

    # Базовые модели LR / NB на тех же примерах (сравнение для типа E)
    from type_e_data import load_ai4i, make_split, classical_baselines
    df = load_ai4i()
    split = make_split(df)
    for n in (5, 20):
        print(f"\nБазовые модели, {n} примеров:", classical_baselines(df, split, n))

    # Точность по моделям на A+B+F с доверительным интервалом
    print()
    abf = bank_df[bank_df["type"].isin(["A", "B", "F"]) & bank_df["correct"].notna()]
    for m, g in abf.groupby("model"):
        acc, lo, hi = bootstrap_ci(g["correct"].astype(bool))
        print(f"{m}: accuracy {acc:.3f} [{lo:.3f}; {hi:.3f}]")
