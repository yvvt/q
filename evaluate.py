import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score

from bank import load_bank
from config import (MODELS, LOGS_DIR, RESULTS_DIR, BANK_PATH,
                    ANSWER_PATTERN, CONFIDENCE_PATTERN, RELATIVE_TOLERANCE)
from prepare_blind_review import NO_ANSWER

TRUE_WORDS = {"верно", "истина", "истинно", "да", "правда", "правильно", "верное утверждение"}
FALSE_WORDS = {"неверно", "ложь", "ложно", "нет", "неправильно", "ложное утверждение"}
UNSOLVABLE = {"нельзя определить", "недостаточно данных", "невозможно определить",
              "данных недостаточно", "нельзя"}
NUM_TOKEN = r"-?\d+(?:[.,]\d+)?\s*%?"

# Типы, оцениваемые вручную; консенсус рецензентов строит evaluate_manual.py
MANUAL_TYPES = ("C", "D", "F")
MANUAL_REVIEW_PATH = os.path.join(RESULTS_DIR, "manual_consensus_long.csv")

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
    s = str(s).strip().strip(' "').rstrip(".")
    sign = -1 if s.startswith(("-", "−")) else 1
    s = s.lstrip("-−+")
    percent = s.endswith("%")
    if percent:
        s = s[:-1]
    s = s.replace(",", ".").replace("  ", " ").replace("\u00a0", " ")
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

# ---------- автопроверки ----------
def check_a(parsed, reference, tol: float = RELATIVE_TOLERANCE) -> bool:
    """Числовой ответ с допуском. Поддерживает эталон из нескольких чисел (список или ';')."""
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

_EXACT_TRUE  = {"верно", "истина", "истинно", "да", "правда", "правильно", "true", "yes", "1"}
_EXACT_FALSE = {"неверно", "неправильно", "ложь", "ложно", "false", "no", "0"}
_SUB_FALSE   = ("неверно", "неправильно", "ложь", "ложно", "false")   # отрицания ищем первыми:
_SUB_TRUE    = ("верно", "правильно", "истин", "true")                # «неверно» содержит «верно»

def _verdict(s):
    if s is None:
        return None
    if isinstance(s, bool):
        return s
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

def check_f(parsed, reference, tol: float = RELATIVE_TOLERANCE) -> bool:
    """Ловушки: либо число, либо 'нельзя определить'.
    После ввода ручной оценки F используется ТОЛЬКО как перекрёстный контроль."""
    if parsed is None:
        return False
    ref = str(reference).strip().lower()
    if any(k in ref for k in UNSOLVABLE):
        p = parsed.strip().lower()
        return any(k in p for k in UNSOLVABLE)
    return check_a(parsed, reference, tol)

# ---------- ручная проверка C/D/F ----------
def load_manual_reviews(path: str = MANUAL_REVIEW_PATH):
    """Консенсус слепой ручной проверки (выход evaluate_manual.py). None, если файла нет."""
    if not os.path.exists(path):
        print(f"ПРЕДУПРЕЖДЕНИЕ: нет файла ручной проверки {path} — "
              f"типы {', '.join(MANUAL_TYPES)} останутся без оценок. "
              f"Сначала запустите evaluate_manual.py.")
        return None
    df = pd.read_csv(path)
    need = {"model_key", "task_id", "type", "score", "ran", "method_error",
            "correct", "noticed_trap", "ответ"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path}: не хватает колонок {sorted(missing)}")
    out = pd.DataFrame({
        "model": df["model_key"].astype(str),
        "task_id": pd.to_numeric(df["task_id"], errors="coerce").astype("Int64").astype("int64"),
        "type": df["type"].astype(str).str.upper(),
        "score": pd.to_numeric(df["score"], errors="coerce"),
        "ran": pd.to_numeric(df["ran"], errors="coerce"),
        "method_error": pd.to_numeric(df["method_error"], errors="coerce"),
        "manual_correct": pd.to_numeric(df["correct"], errors="coerce"),
        "noticed_trap": pd.to_numeric(df["noticed_trap"], errors="coerce"),
        "answered_manual": df["ответ"].astype(str).str.strip() != NO_ANSWER,
    })
    num_cols = ["score", "ran", "method_error", "manual_correct", "noticed_trap"]
    stacked = out[num_cols].stack()
    frac = stacked[stacked.notna() & (stacked % 1 != 0)]
    if len(frac):
        print(f"ПРЕДУПРЕЖДЕНИЕ: {len(frac)} дробных консенсус-значений — расхождения "
              f"рецензентов не согласованы; такие строки ниже считаются неверными.")
    return out

def attach_manual(df: pd.DataFrame, manual) -> pd.DataFrame:
    """Проставляет correct/answered для C, D, F из консенсуса ручной проверки.
    C: correct <=> score == 2;  D: correct <=> ran == 1 и method_error == 0;
    F: correct <=> manual_correct == 1. Строгие равенства: дробный консенсус -> False."""
    if manual is not None:
        df = df.merge(manual, on=["model", "task_id", "type"], how="left")
        cdf = df["type"].isin(MANUAL_TYPES)
        c = (df["type"] == "C") & df["score"].notna()
        df.loc[c, "correct"] = (df.loc[c, "score"] == 2).astype(float)
        d = (df["type"] == "D") & df["ran"].notna() & df["method_error"].notna()
        df.loc[d, "correct"] = ((df.loc[d, "ran"] == 1) & (df.loc[d, "method_error"] == 0)).astype(float)
        f = (df["type"] == "F") & df["manual_correct"].notna()
        df.loc[f, "correct"] = (df.loc[f, "manual_correct"] == 1).astype(float)
        df.loc[cdf, "answered"] = df.loc[cdf, "answered_manual"].astype(bool)
    for col in ("score", "ran", "method_error", "manual_correct", "noticed_trap",
                "answered_manual", "auto_f"):
        if col not in df.columns:
            df[col] = np.nan
    return df

# ---------- сбор результатов ----------
def evaluate_bank(manual=None) -> pd.DataFrame:
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
            else:
                correct = None  # C, D, F оцениваются вручную (attach_manual)
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
                # перекрёстный контроль ручной разметки F, в итог не идёт
                "auto_f": check_f(parsed, task["эталон"],
                                  task.get("допуск", RELATIVE_TOLERANCE)) if t == "F" else None,
            })
    df = pd.DataFrame(rows)
    df = attach_manual(df, manual)
    # единый числовой тип до сохранения: bool -> 1.0/0.0, None -> NaN
    for col in ("correct", "auto_f"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # дедупликация повторов прогонов — в одном месте, как в analysis.py
    df = df.drop_duplicates(subset=["model", "task_id"], keep="first")
    return df

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
        correct = pd.to_numeric(g["correct"], errors="coerce").fillna(0.0).astype(bool)
        n_ans = int(g["answered"].sum())
        row = {
            "model": m, "type": t, "n": len(g), "answered": n_ans,
            "acc_all": round(correct.mean(), 3),
            "acc_answered": round(correct.sum() / n_ans, 3) if n_ans else float("nan"),
        }
        if t == "C" and g["score"].notna().any():          # рубрика 0-2 отдельно от доли двоек
            row["mean_score"] = round(g["score"].dropna().mean(), 3)
            row["mean_score_norm"] = round(g["score"].dropna().mean() / 2, 3)
        if t == "D" and g["ran"].notna().any():
            row["pass_at_1"] = round(g["ran"].dropna().mean(), 3)
            row["method_error_rate"] = round(g["method_error"].dropna().mean(), 3)
        if t == "F" and g["noticed_trap"].notna().any():
            row["noticed_trap_rate"] = round(g["noticed_trap"].dropna().mean(), 3)
        out.append(row)
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
    manual = load_manual_reviews()
    bank_df = evaluate_bank(manual)
    e_df = evaluate_type_e()
    bank_df.to_csv(os.path.join(RESULTS_DIR, "eval_bank.csv"), index=False, encoding="utf-8-sig")
    e_df.to_csv(os.path.join(RESULTS_DIR, "eval_type_e.csv"), index=False)
    bs, es = bank_summary(bank_df), type_e_summary(e_df)
    bs.to_csv(os.path.join(RESULTS_DIR, "summary_bank.csv"), index=False)
    es.to_csv(os.path.join(RESULTS_DIR, "summary_type_e.csv"), index=False)
    print(bs.to_string(index=False))
    print()
    print(es.to_string(index=False))

    # покрытие ручной проверкой: строк C/D/F без консенсуса быть не должно
    miss = bank_df[bank_df["type"].isin(MANUAL_TYPES) & bank_df["correct"].isna()]
    if len(miss):
        print(f"\nПРЕДУПРЕЖДЕНИЕ: нет ручных оценок для {len(miss)} строк C/D/F:")
        print(miss[["model", "task_id", "type"]].to_string(index=False))

    # перекрёстный контроль: ручная разметка F против автопроверки check_f
    fdf = bank_df[(bank_df["type"] == "F") & bank_df["correct"].notna() & bank_df["auto_f"].notna()]
    if len(fdf):
        agree = (fdf["correct"].astype(bool) == fdf["auto_f"].astype(bool)).mean()
        print(f"\nСверка F: ручная оценка совпадает с check_f на {agree:.3f} ({len(fdf)} строк)")

    # Базовые модели LR / NB на тех же примерах (сравнение для типа E)
    from type_e_data import load_ai4i, make_split, FEATURE_COLS, LABEL
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    
    df = load_ai4i()
    split = make_split(df)
    
    preds_rows = []
    for n in (5, 20):
        few = split[f"few{n}"]
        X_tr = df.loc[few, FEATURE_COLS].to_numpy(float)
        y_tr = df.loc[few, LABEL].to_numpy(int)
        X_te = df.loc[split["test_idx"], FEATURE_COLS].to_numpy(float)
        y_te = df.loc[split["test_idx"], LABEL].to_numpy(int)
        
        models = {
            "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
            "naive_bayes": GaussianNB(),
        }
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            
            # Вывод в консоль
            acc = float((pred == y_te).mean())
            f1 = float(f1_score(y_te, pred, pos_label=1, zero_division=0))
            print(f"\nБазовая модель {name}, {n} примеров: acc={acc:.3f}, f1_rare={f1:.3f}")
            
            # Сохранение предсказаний для analysis.py (H12)
            for idx, p, yt in zip(split["test_idx"], pred, y_te):
                preds_rows.append({
                    "classifier": name, "n_shots": n, "row_idx": int(idx),
                    "y_pred": int(p), "y_true": int(yt)
                })
                
    if preds_rows:
        preds_path = os.path.join(RESULTS_DIR, "classical_preds.csv")
        pd.DataFrame(preds_rows).to_csv(preds_path, index=False)
        print(f"Сохранены построчные предсказания базлайнов: {preds_path}")
