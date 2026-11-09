"""Ручная проверка типов C, D, F: чтение файлов рецензентов, метрики, согласие.

Файлы ответов кладутся в review/ под именами blind_review_ans*.{xlsx,xls,csv}
(например blind_review_ans.xlsx, blind_review_ans_2.xlsx). Второй рецензент
подключается без правки кода: как только появляется второй файл, автоматически
считаются каппа Коэна, список расхождений и консенсус-метрики.

Запуск: python evaluate_manual.py
Запускать только в контуре организатора: скрипт вскрывает blind_mapping.json.
Публиковать таблицы и проводить согласование расхождений — после сдачи ОБЕИХ оценок.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from config import RESULTS_DIR
from evaluate import UNSOLVABLE          # единый источник ключей «нельзя определить»

REVIEW_DIR = "review"
MAPPING_PATH = os.path.join(REVIEW_DIR, "blind_mapping.json")
ANS_MASKS = (os.path.join(REVIEW_DIR, "blind_review_ans*.xlsx"),
             os.path.join(REVIEW_DIR, "blind_review_ans*.xls"),
             os.path.join(REVIEW_DIR, "blind_review_ans*.csv"))
NO_ANSWER = "[нет ответа]"

INT_COLS = ["score", "ran", "method_error", "correct", "noticed_trap"]
DOMAINS = {                       # допустимые значения по типам (рубрики методички)
    "C": {"score": {0, 1, 2}},
    "D": {"ran": {0, 1}, "method_error": {0, 1}},
    "F": {"correct": {0, 1}, "noticed_trap": {0, 1}},
}
FIELDS_BY_TYPE = [("C", "score"), ("D", "ran"), ("D", "method_error"),
                  ("F", "correct"), ("F", "noticed_trap")]


# ---------- чтение ----------
def load_mapping() -> pd.DataFrame:
    with open(MAPPING_PATH, encoding="utf-8") as f:
        m = json.load(f)
    return pd.DataFrame([{"response_id": k, **v} for k, v in m.items()])


def _to_int(s: pd.Series) -> pd.Series:
    t = s.astype("string").str.strip()
    t = t.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NONE": pd.NA})
    return pd.to_numeric(t, errors="coerce").round().astype("Int64")


def load_reviewer(path: str, mapping: pd.DataFrame) -> pd.DataFrame:
    if path.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(path)
        sheet = "review" if "review" in xl.sheet_names else xl.sheet_names[0]
        df = xl.parse(sheet)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if "response_id" not in df:
        raise ValueError(f"{path}: нет колонки response_id")
    df["response_id"] = df["response_id"].astype(str).str.strip()
    dup = df["response_id"].duplicated().sum()
    if dup:
        print(f"ПРЕДУПРЕЖДЕНИЕ {path}: дублирующихся response_id {dup}, взяты первые")
        df = df[~df["response_id"].duplicated()]
    for c in INT_COLS:
        df[c] = _to_int(df[c]) if c in df else pd.NA

    df = df.merge(mapping, on="response_id", how="left", suffixes=("", "_map"))
    unknown = df["model_key"].isna()
    if unknown.any():
        print(f"ПРЕДУПРЕЖДЕНИЕ {path}: неизвестных response_id {int(unknown.sum())}: "
              f"{df.loc[unknown, 'response_id'].tolist()}")
        df = df[~unknown]
    df["type"] = df["type_map"].fillna(df.get("type"))

    # полнота: какие строки сетки рецензент пропустил
    expected = set(mapping.loc[mapping["type"].isin(DOMAINS), "response_id"])
    missing = sorted(expected - set(df["response_id"]))
    if missing:
        print(f"ПРЕДУПРЕЖДЕНИЕ {path}: не заполнено строк сетки: {len(missing)} {missing}")

    # валидность значений по рубрикам
    problems = []
    for i, r in df.iterrows():
        for col, dom in DOMAINS.get(r["type"], {}).items():
            v = r[col]
            if pd.isna(v):
                problems.append((r["response_id"], col, "пусто"))
            elif int(v) not in dom:
                problems.append((r["response_id"], col, f"вне домена: {v}"))
                df.loc[i, col] = pd.NA
    if problems:
        print(f"ПРЕДУПРЕЖДЕНИЕ {path}: проблемы значений (строки учтены как NaN):")
        for p in problems:
            print("   ", p)

    keep = ["response_id", "model_key", "task_id", "type",
            "ответ", "эталон", "method_error_types", "comment"] + INT_COLS
    for c in keep:
        if c not in df:
            df[c] = pd.NA
    return df[keep]


# ---------- метрики одного рецензента (или консенсуса) ----------
def summarize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = []
    for (model, typ), g in df.groupby(["model_key", "type"]):
        ans = g[g["ответ"].astype(str).str.strip() != NO_ANSWER]   # строки, где модель дала ответ/код
        r = {"source": label, "model": model, "type": typ,
             "n_tasks": len(g), "n_answered": len(ans), "n_refused": len(g) - len(ans)}
        if typ == "C":
            s = ans["score"].astype("float").dropna()
            r.update(n_scored=len(s),
                     mean_score=round(s.mean(), 3) if len(s) else np.nan,
                     share_full=round((s == 2).mean(), 3) if len(s) else np.nan,
                     acc_norm=round(s.mean() / 2, 3) if len(s) else np.nan)  # для тепловой карты
        elif typ == "D":
            ran = ans["ran"].astype("float").dropna()
            me = ans["method_error"].astype("float").dropna()
            r.update(pass_at_1=round(ran.mean(), 3) if len(ran) else np.nan,
                     method_error_rate=round(me.mean(), 3) if len(me) else np.nan,
                     solved_end_to_end=round(ran.sum() / len(g), 3) if len(g) else np.nan)
        elif typ == "F":
            c = ans["correct"].astype("float").dropna()
            nt = ans["noticed_trap"].astype("float").dropna()
            r.update(correct_rate=round(c.mean(), 3) if len(c) else np.nan,
                     noticed_rate=round(nt.mean(), 3) if len(nt) else np.nan)
            uns = ans[ans["эталон"].astype(str).str.lower()
                      .apply(lambda t: any(k in t for k in UNSOLVABLE))]
            cu = uns["correct"].astype("float").dropna()
            r.update(unresolvable_correct_rate=round(cu.mean(), 3) if len(cu) else np.nan)
        out.append(r)
    return pd.DataFrame(out)


# ---------- согласие двух рецензентов ----------
def _pair(dfs: dict) -> pd.DataFrame:
    (n1, a), (n2, b) = list(dfs.items())[:2]
    # Ключи, которые обязаны совпадать между рецензентами — не дублируются при merge
    merge_keys = ["response_id", "type", "task_id", "model_key"]
    missing_a = [k for k in merge_keys if k not in a.columns]
    missing_b = [k for k in merge_keys if k not in b.columns]
    if missing_a or missing_b:
        raise ValueError(
            f"Не хватает колонок для merge: "
            f"рецензент1={missing_a}, рецензент2={missing_b}")
    m = a.merge(b, on=merge_keys, suffixes=("_1", "_2"), how="outer")
    # Проверка: если type или task_id разошлись для одного response_id — это ошибка данных
    for k in ("type", "task_id", "model_key"):
        if f"{k}_1" in m.columns and f"{k}_2" in m.columns:
            diff = m[m[f"{k}_1"] != m[f"{k}_2"]]
            if len(diff):
                raise ValueError(
                    f"Расхождение по {k} для response_id: "
                    f"{diff[['response_id', f'{k}_1', f'{k}_2']].head().to_dict('records')}")
    return m


def agreement_table(dfs: dict) -> pd.DataFrame:
    m = _pair(dfs)
    rows = []
    for typ, col in FIELDS_BY_TYPE:
        g = m[m["type"] == typ]
        x = g[f"{col}_1"].astype("float")
        y = g[f"{col}_2"].astype("float")
        ok = x.notna() & y.notna()
        xi, yi = x[ok].to_numpy(int), y[ok].to_numpy(int)
        if len(xi) == 0:
            continue
        row = {"type": typ, "field": col, "n": len(xi),
               "raw_agreement": round(float((xi == yi).mean()), 3)}
        try:
            row["kappa"] = round(float(cohen_kappa_score(xi, yi)), 3)
        except ValueError:
            row["kappa"] = np.nan          # вырожденный случай: все оценки одинаковы
        if typ == "C":
            try:                            # порядковая шкала 0-2: взвешенная каппа как дополнение
                row["kappa_linear"] = round(float(cohen_kappa_score(xi, yi, weights="linear")), 3)
            except ValueError:
                row["kappa_linear"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def disagreements(dfs: dict) -> pd.DataFrame:
    m = _pair(dfs)
    bad = []
    for typ, col in FIELDS_BY_TYPE:
        g = m[m["type"] == typ]
        d = g[g[f"{col}_1"].notna() & g[f"{col}_2"].notna() &
              (g[f"{col}_1"] != g[f"{col}_2"])]
        for _, r in d.iterrows():
            bad.append({"response_id": r["response_id"], "type": typ,
                        "task_id": r["task_id"], "field": col,
                        "value_1": int(r[f"{col}_1"]), "value_2": int(r[f"{col}_2"]),
                        "comment_1": r.get("comment_1"), "comment_2": r.get("comment_2")})
    return pd.DataFrame(bad)


# ---------- консенсус по рецензентам ----------
def consensus_long(dfs: dict) -> pd.DataFrame:
    long = pd.concat([d.assign(reviewer=k) for k, d in dfs.items()], ignore_index=True)
    key = ["response_id", "model_key", "task_id", "type"]
    agg = {c: "mean" for c in INT_COLS}          # расхождение двух рецензентов даёт 0.5
    agg.update({"ответ": "first", "эталон": "first",
                "method_error_types": "first", "comment": "first"})
    return long.groupby(key, as_index=False).agg(agg)   # словарь позиционно, НЕ **agg


# ---------- сверка ручной оценки F с автопроверкой ----------
def crosscheck_f_auto(cons: pd.DataFrame):
    path = os.path.join(RESULTS_DIR, "eval_bank.csv")
    if not os.path.exists(path):
        return None
    auto = pd.read_csv(path)
    auto = auto[auto.type == "F"][["model", "task_id", "correct"]]
    man = cons[cons.type == "F"][["model_key", "task_id", "correct"]] \
            .rename(columns={"model_key": "model"})
    m = man.merge(auto, on=["model", "task_id"], how="inner", suffixes=("_man", "_auto"))
    m = m.dropna(subset=["correct_man", "correct_auto"])
    if not len(m):
        return None
    agree = float((m["correct_man"].astype(float) == m["correct_auto"].astype(float)).mean())
    return round(agree, 3), len(m)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    paths = sorted({p for mask in ANS_MASKS for p in glob.glob(mask)})
    if not paths:
        raise SystemExit(f"Не найдено ни одного файла ответов по маскам {ANS_MASKS}")
    print(f"Найдено файлов рецензентов: {len(paths)}: {[os.path.basename(p) for p in paths]}")

    mapping = load_mapping()
    dfs = {os.path.splitext(os.path.basename(p))[0]: load_reviewer(p, mapping) for p in paths}

    long = pd.concat([d.assign(reviewer=k) for k, d in dfs.items()], ignore_index=True)
    long.to_csv(os.path.join(RESULTS_DIR, "manual_long.csv"), index=False, encoding='utf-8-sig')

    sums = pd.concat([summarize(d, k) for k, d in dfs.items()], ignore_index=True)
    sums.to_csv(os.path.join(RESULTS_DIR, "manual_summary_per_reviewer.csv"), index=False)
    print("\n--- Метрики по каждому рецензенту ---")
    print(sums.to_string(index=False))

    if len(dfs) >= 2:
        agr = agreement_table(dfs)
        agr.to_csv(os.path.join(RESULTS_DIR, "manual_agreement.csv"), index=False)
        print("\n--- Согласие рецензентов ---")
        print(agr.to_string(index=False))
        low = agr[agr["kappa"] < 0.6]
        if len(low):
            print("ВНИМАНИЕ: каппа < 0.6 для полей: "
                  f"{low[['type', 'field']].to_dict('records')} — "
                  "уточните рубрику и переоцените (требование методички).")
        dis = disagreements(dfs)
        dis.to_csv(os.path.join(RESULTS_DIR, "manual_disagreements.csv"), index=False)
        print(f"\nРасхождений для обсуждения после разслепления: {len(dis)} "
              "(файл manual_disagreements.csv; колонки model_key намеренно нет)")
    else:
        print("\nВторой рецензент ещё не прислан: каппа и консенсус будут посчитаны "
              "автоматически, как только в review/ появится второй файл blind_review_ans*.*")

    cons = consensus_long(dfs)
    cons.to_csv(os.path.join(RESULTS_DIR, "manual_consensus_long.csv"), index=False, encoding='utf-8-sig')
    cs = summarize(cons, "consensus")
    cs.to_csv(os.path.join(RESULTS_DIR, "manual_consensus_summary.csv"), index=False)
    print("\n--- Консенсус-метрики (итог для отчёта) ---")
    print(cs.to_string(index=False))

    # строки для тепловой карты «модель × тип»: C/D/F вручную, A/B/E берите из summary_bank.csv
    heat = []
    for _, r in cs.iterrows():
        val, name = {"C": (r.get("acc_norm"), "mean_score/2"),
                     "D": (r.get("pass_at_1"), "pass@1"),
                     "F": (r.get("correct_rate"), "correct_rate")}.get(r["type"], (None, None))
        if val is not None and not pd.isna(val):
            heat.append({"model": r["model"], "type": r["type"],
                         "metric": name, "value": val, "n": r["n_answered"]})
    heat_df = pd.DataFrame(heat)
    heat_df.to_csv(os.path.join(RESULTS_DIR, "manual_for_heatmap.csv"), index=False)
    print("\n--- Значения для тепловой карты (C/D/F) ---")
    print(heat_df.to_string(index=False))

    chk = crosscheck_f_auto(cons)
    if chk:
        print(f"\nСверка ручной оценки F с автопроверкой check_f: совпадение {chk[0]} "
              f"на {chk[1]} строках (расхождения — повод проверить и чекер, и рецензентов)")


if __name__ == "__main__":
    main()
