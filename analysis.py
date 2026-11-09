"""Дополнительный анализ ЛР3: ДИ, МакНемар, отказы, гипотезы H3/H5/H12,
стоимость, графики, кандидаты ошибок. Запуск: python analysis.py"""
import itertools
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import f1_score, balanced_accuracy_score
from bank import load_bank
from config import MODELS, RESULTS_DIR, BANK_PATH
from evaluate import UNSOLVABLE, bootstrap_ci, load_logs

OUT = os.path.join(RESULTS_DIR, "analysis")
FIG = "figures"
# РУБ за 1000 токенов по официальным тарифам; впишите и укажите дату прайса в отчёте
TARIFFS = {
    "gigachat":   0.0,     # бесплатный лимит для физлиц; укажите 0 и в отчёте поясните
    "yandexgpt":  0.35,    # yandexgpt-5-lite, уточните по ai.cloud.yandex.ru
    "foreign":    0.15,    # gpt-oss-20b/latest через Yandex AI Studio, уточните
    "local_qwen": None,    # локально — тарифа нет; в отчёте укажите время и память
}
CLASSICAL_PREDS = os.path.join(RESULTS_DIR, "classical_preds.csv")  # classifier,n_shots,row_idx,y_pred


def load_bank_df() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RESULTS_DIR, "eval_bank.csv"))
    dup = df[df.duplicated(["model", "task_id"], keep=False)]
    if len(dup):
        print(f"ПРЕДУПРЕЖДЕНИЕ: дубли (model, task_id): {len(dup)} строк; "
              f"оставлен первый прогон (согласовано с prepare_blind_review): "
              f"{dup[['model', 'task_id']].drop_duplicates().to_dict('records')}")
        df = df.drop_duplicates(["model", "task_id"], keep="first")
    return df


def ci_table(bank: pd.DataFrame) -> pd.DataFrame:
    """Точность с 95% ДИ бутстрепом по каждой модели и типу."""
    rows = []
    for (m, t), g in bank.groupby(["model", "type"]):
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: correct может быть bool/str/float, приводим к numeric
        c = pd.to_numeric(g["correct"], errors="coerce").dropna()
        if not len(c):
            continue
        c_bool = c.astype(bool)
        acc, lo, hi = bootstrap_ci(c_bool)
        rows.append({"model": m, "type": t, "n": len(c),
                     "acc": round(acc, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    for m, g in bank.groupby("model"):
        c = pd.to_numeric(g["correct"], errors="coerce").dropna()
        if not len(c):
            continue
        c_bool = c.astype(bool)
        acc, lo, hi = bootstrap_ci(c_bool)
        rows.append({"model": m, "type": "ALL", "n": len(c),
                     "acc": round(acc, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    return pd.DataFrame(rows)


def mcnemar_pairs(bank: pd.DataFrame, types=None) -> pd.DataFrame:
    sub = bank[bank["correct"].notna()].copy()
    sub["correct"] = pd.to_numeric(sub["correct"], errors="coerce")
    if types:
        sub = sub[sub["type"].isin(types)]
    piv = sub.drop_duplicates(["model", "task_id"]).pivot(
        index="task_id", columns="model", values="correct").astype(bool)
    rows = []
    for a, b in itertools.combinations(sorted(piv.columns), 2):
        x = piv[[a, b]].dropna()
        ya, yb = x[a].to_numpy(), x[b].to_numpy()
        tab = [[int(((ya) & (yb)).sum()), int(((ya) & (~yb)).sum())],
               [int(((~ya) & (yb)).sum()), int(((~ya) & (~yb)).sum())]]
        res = mcnemar(tab, exact=True, correction=False)
        rows.append({"model_a": a, "model_b": b, "n": len(x),
                     "discordant": tab[0][1] + tab[1][0], "p_value": round(float(res.pvalue), 4)})
    return pd.DataFrame(rows)


def refusal_table(bank: pd.DataFrame) -> pd.DataFrame:
    g = bank.groupby(["model", "type"]).agg(n=("answered", "size"), n_ans=("answered", "sum"))
    g["refusal_rate"] = round(1 - g["n_ans"] / g["n"], 3)
    return g.reset_index()


def h3_unresolvable(bank: pd.DataFrame) -> pd.DataFrame:
    """H3: в нерешаемых задачах модели чаще додумывают число, чем отказываются."""
    tasks = {t["id"]: t for t in load_bank(BANK_PATH)}
    uns = {i for i, t in tasks.items()
           if t["тип"].upper() == "F" and any(k in str(t["эталон"]).lower() for k in UNSOLVABLE)}
    f = bank[(bank["type"] == "F") & bank["task_id"].isin(uns) & bank["answered"]
             & bank["correct"].notna()].copy()
    f["correct"] = pd.to_numeric(f["correct"], errors="coerce")
    rows = []
    for m, g in f.groupby("model"):
        fab = int((g["correct"] == False).sum())          # дали число вместо отказа
        n = len(g)
        p = binomtest(fab, n, p=0.5, alternative="greater").pvalue if n else np.nan
        rows.append({"model": m, "n_unresolvable": n, "fabricated": fab,
                     "fabrication_rate": round(fab / n, 3) if n else np.nan,
                     "binom_p_gt_0.5": round(float(p), 4) if n else np.nan})
    tot_fab = int((f["correct"] == False).sum()); tot_n = len(f)
    rows.append({"model": "ALL", "n_unresolvable": tot_n, "fabricated": tot_fab,
                 "fabrication_rate": round(tot_fab / tot_n, 3) if tot_n else np.nan,
                 "binom_p_gt_0.5": round(float(binomtest(tot_fab, tot_n, p=0.5,
                            alternative="greater").pvalue), 4) if tot_n else np.nan})
    return pd.DataFrame(rows)


def h5_gap(ci: pd.DataFrame, bank: pd.DataFrame) -> dict:
    """H5: локальная модель отстаёт сильнее на A, чем на B."""
    # Защита: если типов A или B нет в таблице, возвращаем NaN
    ab_types = ["A", "B"]
    available = [t for t in ab_types if t in ci["type"].values]
    if not available:
        return {"gap_A": np.nan, "gap_B": np.nan, 
                "mcnemar_A_local_vs_others": [], "mcnemar_B_local_vs_others": []}
    
    acc = ci[ci["type"].isin(ab_types)].pivot(index="model", columns="type", values="acc")
    big = [m for m in acc.index if m != "local_qwen"]
    out = {}
    for t in ab_types:
        if t not in acc.columns or "local_qwen" not in acc.index:
            out[f"gap_{t}"] = np.nan
            continue
        try:
            max_big = acc.loc[big, t].max() if len(big) else np.nan
            loc_val = acc.loc["local_qwen", t]
            out[f"gap_{t}"] = round(float(max_big - loc_val), 3) if not np.isnan(max_big) else np.nan
        except Exception:
            out[f"gap_{t}"] = np.nan
    
    mn = mcnemar_pairs(bank, types=["A"])
    mn = mn[(mn.model_a == "local_qwen") | (mn.model_b == "local_qwen")]
    out["mcnemar_A_local_vs_others"] = mn.to_dict("records")
    mnB = mcnemar_pairs(bank, types=["B"])
    mnB = mnB[(mnB.model_a == "local_qwen") | (mnB.model_b == "local_qwen")]
    out["mcnemar_B_local_vs_others"] = mnB.to_dict("records")
    return out


def h12_table(bank_e: pd.DataFrame) -> pd.DataFrame:
    """H12: few-shot LLM против LR/NB на тех же примерах (основные метрики — F1 редкого и bal.acc)."""
    rows = []
    for n, g in bank_e[bank_e["y_pred"].notna()].groupby("n_shots"):
        for m, gg in g.groupby("model"):
            yt, yp = gg["y_true"].to_numpy(int), gg["y_pred"].to_numpy(int)
            rows.append({"who": m, "n_shots": n, "n": len(gg),
                         "acc": round(float((yp == yt).mean()), 3),
                         "f1_rare": round(float(f1_score(yt, yp, pos_label=1, zero_division=0)), 3),
                         "bal_acc": round(float(balanced_accuracy_score(yt, yp)), 3)})
    df = pd.DataFrame(rows)
    if os.path.exists(CLASSICAL_PREDS):
        cp = pd.read_csv(CLASSICAL_PREDS)
        extra = []
        for n, g in cp.groupby("n_shots"):
            for c, gg in g.groupby("classifier"):
                yt, yp = gg["y_true"].to_numpy(int), gg["y_pred"].to_numpy(int)
                extra.append({"who": c, "n_shots": n, "n": len(gg),
                              "acc": round(float((yp == yt).mean()), 3),
                              "f1_rare": round(float(f1_score(yt, yp, pos_label=1, zero_division=0)), 3),
                              "bal_acc": round(float(balanced_accuracy_score(yt, yp)), 3)})
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    else:
        print(f"ПРЕДУПРЕЖДЕНИЕ: нет {CLASSICAL_PREDS} — сравнение H12 описательное, "
              f"парные тесты пропущены (нужен type_e_data.py для построчных предсказаний LR/NB)")
    return df.sort_values(["n_shots", "who"])


def cost_table(bank: pd.DataFrame, e_df: pd.DataFrame) -> pd.DataFrame:
    bank = bank.copy()
    # Принудительно приводим correct к numeric перед суммированием
    bank["correct"] = pd.to_numeric(bank["correct"], errors="coerce").fillna(0)
    
    tok = pd.concat([bank.groupby("model")["total_tokens"].sum(),
                     e_df.groupby("model")["total_tokens"].sum()]).groupby(level=0).sum()
    lat = pd.concat([bank.groupby("model")["elapsed_seconds"].mean(),
                     e_df.groupby("model")["elapsed_seconds"].mean()]).groupby(level=0).mean()
    ncor = bank[bank["correct"].notna()].groupby("model")["correct"].sum()
    df = pd.DataFrame({"total_tokens": tok, "mean_latency_s": lat.round(1),
                       "n_correct": ncor})
    df["cost_rub"] = [round(tok[m] / 1000 * TARIFFS[m], 2) if TARIFFS.get(m) else np.nan
                      for m in df.index]
    # Защита от деления на ноль и object dtype
    df["rub_per_correct"] = pd.to_numeric(df["cost_rub"] / df["n_correct"], errors="coerce").round(2)
    if df["cost_rub"].isna().any():
        print("ПРЕДУПРЕЖДЕНИЕ: тарифы не заполнены (TARIFFS) — стоимостные колонки NaN; "
              "для local_qwen укажите в отчёте время и память на вашем железе вместо тарифа")
    return df


def make_figures(ci: pd.DataFrame, bank: pd.DataFrame, e_df: pd.DataFrame,
                 h3: pd.DataFrame, h12: pd.DataFrame, cost: pd.DataFrame) -> None:
    os.makedirs(FIG, exist_ok=True)
    # Тепловая карта: только если есть данные
    if len(ci[ci["type"] != "ALL"]) > 0:
        piv = ci[ci["type"] != "ALL"].pivot(index="model", columns="type", values="acc")
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(piv.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(range(piv.shape[1]), piv.columns)
        ax.set_yticks(range(piv.shape[0]), piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.iat[i, j]:.2f}", ha="center", va="center", fontsize=9)
        ax.set_title("Точность: модель × тип задачи")
        fig.colorbar(im)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "heatmap.png"), dpi=150); plt.close(fig)

    allr = ci[ci["type"] == "ALL"].sort_values("acc")
    if len(allr) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(allr["acc"], range(len(allr)),
                    xerr=[allr["acc"] - allr["ci_lo"], allr["ci_hi"] - allr["acc"]],
                    fmt="o", capsize=4)
        ax.set_yticks(range(len(allr)), allr["model"])
        ax.set_xlabel("точность (A–F) с 95% ДИ бутстрепом")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "accuracy_ci.png"), dpi=150); plt.close(fig)

    if cost["cost_rub"].notna().any():
        c = cost.dropna(subset=["cost_rub"]).join(
            ci[ci["type"] == "ALL"].set_index("model")[["acc"]])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(c["rub_per_correct"], c["acc"])
        for m, r in c.iterrows():
            ax.annotate(m, (r["rub_per_correct"], r["acc"]), fontsize=8)
        ax.set_xlabel("рублей за один правильный ответ"); ax.set_ylabel("точность")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "cost_accuracy.png"), dpi=150); plt.close(fig)

    h3p = h3[h3["model"] != "ALL"]
    if len(h3p) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(h3p["model"], h3p["fabrication_rate"])
        ax.axhline(0.5, ls="--", c="gray")
        for i, r in h3p.iterrows():
            ax.text(i, r["fabrication_rate"] + 0.02, f"n={r['n_unresolvable']}", ha="center", fontsize=8)
        ax.set_ylabel("доля задач, где вместо отказа дано число")
        ax.set_title("H3: нерешаемые задачи")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "h3.png"), dpi=150); plt.close(fig)

    ab = ci[ci["type"].isin(["A", "B"])].pivot(index="model", columns="type", values="acc")
    if len(ab) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(ab)); w = 0.35
        if "A" in ab.columns:
            ax.bar(x - w / 2, ab["A"], w, label="A (расчётные)")
        if "B" in ab.columns:
            ax.bar(x + w / 2, ab["B"], w, label="B (концептуальные)")
        ax.set_xticks(x, ab.index); ax.legend(); ax.set_ylim(0, 1.05)
        ax.set_title("H5: разрыв локальной модели по типам")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "h5.png"), dpi=150); plt.close(fig)

    if len(h12) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, met in zip(axes, ["f1_rare", "bal_acc"]):
            for m, g in h12.groupby("who"):
                g = g.sort_values("n_shots")
                if m in ("logreg", "naive_bayes"):
                    ax.plot(g["n_shots"], g[met], "s--", label=m)
                else:
                    ax.plot(g["n_shots"], g[met], "o-", label=m)
            ax.set_xlabel("примеров в промпте"); ax.set_title(met)
        axes[0].legend(fontsize=8); axes[0].set_ylabel("метрика")
        fig.suptitle("H12: LLM против классических базлайнов")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "h12.png"), dpi=150); plt.close(fig)


def error_candidates(bank: pd.DataFrame) -> pd.DataFrame:
    """Кандидаты для разбора пяти ошибок: неверные отвеченные задачи с цитатой."""
    first = {}
    for mk in MODELS:
        for e in load_logs(mk):
            k = (mk, e.get("task_id"))
            if e.get("task_type") == "E" or k in first:
                continue
            first[k] = e.get("answer", "")
    df = bank[(bank["correct"] == False) & bank["answered"]].copy()
    df["answer"] = [first.get((m, t), "")[:600] for m, t in zip(df["model"], df["task_id"])]
    return df[["model", "task_id", "type", "parsed", "answer"]].sort_values(["type", "model"])


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    bank = load_bank_df()
    e_df = pd.read_csv(os.path.join(RESULTS_DIR, "eval_type_e.csv"))

    ci = ci_table(bank);            ci.to_csv(os.path.join(OUT, "accuracy_ci.csv"), index=False)
    mn = mcnemar_pairs(bank);       mn.to_csv(os.path.join(OUT, "mcnemar_all_types.csv"), index=False)
    ref = refusal_table(bank);      ref.to_csv(os.path.join(OUT, "refusal_rates.csv"), index=False)
    h3 = h3_unresolvable(bank);     h3.to_csv(os.path.join(OUT, "h3_unresolvable.csv"), index=False)
    h5 = h5_gap(ci, bank)
    h12 = h12_table(e_df);          h12.to_csv(os.path.join(OUT, "h12_fewshot.csv"), index=False)
    cost = cost_table(bank, e_df);  cost.to_csv(os.path.join(OUT, "cost_speed.csv"))
    err = error_candidates(bank);   err.to_csv(os.path.join(OUT, "error_candidates.csv"), index=False)
    make_figures(ci, bank, e_df, h3, h12, cost)

    print("=== Точность с 95% ДИ ===");            print(ci.to_string(index=False))
    print("\n=== МакНемар (все оценённые типы) ==="); print(mn.to_string(index=False))
    print("\n=== Отказы ===");                     print(ref.to_string(index=False))
    print("\n=== H3 ===");                         print(h3.to_string(index=False))
    print("\n=== H5 ===");                         print(h5["gap_A"], h5["gap_B"])
    print("\n=== H12 ===");                        print(h12.to_string(index=False))
    print("\n=== Стоимость/скорость ===");        print(cost.to_string())
    print(f"\nКандидаты ошибок: {os.path.join(OUT, 'error_candidates.csv')} "
          f"(выберите 5 с цитатами вручную); графики: {FIG}/")
    print("Вердикты формулируйте по правилам, зафиксированным в отчёте: "
          "H3 — fabrication_rate > 0.5 в агрегате; H5 — gap_A > gap_B при значимом МакНемаре на A "
          "(иначе 'направленно, незначимо'); H12 — по основным метрикам f1_rare/bal_acc, не по accuracy.")


if __name__ == "__main__":
    main()
