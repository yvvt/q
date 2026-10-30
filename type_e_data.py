import json
import os

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]
LABEL = "Machine failure"
# НЕ используем: UDI, Product ID, TWF, HDF, PWF, OSF, RNF (утечка: отказ = следствие этих флагов).


def load_ai4i(path: str = "data/ai4i2020.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS + [LABEL, "Type"] if c not in df.columns]
    if missing:
        raise ValueError(f"В csv не хватает колонок {missing}; проверьте заголовок")
    return df


def row_text(row) -> str:
    return (f"Тип {row['Type']}; температура воздуха {row['Air temperature [K]']} K; "
            f"температура процесса {row['Process temperature [K]']} K; "
            f"скорость вращения {row['Rotational speed [rpm]']} об/мин; "
            f"момент {row['Torque [Nm]']} Н·м; износ инструмента {row['Tool wear [min]']} мин")


def make_split(df: pd.DataFrame, seed: int = 42, n_test: int = 100,
               test_mode: str = "natural",
               save_path: str = "results/type_e_split.json") -> dict:
    """
    100 тестовых строк и пулы примеров 5/20, не пересекающиеся с тестом.
    Примеры стратифицированы по классу: при базовой частоте 3,4% случайные 5 примеров
    почти наверняка однноклассны, и логистическая регрессия с Байесом вырождаются в константу.
    few5 ⊂ few20. Разбиение фиксируется файлом до прогонов.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        with open(save_path, encoding="utf-8") as f:
            return json.load(f)

    rng = np.random.default_rng(seed)
    y = df[LABEL].to_numpy(int)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng.shuffle(pos)
    rng.shuffle(neg)

    if test_mode == "balanced":
        test_idx = np.concatenate([pos[: n_test // 2], neg[: n_test // 2]])
        test_set = set(test_idx.tolist())
        rest_pos = np.array([i for i in pos if i not in test_set])
        rest_neg = np.array([i for i in neg if i not in test_set])
    else:  # natural: естественная доля отказов ~3,4%
        all_idx = np.arange(len(df))
        rng.shuffle(all_idx)
        test_idx = all_idx[:n_test]
        test_set = set(test_idx.tolist())
        rest_pos = np.array([i for i in pos if i not in test_set])
        rest_neg = np.array([i for i in neg if i not in test_set])

    few20 = np.concatenate([rest_pos[:10], rest_neg[:10]])
    few5 = np.concatenate([rest_pos[:2], rest_neg[:3]])

    split = {
        "seed": seed,
        "test_mode": test_mode,
        "test_idx": [int(i) for i in test_idx],
        "few5": [int(i) for i in few5],
        "few20": [int(i) for i in few20],
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(split, f)

    n_pos_test = int(y[test_idx].sum())
    print(f"Тест: {len(test_idx)} строк, из них класс 1: {n_pos_test}")
    if n_pos_test < 10:
        print("ПРЕДУПРЕЖДЕНИЕ: объектов класса 1 в тесте мало — F1 редкого класса "
              "будет иметь очень широкий доверительный интервал. Оставьте как основной "
              "вариант и добавьте test_mode='balanced' как анализ чувствительности.")
    return split


def examples_for(df: pd.DataFrame, split: dict, n: int) -> list[tuple[str, int]]:
    idx = split.get(f"few{n}", [])
    return [(row_text(df.loc[i]), int(df.loc[i, LABEL])) for i in idx]


def classical_baselines(df: pd.DataFrame, split: dict, n_shots: int) -> dict | None:
    """LR и наивный Байес на тех же 5/20 примерах, оценка на тех же 100 тестовых строках."""
    if n_shots == 0:
        return None
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, balanced_accuracy_score

    few = split[f"few{n_shots}"]
    X_tr = df.loc[few, FEATURE_COLS].to_numpy(float)
    y_tr = df.loc[few, LABEL].to_numpy(int)
    X_te = df.loc[split["test_idx"], FEATURE_COLS].to_numpy(float)
    y_te = df.loc[split["test_idx"], LABEL].to_numpy(int)

    out = {}
    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "naive_bayes": GaussianNB(),
    }
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        out[name] = {
            "accuracy": float((pred == y_te).mean()),
            "f1_rare": float(f1_score(y_te, pred, pos_label=1, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
        }
    return out
