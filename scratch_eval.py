import numpy as np
import pandas as pd

from src import config as C
from src.data_loader import clean_charging, load_raw_charging
from src.features import add_lag_features, add_time_features, build_station_load_curve


def build_features(df):
    f = pd.DataFrame(index=df.index)
    f["hour"] = df["hour"].astype(float)
    f["minute_of_day"] = df["minute_of_day"].astype(float)
    f["weekday"] = df["weekday"].astype(float)
    f["is_weekend"] = df["is_weekend"].astype(float)
    f["hour_sin"] = df["hour_sin"].astype(float)
    f["hour_cos"] = df["hour_cos"].astype(float)
    f["lag1"] = df["power_kw_lag1"]
    f["lag4"] = df["power_kw_lag4"]
    f["lag24"] = df["power_kw_lag24"]
    f["lag96"] = df["power_kw_lag96"]
    f["roll24"] = df["power_kw_rolling24"]
    f["occ1"] = df["active_piles_lag1"]
    f["occ4"] = df["active_piles_lag4"]
    f["occ24"] = df["active_piles_lag24"]
    f["occ96"] = df["active_piles_lag96"]
    f["occ_roll"] = df["active_piles_rolling24"]
    f["delta14"] = df["power_kw_lag1"] - df["power_kw_lag4"]
    f["delta124"] = df["power_kw_lag1"] - df["power_kw_lag24"]
    f["delta496"] = df["power_kw_lag4"] - df["power_kw_lag96"]
    f["ppp1"] = df["power_kw_lag1"] / (df["active_piles_lag1"] + 1)
    f["ppp4"] = df["power_kw_lag4"] / (df["active_piles_lag4"] + 1)
    f["ppp24"] = df["power_kw_lag24"] / (df["active_piles_lag24"] + 1)
    f["interact1"] = df["power_kw_lag1"] * df["active_piles_lag1"]
    return f


def build_features_ext(df):
    f = build_features(df)
    if "price_yuan_per_kwh" in df.columns:
        f["price"] = df["price_yuan_per_kwh"].astype(float)
    if "temperature_c" in df.columns:
        f["temperature"] = df["temperature_c"].astype(float)
    return f


def fit_ridge(x, y, lam=1.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mu = x.mean(axis=0)
    sd = x.std(axis=0) + 1e-9
    z = (x - mu) / sd
    a = np.column_stack([z, np.ones(len(z))])
    ata = a.T @ a + lam * np.eye(a.shape[1])
    coef = np.linalg.solve(ata, a.T @ y)
    w = coef[:-1] / sd
    b = coef[-1] - (w * mu).sum()
    return w, b


def predict_ridge(wb, x):
    x = np.asarray(x, dtype=float)
    return x @ wb[0] + wb[1]


def choose_lam(xtr, ytr, holdout=48, lams=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)):
    best = None
    for lam in lams:
        wb = fit_ridge(xtr[:-holdout], ytr[:-holdout], lam)
        err = np.abs(ytr[-holdout:] - predict_ridge(wb, xtr[-holdout:])).mean()
        if best is None or err < best[0]:
            best = (err, lam)
    return best[1]


class _Node:
    __slots__ = ("value", "j", "thr", "left", "right")

    def __init__(self):
        self.value = 0.0
        self.j = -1
        self.thr = 0.0
        self.left = None
        self.right = None


def _fit_node(x, y, depth, min_leaf):
    node = _Node()
    node.value = float(y.mean())
    if depth <= 0 or len(y) < 2 * min_leaf:
        return node
    n, p = x.shape
    total = float(y.sum())
    best_score = -1e18
    best_j = -1
    best_thr = 0.0
    for j in range(p):
        order = np.argsort(x[:, j], kind="stable")
        xs = x[order, j]
        ys = y[order]
        cum = np.cumsum(ys)
        for i in range(min_leaf, len(ys) - min_leaf + 1):
            if xs[i - 1] == xs[i]:
                continue
            nl = i
            left_sum = cum[i - 1]
            right_sum = total - left_sum
            score = left_sum * left_sum / nl + right_sum * right_sum / (len(ys) - nl)
            if score > best_score:
                best_score = score
                best_j = j
                best_thr = (xs[i - 1] + xs[i]) / 2.0
    if best_j < 0:
        return node
    mask = x[:, best_j] <= best_thr
    node.j = best_j
    node.thr = best_thr
    node.left = _fit_node(x[mask], y[mask], depth - 1, min_leaf)
    node.right = _fit_node(x[~mask], y[~mask], depth - 1, min_leaf)
    return node


def _predict_node(node, x):
    out = np.full(len(x), node.value)
    if node.j >= 0:
        mask = x[:, node.j] <= node.thr
        out[mask] = _predict_node(node.left, x[mask])
        out[~mask] = _predict_node(node.right, x[~mask])
    return out


def gbm_predict(xtr, ytr, xva, n_trees=400, depth=2, lr=0.05, min_leaf=8, subsample=0.8, early_rounds=40, seed=0):
    rng = np.random.default_rng(seed)
    hold = int(len(ytr) * 0.9)
    xtr_fit, ytr_fit = xtr[:hold], ytr[:hold]
    xtr_es, ytr_es = xtr[hold:], ytr[hold:]
    pred = np.full(len(ytr_fit), ytr_fit.mean())
    pred_es = np.full(len(ytr_es), ytr_fit.mean())
    trees = []
    best_es = np.inf
    best_trees = None
    n = len(ytr_fit)
    for it in range(n_trees):
        resid = ytr_fit - pred
        m = max(int(n * subsample), 2 * min_leaf)
        idx = rng.choice(n, size=m, replace=False)
        tree = _fit_node(xtr_fit[idx], resid[idx], depth, min_leaf)
        trees.append(tree)
        pred += lr * _predict_node(tree, xtr_fit)
        pred_es += lr * _predict_node(tree, xtr_es)
        es = float(np.abs(ytr_es - pred_es).mean())
        if es < best_es:
            best_es = es
            best_trees = list(trees)
            best_iter = it + 1
        if it - best_iter >= early_rounds:
            break
    out = np.full(len(xva), ytr.mean())
    for tree in best_trees:
        out += lr * _predict_node(tree, xva)
    return out


def rf_predict(xtr, ytr, xva, n_trees=300, depth=5, min_leaf=4, seed=0):
    rng = np.random.default_rng(seed)
    n = len(ytr)
    preds = np.zeros((n_trees, len(xva)))
    for it in range(n_trees):
        idx = rng.choice(n, size=n, replace=True)
        tree = _fit_node(xtr[idx], ytr[idx], depth, min_leaf)
        preds[it] = _predict_node(tree, xva)
    return preds.mean(axis=0)


def standardize_fit(x):
    x = np.asarray(x, dtype=float)
    mu = x.mean(axis=0)
    sd = x.std(axis=0) + 1e-9
    return (x - mu) / sd, mu, sd


def standardize_apply(x, mu, sd):
    return (np.asarray(x, dtype=float) - mu) / sd


def main():
    sessions = clean_charging(load_raw_charging())
    curve = build_station_load_curve(sessions, C.STEP_MINUTES)
    curve = add_time_features(curve, "timestamp")

    for station, group in curve.groupby(C.STATION_COL):
        group = add_lag_features(group, [C.LOAD_COL, C.OCCUPANCY_COL])
        group = group.dropna(subset=["power_kw_lag96"]).reset_index(drop=True)
        split = int(len(group) * C.TRAIN_RATIO)
        tr, va = group.iloc[:split], group.iloc[split:]
        ytr, yva = tr[C.LOAD_COL].values, va[C.LOAD_COL].values
        byh = tr.groupby("hour")[C.LOAD_COL].mean()
        naive = va["hour"].map(lambda h: byh.get(h, byh.mean())).values

        xtr = build_features(tr)
        xva = build_features(va)
        xtr_e = build_features_ext(tr)
        xva_e = build_features_ext(va)
        lam = choose_lam(xtr.values, ytr)
        wb = fit_ridge(xtr.values, ytr, lam)
        pred = predict_ridge(wb, xva.values)
        wb_e = fit_ridge(xtr_e.values, ytr, choose_lam(xtr_e.values, ytr))
        pred_e = predict_ridge(wb_e, xva_e.values)

        ztr, mu, sd = standardize_fit(xtr.values)
        zva = standardize_apply(xva.values, mu, sd)
        gbm = gbm_predict(ztr, ytr, zva)
        rf = rf_predict(ztr, ytr, zva)

        print(
            station,
            "rows",
            len(group),
            "naive",
            round(float(np.abs(yva - naive).mean()), 2),
            "ridge_cv",
            round(float(np.abs(yva - pred).mean()), 2),
            "ridge_ext",
            round(float(np.abs(yva - pred_e).mean()), 2),
            "gbm",
            round(float(np.abs(yva - gbm).mean()), 2),
            "rf",
            round(float(np.abs(yva - rf).mean()), 2),
            "lam",
            lam,
        )


if __name__ == "__main__":
    main()
