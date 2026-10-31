import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def load():
    f = pd.DataFrame(json.load(open("data/training_office1.json", encoding="utf-8")))
    f["day"] = pd.to_datetime(f["day"])
    return f


def split(f, r=0.8):
    days = sorted(f.day.unique())
    cut = days[max(1, int(len(days) * r)) - 1]
    return f[f.day <= cut].copy(), f[f.day > cut].copy()


def base(h, w, p):
    h = np.asarray(h, float) % 24.0
    w = np.asarray(w, float)
    p = np.asarray(p, float)
    return np.column_stack([
        np.sin(h * np.pi / 12.0), np.cos(h * np.pi / 12.0),
        np.sin(h * np.pi / 6.0), np.cos(h * np.pi / 6.0),
        w, (w >= 5.0).astype(float), p,
    ])


def fit_q(x, y):
    m = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=200,
                                      max_leaf_nodes=15, l2_regularization=1.0, random_state=0)
    m.fit(x, y)
    return m


def main():
    f = load()
    tr, va = split(f)
    xt = base(tr.hour, tr.weekday, tr.charger_limit_kw)
    xv = base(va.hour, va.weekday, va.charger_limit_kw)
    y = tr.demand_kwh.to_numpy()
    truth = va.demand_kwh.to_numpy()

    raw = fit_q(xt, y).predict(xv)
    logp = np.expm1(fit_q(xt, np.log1p(y)).predict(xv))
    sq = fit_q(xt, np.sqrt(y)).predict(xv) ** 2
    print("demand raw MAE:", round(float(np.abs(truth - raw).mean()), 4))
    print("demand log1p MAE:", round(float(np.abs(truth - logp).mean()), 4))
    print("demand sqrt MAE:", round(float(np.abs(truth - sq).mean()), 4))

    # KNN median 与 log1p 的 alpha 集成
    xtn = xt / (np.linalg.norm(xt, axis=1, keepdims=True) + 1e-9)
    xvn = xv / (np.linalg.norm(xv, axis=1, keepdims=True) + 1e-9)
    knn = []
    for q in xvn:
        sim = xtn @ q
        idx = np.argsort(-sim)[:40]
        w = np.maximum(sim[idx], 0.0)
        if w.sum() <= 1e-12:
            w = np.ones(40)
        v = y[idx]
        o = np.argsort(v)
        c = np.cumsum(w[o])
        knn.append(float(v[o][np.searchsorted(c, 0.5 * c[-1])]))
    knn = np.array(knn)
    for name, m in (("raw", raw), ("log1p", logp), ("sqrt", sq)):
        for a in (0.3, 0.5):
            p = (1 - a) * m + a * knn
            print(f"demand {name}+KNN a={a} MAE:", round(float(np.abs(truth - p).mean()), 4))


if __name__ == "__main__":
    main()
