"""严格时间 holdout 评估 office1，避免加载全量预训练权重造成的数据泄露。

与 public_self_eval.py 的日期 80/20 时间划分一致，只用训练切片现场训练
LightGBM，再在验证切片上算 office1 的 site-only / pooled / blend 指标。
"""
from __future__ import annotations

import json
import pathlib
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
SITES = ("caltech", "jpl", "office1")
SITE_INDEX = {site: i for i, site in enumerate(SITES)}


def features_base(hour, weekday, pmax):
    hour = np.asarray(hour, dtype=float) % 24.0
    weekday = np.asarray(weekday, dtype=float)
    pmax = np.asarray(pmax, dtype=float)
    return np.column_stack(
        [
            np.sin(hour * np.pi / 12.0),
            np.cos(hour * np.pi / 12.0),
            np.sin(hour * np.pi / 6.0),
            np.cos(hour * np.pi / 6.0),
            weekday,
            (weekday >= 5.0).astype(float),
            pmax,
        ]
    )


def pooled_features(hour, weekday, pmax, site):
    base = features_base(hour, weekday, pmax)
    onehot = np.zeros((len(base), len(SITES)), dtype=float)
    onehot[:, SITE_INDEX[site]] = 1.0
    return np.column_stack([base, onehot])


def fit(x, y, leaves):
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=leaves,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=20261008,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(x, y)
    return model


def split_day(frame, ratio):
    days = sorted(frame.day.unique())
    cut = days[max(1, int(len(days) * ratio)) - 1]
    return frame[frame.day <= cut].copy(), frame[frame.day > cut].copy()


def evaluate(ratio):
    training = {}
    for site in SITES:
        frame = pd.DataFrame(json.loads((ROOT / "data" / f"training_{site}.json").read_text(encoding="utf-8")))
        frame["day"] = pd.to_datetime(frame.day)
        training[site] = frame

    office_fit, office_val = split_day(training["office1"], ratio)
    train_slices = {}
    for site in SITES:
        train_slices[site], _ = split_day(training[site], ratio)

    results = {}
    for target, leaves in (("demand_kwh", 31), ("dwell_hours", 15)):
        site_model = fit(
            features_base(office_fit.hour, office_fit.weekday, office_fit.charger_limit_kw),
            office_fit[target].to_numpy(),
            leaves,
        )

        xs, ys = [], []
        for site in SITES:
            frame = train_slices[site]
            xs.append(pooled_features(frame.hour, frame.weekday, frame.charger_limit_kw, site))
            ys.append(frame[target].to_numpy())
        pooled = fit(np.vstack(xs), np.concatenate(ys), leaves)

        xb_val = features_base(office_val.hour, office_val.weekday, office_val.charger_limit_kw)
        xp_val = pooled_features(office_val.hour, office_val.weekday, office_val.charger_limit_kw, "office1")
        truth = office_val[target].to_numpy()
        site_pred = site_model.predict(xb_val)
        pooled_pred = pooled.predict(xp_val)

        best_mae, best_alpha = None, None
        for alpha in np.arange(0.0, 1.01, 0.01):
            mixed = alpha * site_pred + (1.0 - alpha) * pooled_pred
            mae = float(np.abs(truth - mixed).mean())
            if best_mae is None or mae < best_mae:
                best_mae, best_alpha = mae, float(alpha)

        results[target] = {
            "val_rows": len(office_val),
            "site_only_mae": float(np.abs(truth - site_pred).mean()),
            "pooled_mae": float(np.abs(truth - pooled_pred).mean()),
            "best_blend_mae": best_mae,
            "best_alpha": best_alpha,
        }
    return results


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    for ratio in (0.8, 0.85):
        print(f"office1 严格时间 holdout（训练前 {ratio:.0%} 天）")
        print("-" * 46)
        results = evaluate(ratio)
        for target in ("demand_kwh", "dwell_hours"):
            r = results[target]
            print(
                f"{target}: site-only {r['site_only_mae']:.4f} | "
                f"pooled {r['pooled_mae']:.4f} | "
                f"best blend {r['best_blend_mae']:.4f} (alpha={r['best_alpha']:.2f})"
            )
        print()


if __name__ == "__main__":
    main()
