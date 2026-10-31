"""离线训练 module_2_prediction 的 LightGBM 权重。

使用公开三站训练标签，在评测沙箱外拟合 demand/dwell 两个 L1(MAE) 回归器，
并把 booster 存成 LightGBM 文本格式。沙箱内的 module_2_prediction.py 只加载
这些权重，不重新训练；加载失败时才退回现场拟合。
"""
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
SITES = ("caltech", "jpl", "office1")


def make_features(hour, weekday, pmax, requested):
    hour = np.asarray(hour, dtype=float)
    weekday = np.asarray(weekday, dtype=float)
    pmax = np.asarray(pmax, dtype=float)
    requested = np.asarray(requested, dtype=float)
    missing = np.isnan(requested).astype(float)
    return np.column_stack(
        [
            np.sin(hour * np.pi / 12.0),
            np.cos(hour * np.pi / 12.0),
            np.sin(weekday * 2.0 * np.pi / 7.0),
            np.cos(weekday * 2.0 * np.pi / 7.0),
            weekday,
            pmax,
            requested,
            missing,
        ]
    )


def fit_booster(x, y, num_leaves):
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=num_leaves,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=20261008,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(x, y)
    return model.booster_


def load_training():
    training = {}
    for site in SITES:
        rows = json.load(open(ROOT / "data" / f"training_{site}.json", encoding="utf-8"))
        frame = pd.DataFrame(rows)
        frame["day"] = pd.to_datetime(frame["day"])
        training[site] = frame
    return training


def main():
    training = load_training()
    WEIGHTS_DIR.mkdir(exist_ok=True)
    started = time.time()
    for site in SITES:
        frame = training[site]
        requested = (
            frame["last_requested_energy_kwh"]
            if "last_requested_energy_kwh" in frame.columns
            else np.full(len(frame), np.nan)
        )
        x = make_features(
            frame.hour, frame.weekday, frame.charger_limit_kw, requested
        )
        demand = fit_booster(x, frame.demand_kwh, num_leaves=31)
        dwell = fit_booster(x, frame.dwell_hours, num_leaves=15)
        demand.save_model(WEIGHTS_DIR / f"demand_{site}.txt")
        dwell.save_model(WEIGHTS_DIR / f"dwell_{site}.txt")
        print(site, "rows", len(frame), "saved")
    print("weights dir:", WEIGHTS_DIR)
    print("fit seconds:", round(time.time() - started, 2))


if __name__ == "__main__":
    main()
