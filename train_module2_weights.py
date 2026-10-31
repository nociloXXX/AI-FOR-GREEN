"""离线训练 module_2_prediction 的 LightGBM 权重。

caltech/jpl 每个站点训练单目标 demand/dwell 模型：
- fallback：只含 hour/weekday/charger_limit，在申报需求缺失样本上训练；
- with：含 last_requested_energy_kwh，只在正有限申报值样本上训练。

office1 不产权重：小样本且无申报值，由 module_2_prediction 在沙箱内用
HistGradientBoosting quantile q50 现场拟合。
"""
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
SITES = ("caltech", "jpl")


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


def main():
    WEIGHTS_DIR.mkdir(exist_ok=True)
    started = time.time()
    for site in SITES:
        rows = json.load(open(ROOT / "data" / f"training_{site}.json", encoding="utf-8"))
        frame = pd.DataFrame(rows)
        requested = (
            np.asarray(frame["last_requested_energy_kwh"], dtype=float)
            if "last_requested_energy_kwh" in frame.columns
            else np.full(len(frame), np.nan)
        )
        valid = np.isfinite(requested) & (requested > 0.0)
        xb = features_base(frame.hour, frame.weekday, frame.charger_limit_kw)

        fallback_mask = ~valid if valid.any() and (~valid).sum() >= 20 else np.ones(len(frame), dtype=bool)
        fit_booster(xb[fallback_mask], frame.demand_kwh.to_numpy()[fallback_mask], 31).save_model(
            WEIGHTS_DIR / f"demand_fallback_{site}.txt"
        )
        fit_booster(xb[fallback_mask], frame.dwell_hours.to_numpy()[fallback_mask], 15).save_model(
            WEIGHTS_DIR / f"dwell_fallback_{site}.txt"
        )
        if valid.any():
            xw = np.column_stack([xb[valid], requested[valid]])
            fit_booster(xw, frame.demand_kwh.to_numpy()[valid], 31).save_model(
                WEIGHTS_DIR / f"demand_with_{site}.txt"
            )
            fit_booster(xw, frame.dwell_hours.to_numpy()[valid], 15).save_model(
                WEIGHTS_DIR / f"dwell_with_{site}.txt"
            )
        print(site, "rows", len(frame), "valid", int(valid.sum()), "saved")
    print("weights dir:", WEIGHTS_DIR)
    print("fit seconds:", round(time.time() - started, 2))


if __name__ == "__main__":
    main()
