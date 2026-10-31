"""离线训练 module_2_prediction 的 LightGBM 权重。

每个站点训练单目标 demand/dwell 模型：
- fallback：只含 hour/weekday/charger_limit，在申报需求缺失样本上训练；
- with：含 last_requested_energy_kwh，只在正有限申报值样本上训练。

另训练全站 pooled 模型（站点 one-hot），并用 office1 的时间序 holdout 选出
``pred = alpha*site + (1-alpha)*pooled`` 的最优 alpha，写成 office1_alpha.json。
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
SITE_INDEX = {site: i for i, site in enumerate(SITES)}
ALPHA_GRID = np.arange(0.0, 1.01, 0.01)
GAMMA_GRID = np.arange(0.80, 1.21, 0.01)


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


def fit_booster(x, y, num_leaves, sample_weight=None):
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
    model.fit(x, y, sample_weight=sample_weight)
    return model.booster_


def load_training():
    training = {}
    for site in SITES:
        rows = json.load(open(ROOT / "data" / f"training_{site}.json", encoding="utf-8"))
        frame = pd.DataFrame(rows)
        frame["day"] = pd.to_datetime(frame["day"])
        training[site] = frame
    return training


def office_split(frame, ratio=0.85):
    days = sorted(frame.day.unique())
    cut = days[max(1, int(len(days) * ratio)) - 1]
    return frame[frame.day <= cut].copy(), frame[frame.day > cut].copy()


def select_alpha(office, training, target):
    office_fit, office_val = office_split(office)
    site_model = fit_booster(
        features_base(office_fit.hour, office_fit.weekday, office_fit.charger_limit_kw),
        office_fit[target],
        31 if target == "demand_kwh" else 15,
    )

    xs, ys = [], []
    for site in SITES:
        frame = training[site] if site != "office1" else office_fit
        xs.append(pooled_features(frame.hour, frame.weekday, frame.charger_limit_kw, site))
        ys.append(frame[target].to_numpy())
    pooled = fit_booster(
        np.vstack(xs), np.concatenate(ys), 31 if target == "demand_kwh" else 15
    )

    xb_val = features_base(office_val.hour, office_val.weekday, office_val.charger_limit_kw)
    xp_val = pooled_features(office_val.hour, office_val.weekday, office_val.charger_limit_kw, "office1")
    site_pred = site_model.predict(xb_val)
    pooled_pred = pooled.predict(xp_val)
    truth = office_val[target].to_numpy()

    blend = np.column_stack([site_pred, pooled_pred])
    best = None
    for alpha in ALPHA_GRID:
        mixed = alpha * site_pred + (1.0 - alpha) * pooled_pred
        for gamma in GAMMA_GRID:
            mae = float(np.abs(truth - gamma * mixed).mean())
            if best is None or mae < best[0]:
                best = (mae, float(alpha), float(gamma))
    mae_site = float(np.abs(truth - site_pred).mean())
    mae_pooled = float(np.abs(truth - pooled_pred).mean())
    return best[1], best[2], mae_site, mae_pooled


def main():
    training = load_training()
    WEIGHTS_DIR.mkdir(exist_ok=True)
    started = time.time()

    for site in SITES:
        frame = training[site]
        requested = (
            np.asarray(frame["last_requested_energy_kwh"], dtype=float)
            if "last_requested_energy_kwh" in frame.columns
            else np.full(len(frame), np.nan)
        )
        valid = np.isfinite(requested) & (requested > 0.0)
        xb = features_base(frame.hour, frame.weekday, frame.charger_limit_kw)

        if valid.any() and (~valid).sum() >= 20:
            fallback_mask = ~valid
        else:
            fallback_mask = np.ones(len(frame), dtype=bool)
        fit_booster(
            xb[fallback_mask], frame.demand_kwh.to_numpy()[fallback_mask], 31
        ).save_model(WEIGHTS_DIR / f"demand_fallback_{site}.txt")
        fit_booster(
            xb[fallback_mask], frame.dwell_hours.to_numpy()[fallback_mask], 15
        ).save_model(WEIGHTS_DIR / f"dwell_fallback_{site}.txt")

        if valid.any():
            xw = np.column_stack([xb[valid], requested[valid]])
            fit_booster(
                xw, frame.demand_kwh.to_numpy()[valid], 31
            ).save_model(WEIGHTS_DIR / f"demand_with_{site}.txt")
            fit_booster(
                xw, frame.dwell_hours.to_numpy()[valid], 15
            ).save_model(WEIGHTS_DIR / f"dwell_with_{site}.txt")

    xs, ys_d, ys_w = [], [], []
    for site in SITES:
        frame = training[site]
        xs.append(pooled_features(frame.hour, frame.weekday, frame.charger_limit_kw, site))
        ys_d.append(frame.demand_kwh.to_numpy())
        ys_w.append(frame.dwell_hours.to_numpy())
    fit_booster(np.vstack(xs), np.concatenate(ys_d), 31).save_model(
        WEIGHTS_DIR / "demand_pooled.txt"
    )
    fit_booster(np.vstack(xs), np.concatenate(ys_w), 15).save_model(
        WEIGHTS_DIR / "dwell_pooled.txt"
    )

    alpha_demand, gamma_demand, site_mae_d, pooled_mae_d = select_alpha(
        training["office1"], training, "demand_kwh"
    )
    alpha_dwell, gamma_dwell, site_mae_w, pooled_mae_w = select_alpha(
        training["office1"], training, "dwell_hours"
    )
    with open(WEIGHTS_DIR / "office1_alpha.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "demand": {"alpha": alpha_demand, "gamma": gamma_demand},
                "dwell": {"alpha": alpha_dwell, "gamma": gamma_dwell},
            },
            fh,
        )

    print("office1 demand: site-only MAE", round(site_mae_d, 4),
          "| pooled MAE", round(pooled_mae_d, 4),
          "| alpha", alpha_demand, "| gamma", gamma_demand)
    print("office1 dwell: site-only MAE", round(site_mae_w, 4),
          "| pooled MAE", round(pooled_mae_w, 4),
          "| alpha", alpha_dwell, "| gamma", gamma_dwell)
    print("weights dir:", WEIGHTS_DIR)
    print("fit seconds:", round(time.time() - started, 2))


if __name__ == "__main__":
    main()
