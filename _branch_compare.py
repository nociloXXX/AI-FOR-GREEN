import json

import numpy as np
import pandas as pd


def load_training():
    training = {}
    for site in ("caltech", "jpl", "office1"):
        rows = json.load(open(f"data/training_{site}.json", encoding="utf-8"))
        frame = pd.DataFrame(rows)
        frame["day"] = pd.to_datetime(frame["day"])
        training[site] = frame
    return training


def split_time(training):
    train, held = {}, []
    for site, frame in training.items():
        days = sorted(frame.day.unique())
        cut = days[max(1, int(len(days) * 0.8)) - 1]
        train[site] = frame[frame.day <= cut].copy()
        held.extend((site, r) for r in frame[frame.day > cut].to_dict("records"))
    return train, held


def base(hour, weekday, pmax):
    hour = np.asarray(hour, dtype=float)
    weekday = np.asarray(weekday, dtype=float)
    pmax = np.asarray(pmax, dtype=float)
    return np.column_stack(
        [
            np.sin(hour * np.pi / 12.0),
            np.cos(hour * np.pi / 12.0),
            np.sin(weekday * 2.0 * np.pi / 7.0),
            np.cos(weekday * 2.0 * np.pi / 7.0),
            weekday,
            pmax,
        ]
    )


def fit(x, y, leaves):
    import lightgbm as lgb

    m = lgb.LGBMRegressor(
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
    m.fit(x, y)
    return m.booster_


def per_site_mae(models, held, mode):
    out = {}
    for site, r in held:
        req = r["last_requested_energy_kwh"]
        valid = req is not None and np.isfinite(float(req)) and float(req) > 0
        xb = base([r["hour"]], [r["weekday"]], [r["charger_limit_kw"]])
        if mode == "single":
            x = np.column_stack(
                [xb, [float(req) if valid else np.nan], [0.0 if valid else 1.0]]
            )
        else:
            x = np.column_stack([xb, [float(req)]]) if valid else xb
        dm, wm = models[site]
        pdm = float(dm.predict(x)[0])
        pdw = float(wm.predict(x)[0])
        out.setdefault(site, []).append(
            (abs(pdm - r["demand_kwh"]), abs(pdw - r["dwell_hours"]))
        )
    return {s: (np.mean([a for a, b in v]), np.mean([b for a, b in v])) for s, v in out.items()}


def single_branch(train):
    models = {}
    for site, frame in train.items():
        req = np.asarray(frame["last_requested_energy_kwh"], dtype=float)
        valid = np.isfinite(req) & (req > 0)
        missing = (~valid).astype(float)
        x = np.column_stack(
            [base(frame.hour, frame.weekday, frame.charger_limit_kw), req, missing]
        )
        models[site] = (fit(x, frame.demand_kwh, 31), fit(x, frame.dwell_hours, 15))
    return models


def dual_branch(train, fallback_all=True):
    models = {}
    for site, frame in train.items():
        req = np.asarray(frame["last_requested_energy_kwh"], dtype=float)
        valid = np.isfinite(req) & (req > 0)
        xb = base(frame.hour, frame.weekday, frame.charger_limit_kw)
        fb_mask = np.ones(len(frame), dtype=bool) if fallback_all else ~valid
        fb_d = fit(xb[fb_mask], frame.demand_kwh.to_numpy()[fb_mask], 31)
        fb_w = fit(xb[fb_mask], frame.dwell_hours.to_numpy()[fb_mask], 15)
        if valid.any():
            xw = np.column_stack([xb[valid], req[valid]])
            with_d = fit(xw, frame.demand_kwh.to_numpy()[valid], 31)
            with_w = fit(xw, frame.dwell_hours.to_numpy()[valid], 15)
        else:
            with_d, with_w = fb_d, fb_w
        models[site] = (_Branch(fb_d, with_d), _Branch(fb_w, with_w))
    return models


class _Branch:
    def __init__(self, fallback, with_req):
        self.fallback = fallback
        self.with_req = with_req

    def predict(self, x):
        # x 已按有效/缺失选好列数：7 列走 with_req，6 列走 fallback。
        if x.shape[1] == 7:
            return self.with_req.predict(x)
        return self.fallback.predict(x)


def main():
    training = load_training()
    train, held = split_time(training)
    single = single_branch(train)
    dual_all = dual_branch(train, fallback_all=True)
    dual_missing = dual_branch(train, fallback_all=False)
    print("single-branch per-site:", per_site_mae(single, held, "single"))
    print("dual fallback=all rows:", per_site_mae(dual_all, held, "dual"))
    print("dual fallback=missing only:", per_site_mae(dual_missing, held, "dual"))


if __name__ == "__main__":
    main()
