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


def fit(x, y, leaves=31, w=None):
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
    m.fit(x, y, sample_weight=w)
    return m.booster_


def demand_mae_per_site(predict_fn, held):
    errs = {}
    for site, r in held:
        pred = predict_fn(site, r)
        errs.setdefault(site, []).append(abs(pred - r["demand_kwh"]))
    return {s: float(np.mean(v)) for s, v in errs.items()}


def approach_office_only(train, held):
    models = {}
    for site, frame in train.items():
        x = base(frame.hour, frame.weekday, frame.charger_limit_kw)
        models[site] = fit(x, frame.demand_kwh)

    def pred(site, r):
        x = base([r["hour"]], [r["weekday"]], [r["charger_limit_kw"]])
        return float(models[site].predict(x)[0])

    return demand_mae_per_site(pred, held)


def approach_joint(train, held, weight_office=1.0, normalize=False):
    site_idx = {"caltech": 0, "jpl": 1, "office1": 2}
    medians = {s: float(f.demand_kwh.median()) for s, f in train.items()}
    xs, ys, ws = [], [], []
    for site, frame in train.items():
        xb = base(frame.hour, frame.weekday, frame.charger_limit_kw)
        onehot = np.zeros((len(frame), 3))
        onehot[:, site_idx[site]] = 1.0
        x = np.column_stack([xb, onehot])
        y = (
            frame.demand_kwh.to_numpy() / medians[site]
            if normalize
            else frame.demand_kwh.to_numpy()
        )
        w = np.full(len(frame), weight_office if site == "office1" else 1.0)
        xs.append(x)
        ys.append(y)
        ws.append(w)
    model = fit(np.vstack(xs), np.concatenate(ys), w=np.concatenate(ws))

    def pred(site, r):
        xb = base([r["hour"]], [r["weekday"]], [r["charger_limit_kw"]])
        onehot = np.zeros((1, 3))
        onehot[0, site_idx[site]] = 1.0
        x = np.column_stack([xb, onehot])
        raw = float(model.predict(x)[0])
        return raw * medians[site] if normalize else raw

    return demand_mae_per_site(pred, held)


def approach_weekday_hour_median(train, held):
    tables = {}
    for site, frame in train.items():
        g = frame.copy()
        g["bin"] = np.clip(np.asarray(g.hour, dtype=float).astype(int) % 24, 0, 23)
        tables[site] = g.groupby(["weekday", "bin"]).demand_kwh.median()
        tables[site].default = g.demand_kwh.median()

    def pred(site, r):
        key = (int(r["weekday"]), int(float(r["hour"])) % 24)
        t = tables[site]
        return float(t.get(key, t.default))

    return demand_mae_per_site(pred, held)


def main():
    training = load_training()
    train, held = split_time(training)
    print("office-only:", approach_office_only(train, held))
    print("joint:", approach_joint(train, held))
    print("joint office x3:", approach_joint(train, held, weight_office=3.0))
    print("joint normalized:", approach_joint(train, held, normalize=True))
    print("joint normalized office x3:", approach_joint(train, held, weight_office=3.0, normalize=True))
    print("weekday-hour median:", approach_weekday_hour_median(train, held))


if __name__ == "__main__":
    main()
