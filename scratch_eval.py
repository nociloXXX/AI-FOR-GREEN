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


def split_time(training, ratio=0.8):
    train, held = {}, []
    for site, frame in training.items():
        days = sorted(frame.day.unique())
        cut = days[max(1, int(len(days) * ratio)) - 1]
        train[site] = frame[frame.day <= cut].copy()
        held.extend((site, r) for r in frame[frame.day > cut].to_dict("records"))
    return train, held


def base_features(hour, weekday, pmax):
    hour = np.asarray(hour, dtype=float)
    weekday = np.asarray(weekday, dtype=float)
    pmax = np.asarray(pmax, dtype=float)
    return np.column_stack(
        [np.sin(hour * np.pi / 12.0), np.cos(hour * np.pi / 12.0), weekday, pmax]
    )


def rich_features(hour, weekday, pmax, requested):
    hour = np.asarray(hour, dtype=float)
    weekday = np.asarray(weekday, dtype=float)
    pmax = np.asarray(pmax, dtype=float)
    requested = np.asarray(requested, dtype=float)
    missing = np.isnan(requested).astype(float)
    req = np.nan_to_num(requested, nan=0.0)
    return np.column_stack(
        [
            np.sin(hour * np.pi / 12.0),
            np.cos(hour * np.pi / 12.0),
            np.sin(weekday * 2 * np.pi / 7),
            np.cos(weekday * 2 * np.pi / 7),
            weekday,
            pmax,
            req,
            missing,
            req * missing,
        ]
    )


def eval_mae(preds, held):
    d_err, w_err = [], []
    for (site, r), pdm, pdw in zip(held, preds[:, 0], preds[:, 1]):
        d_err.append(abs(pdm - r["demand_kwh"]))
        w_err.append(abs(pdw - r["dwell_hours"]))
    return float(np.mean(d_err)), float(np.mean(w_err))


def run_baseline(train, held):
    from sklearn.ensemble import ExtraTreesRegressor

    models = {}
    preds = []
    for site, frame in train.items():
        x = base_features(frame.hour, frame.weekday, frame.charger_limit_kw)
        y = np.column_stack([frame.demand_kwh, frame.dwell_hours])
        m = ExtraTreesRegressor(
            n_estimators=96, min_samples_leaf=15, max_depth=12, n_jobs=1, random_state=20261008
        )
        m.fit(x, y)
        models[site] = m
    for site, r in held:
        x = base_features([r["hour"]], [r["weekday"]], [r["charger_limit_kw"]])
        y = models[site].predict(x)[0]
        preds.append(y)
    return eval_mae(np.asarray(preds), held)


def run_rich(train, held, model_name="hist", separate=True):
    preds = []
    if model_name == "hist":
        from sklearn.ensemble import HistGradientBoostingRegressor

        def make():
            return HistGradientBoostingRegressor(
                max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=7,
            )
    elif model_name == "gbr":
        from sklearn.ensemble import GradientBoostingRegressor

        def make():
            return GradientBoostingRegressor(
                n_estimators=250, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=7,
            )
    elif model_name == "lgb":
        import lightgbm as lgb

        def make():
            return lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.05, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=7, verbose=-1, n_jobs=1,
            )
    elif model_name == "xgb":
        import xgboost as xgb

        def make():
            return xgb.XGBRegressor(
                n_estimators=400, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=7, verbosity=0, n_jobs=1,
            )

    demand_models, dwell_models = {}, {}
    for site, frame in train.items():
        x = rich_features(
            frame.hour, frame.weekday, frame.charger_limit_kw,
            frame.last_requested_energy_kwh,
        )
        dm = make()
        dm.fit(x, frame.demand_kwh)
        wm = make()
        wm.fit(x, frame.dwell_hours)
        demand_models[site] = dm
        dwell_models[site] = wm
    for site, r in held:
        x = rich_features(
            [r["hour"]], [r["weekday"]], [r["charger_limit_kw"]],
            [r["last_requested_energy_kwh"]],
        )
        preds.append([demand_models[site].predict(x)[0], dwell_models[site].predict(x)[0]])
    return eval_mae(np.asarray(preds), held)


def run_median(train, held):
    med = {s: (f.demand_kwh.median(), f.dwell_hours.median()) for s, f in train.items()}
    preds = [med[s] for s, _ in held]
    return eval_mae(np.asarray(preds), held)


def run_two_stage(train, held, model_name="xgb"):
    import xgboost as xgb

    def make():
        return xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=7, verbosity=0, n_jobs=1,
        )

    demand_models, dwell_models = {}, {}
    for site, frame in train.items():
        x = rich_features(
            frame.hour, frame.weekday, frame.charger_limit_kw,
            frame.last_requested_energy_kwh,
        )
        dm = make()
        dm.fit(x, frame.demand_kwh)
        demand_models[site] = dm
        x2 = np.column_stack([x, dm.predict(x)])
        wm = make()
        wm.fit(x2, frame.dwell_hours)
        dwell_models[site] = wm
    preds = []
    for site, r in held:
        x = rich_features(
            [r["hour"]], [r["weekday"]], [r["charger_limit_kw"]],
            [r["last_requested_energy_kwh"]],
        )
        pdm = demand_models[site].predict(x)[0]
        pdw = dwell_models[site].predict(np.column_stack([x, [pdm]]))[0]
        preds.append([pdm, pdw])
    return eval_mae(np.asarray(preds), held)


def run_lgb_objective(train, held, obj="regression_l1"):
    import lightgbm as lgb

    demand_models, dwell_models = {}, {}
    for site, frame in train.items():
        x = rich_features(
            frame.hour, frame.weekday, frame.charger_limit_kw,
            frame.last_requested_energy_kwh,
        )
        dm = lgb.LGBMRegressor(
            objective=obj, n_estimators=500, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=7, verbose=-1, n_jobs=1,
        )
        dm.fit(x, frame.demand_kwh)
        wm = lgb.LGBMRegressor(
            objective=obj, n_estimators=500, learning_rate=0.05, num_leaves=15,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=7, verbose=-1, n_jobs=1,
        )
        wm.fit(x, frame.dwell_hours)
        demand_models[site] = dm
        dwell_models[site] = wm
    preds = []
    for site, r in held:
        x = rich_features(
            [r["hour"]], [r["weekday"]], [r["charger_limit_kw"]],
            [r["last_requested_energy_kwh"]],
        )
        preds.append([demand_models[site].predict(x)[0], dwell_models[site].predict(x)[0]])
    return eval_mae(np.asarray(preds), held)


def run_lgb_tuned(train, held):
    import lightgbm as lgb

    demand_models, dwell_models = {}, {}
    for site, frame in train.items():
        x = rich_features(
            frame.hour, frame.weekday, frame.charger_limit_kw,
            frame.last_requested_energy_kwh,
        )
        dm = lgb.LGBMRegressor(
            objective="regression_l1", n_estimators=800, learning_rate=0.04,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=7,
            verbose=-1, n_jobs=1,
        )
        dm.fit(x, frame.demand_kwh)
        wm = lgb.LGBMRegressor(
            objective="regression_l1", n_estimators=600, learning_rate=0.04,
            num_leaves=15, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=7,
            verbose=-1, n_jobs=1,
        )
        wm.fit(x, frame.dwell_hours)
        demand_models[site] = dm
        dwell_models[site] = wm
    preds = []
    for site, r in held:
        x = rich_features(
            [r["hour"]], [r["weekday"]], [r["charger_limit_kw"]],
            [r["last_requested_energy_kwh"]],
        )
        preds.append([demand_models[site].predict(x)[0], dwell_models[site].predict(x)[0]])
    return eval_mae(np.asarray(preds), held)


def run_lgb_early(train, held):
    import lightgbm as lgb

    demand_models, dwell_models = {}, {}
    site_errors = {}
    for site, frame in train.items():
        x = rich_features(
            frame.hour, frame.weekday, frame.charger_limit_kw,
            frame.last_requested_energy_kwh,
        )
        days = sorted(frame.day.unique())
        cut = days[max(1, int(len(days) * 0.9)) - 1]
        fit_mask = (frame.day <= cut).to_numpy()
        val_mask = ~fit_mask

        def fit_one(y_col, leaves, n_est):
            m = lgb.LGBMRegressor(
                objective="regression_l1", n_estimators=n_est, learning_rate=0.04,
                num_leaves=leaves, min_child_samples=20, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=7,
                verbose=-1, n_jobs=1,
            )
            m.fit(
                x[fit_mask], frame[y_col].to_numpy()[fit_mask],
                eval_set=[(x[val_mask], frame[y_col].to_numpy()[val_mask])],
                eval_metric="mae",
                callbacks=[lgb.early_stopping(40, verbose=False)],
            )
            return m

        dm = fit_one("demand_kwh", 31, 1500)
        wm = fit_one("dwell_hours", 15, 1200)
        demand_models[site] = dm
        dwell_models[site] = wm

        site_held = [(s, r) for s, r in held if s == site]
        sp = []
        for s, r in site_held:
            xx = rich_features(
                [r["hour"]], [r["weekday"]], [r["charger_limit_kw"]],
                [r["last_requested_energy_kwh"]],
            )
            sp.append([dm.predict(xx)[0], wm.predict(xx)[0]])
        site_errors[site] = eval_mae(np.asarray(sp), site_held)
    preds = []
    for site, r in held:
        xx = rich_features(
            [r["hour"]], [r["weekday"]], [r["charger_limit_kw"]],
            [r["last_requested_energy_kwh"]],
        )
        preds.append([demand_models[site].predict(xx)[0], dwell_models[site].predict(xx)[0]])
    return eval_mae(np.asarray(preds), held), site_errors


def eda(training):
    for site, frame in training.items():
        print(site, len(frame), "days", frame.day.nunique())
        print("  charger_limit unique:", sorted(frame.charger_limit_kw.unique()))
        print(
            "  last_requested coverage:",
            frame.last_requested_energy_kwh.notna().mean().round(3),
        )
        print("  demand q:", frame.demand_kwh.quantile([0.25, 0.5, 0.75, 0.95]).round(2).tolist())
        print("  dwell  q:", frame.dwell_hours.quantile([0.25, 0.5, 0.75, 0.95]).round(2).tolist())
        if frame.last_requested_energy_kwh.notna().any():
            corr = frame[["demand_kwh", "dwell_hours", "last_requested_energy_kwh"]].corr()
            print("  corr:\n", corr.round(3))


def main():
    training = load_training()
    eda(training)
    train, held = split_time(training)
    print("train rows", sum(len(f) for f in train.values()), "held", len(held))
    print("site-median MAE:", run_median(train, held))
    print("baseline ExtraTrees MAE:", run_baseline(train, held))
    for name in ("hist", "gbr", "lgb", "xgb"):
        print(name, "rich MAE:", run_rich(train, held, name))
    print("xgb two-stage MAE:", run_two_stage(train, held))
    print("lgb L1 MAE:", run_lgb_objective(train, held, "regression_l1"))
    print("lgb L2 MAE:", run_lgb_objective(train, held, "regression_l2"))
    print("lgb tuned L1 MAE:", run_lgb_tuned(train, held))
    total, per_site = run_lgb_early(train, held)
    print("lgb early L1 MAE:", total)
    for site, err in per_site.items():
        print("  ", site, "held", sum(1 for s, _ in held if s == site), "MAE", err)


if __name__ == "__main__":
    main()
