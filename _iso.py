import json

import numpy as np
import pandas as pd

from acn_causal.student_api import ArrivalFeatures, Prediction
from completion_proxy import (
    _apply_condition, _background_curve, _build_day_vehicles,
    _peak_concurrent_kw, _simulate, load_training,
)


def base(hour, weekday, pmax):
    hour = np.asarray(hour, dtype=float) % 24.0
    weekday = np.asarray(weekday, dtype=float)
    pmax = np.asarray(pmax, dtype=float)
    return np.column_stack([
        np.sin(hour * np.pi / 12.0), np.cos(hour * np.pi / 12.0),
        np.sin(hour * np.pi / 6.0), np.cos(hour * np.pi / 6.0),
        weekday, (weekday >= 5.0).astype(float), pmax,
    ])


def sw(days, hl):
    days = pd.to_datetime(days)
    age = (days.max() - days).dt.total_seconds() / 86400.0
    return np.exp(-np.log(2.0) * age / hl)


def fit_q(x, y, q=0.5, w=None, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=200,
                                      max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)
    m.fit(x, y, sample_weight=w)
    return m


def run(vehicles, peak, predict_fn):
    bg = _background_curve(peak)
    return _simulate(vehicles, "office1", peak, bg, predict_fn, True)[0]


def main():
    training = load_training()
    office = pd.DataFrame(training["office1"])
    office["day"] = pd.to_datetime(office["day"])
    rows = office[office["day"] == "2019-10-18"].to_dict("records")
    dv = _build_day_vehicles(rows)
    vehicles = _apply_condition(dv, "arrival_burst")
    peak = _peak_concurrent_kw(dv)

    X = base(office.hour, office.weekday, office.charger_limit_kw)
    yd = office.demand_kwh.to_numpy()
    yw = office.dwell_hours.to_numpy()

    d1 = fit_q(X, yd, 0.5)
    w1 = fit_q(X, yw, 0.5)
    def p1(f):
        xb = base([f.hour], [f.weekday], [f.charger_limit_kw])
        return Prediction(float(d1.predict(xb)[0]), float(w1.predict(xb)[0]))
    print("base_single:", round(run(vehicles, peak, p1), 4))

    ws = sw(office.day, 180)
    d2 = fit_q(X, yd, 0.5, ws)
    w2 = fit_q(X, yw, 0.5, ws)
    def p2(f):
        xb = base([f.hour], [f.weekday], [f.charger_limit_kw])
        return Prediction(float(d2.predict(xb)[0]), float(w2.predict(xb)[0]))
    print("sample_weight180:", round(run(vehicles, peak, p2), 4))

    d3 = [fit_q(X, yd, 0.5, seed=s) for s in (20261008, 20261009)]
    w3 = [fit_q(X, yw, 0.5, seed=s) for s in (20261008, 20261009)]
    def p3(f):
        xb = base([f.hour], [f.weekday], [f.charger_limit_kw])
        d = float(np.mean([m.predict(xb)[0] for m in d3]))
        w = float(np.mean([m.predict(xb)[0] for m in w3]))
        return Prediction(d, w)
    print("double_seed:", round(run(vehicles, peak, p3), 4))

    def p4(f):
        p = p1(f)
        return Prediction(min(p.demand_kwh, f.charger_limit_kw * p.dwell_hours), p.dwell_hours)
    print("demand_cap:", round(run(vehicles, peak, p4), 4))


if __name__ == "__main__":
    main()
