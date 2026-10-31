import json
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from acn_causal.student_api import ArrivalFeatures, Observation, Prediction, VehicleObservation
from completion_proxy import (
    DT, STEPS, _apply_condition, _background_curve, _build_day_vehicles,
    _peak_concurrent_kw, TOU_PRICES, load_training,
)
from module_2_prediction import _features_base
from module_5_decision import act_edf


def fit_quantile(x, y, q):
    m = HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_iter=200, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=0,
    )
    m.fit(x, y)
    return m


def simulate(vehicles, site, ev_capacity, background, predict_fn):
    by_arrival = defaultdict(list)
    for v in vehicles:
        by_arrival[v["arrive"]].append(v)
    delivered = {v["token"]: 0.0 for v in vehicles}
    active = []
    completed = set()
    fc = tuple(ev_capacity + b for b in background)
    for t in range(STEPS):
        for v in by_arrival.get(t, []):
            active.append(v)
        obs = []
        for v in active:
            if not (v["arrive"] <= t < v["depart"]):
                continue
            obs.append(VehicleObservation(
                ArrivalFeatures(v["token"], site, v["hour"], v["weekday"], v["limit"],
                                None if not np.isfinite(v["requested"]) else v["requested"]),
                v["arrive"], t - v["arrive"], delivered[v["token"]],
                v["target"] - delivered[v["token"]] <= 1e-9))
        observation = Observation(
            step=t, dt_hours=DT, horizon_steps=STEPS, site=site,
            local_hour=(t * DT) % 24.0, weekday=0, vehicles=tuple(obs),
            capacity_kw=ev_capacity + background[t], background_kw=background[t],
            prices=TOU_PRICES, forecast_capacity=fc, forecast_background=background,
            historical_arrivals=(0,) * STEPS, intervention_capacity=())
        power = act_edf(observation, predict_fn, price_deferral=True)
        for v in active:
            if not (v["arrive"] <= t < v["depart"]):
                continue
            need = max(v["target"] - delivered[v["token"]], 0.0)
            delivered[v["token"]] += min(float(power.get(v["token"], 0.0)) * DT, need)
            if v["depart"] == t + 1:
                rem = max(v["target"] - delivered[v["token"]], 0.0)
                if rem <= 0.03 * v["target"] + 1e-9:
                    completed.add(v["token"])
    return len(completed), len(vehicles)


def main():
    training = load_training()
    office = pd.DataFrame(training["office1"])
    office["day"] = pd.to_datetime(office["day"])
    x = _features_base(office.hour, office.weekday, office.charger_limit_kw)
    y_d = office.demand_kwh.to_numpy()
    y_w = office.dwell_hours.to_numpy()

    demand_models = {q: fit_quantile(x, y_d, q) for q in (0.5, 0.75, 0.9)}
    dwell_models = {q: fit_quantile(x, y_w, q) for q in (0.25, 0.5, 0.75)}

    by_day = defaultdict(list)
    for row in training["office1"]:
        by_day[row["day"]].append(row)
    chosen = sorted(by_day.values(), key=len, reverse=True)[:4]

    for dq in (0.5, 0.75, 0.9):
        for wq in (0.25, 0.5, 0.75):
            rates = []
            for day_rows in chosen:
                day_vehicles = _build_day_vehicles(day_rows)
                peak = _peak_concurrent_kw(day_vehicles)
                if peak <= 1e-6:
                    continue
                bg = _background_curve(peak)
                for cond in ("normal", "capacity_limited", "arrival_burst", "demand_shift"):
                    vehicles = _apply_condition(day_vehicles, cond)
                    cap = peak
                    def predict_fn(f, dq=dq, wq=wq):
                        xb = _features_base([f.hour], [f.weekday], [f.charger_limit_kw])
                        return Prediction(
                            float(demand_models[dq].predict(xb)[0]),
                            float(dwell_models[wq].predict(xb)[0]))
                    done, total = simulate(vehicles, "office1", cap, bg, predict_fn)
                    rates.append((cond, day_rows[0]["day"], done / total))
            min_rate = min(r for _, _, r in rates)
            fail = [r for r in rates if r[2] < 0.85]
            print(f"demand q{dq} dwell q{wq}: min_completion={min_rate:.4f} fails={len(fail)}")
            for r in fail:
                print("   ", r)

    # 现有 LightGBM L1 office 单站模型组合验证。
    from module_2_prediction import _fit_booster
    lgb_d = _fit_booster(x, y_d, 31)
    lgb_w = _fit_booster(x, y_w, 15)
    rates = []
    for day_rows in chosen:
        day_vehicles = _build_day_vehicles(day_rows)
        peak = _peak_concurrent_kw(day_vehicles)
        bg = _background_curve(peak)
        for cond in ("normal", "capacity_limited", "arrival_burst", "demand_shift"):
            vehicles = _apply_condition(day_vehicles, cond)
            def predict_fn(f):
                xb = _features_base([f.hour], [f.weekday], [f.charger_limit_kw])
                return Prediction(float(lgb_d.predict(xb)[0]), float(lgb_w.predict(xb)[0]))
            done, total = simulate(vehicles, "office1", peak, bg, predict_fn)
            rates.append((cond, day_rows[0]["day"], done / total))
    print("LightGBM L1 office-only: min", min(r for _, _, r in rates),
          "fails", [r for r in rates if r[2] < 0.85])


if __name__ == "__main__":
    main()
