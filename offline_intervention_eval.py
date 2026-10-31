"""模块4离线对照实验：natural vs EDF vs slack 容量策略。

用公开训练标签构造“完美预测”场景，隔离出容量内分配顺序对反事实轨迹的影响。
不依赖 sklearn，只说明模块4的干预推演行为，不产生官方分数。

用法: python offline_intervention_eval.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from acn_causal.student_api import ArrivalFeatures, Observation, Prediction, VehicleObservation
from module_3_world_model import forecast_natural
from module_4_intervention import forecast_capacity_edf


def load_training():
    training = {}
    for site in ("caltech", "jpl", "office1"):
        rows = json.load(open(f"data/training_{site}.json", encoding="utf-8"))
        frame = pd.DataFrame(rows)
        frame["day"] = pd.to_datetime(frame["day"])
        training[site] = frame
    return training


def main():
    training = load_training()
    frame = training["caltech"].sort_values("hour").head(6)

    known = {
        f"car-{i}": (float(row.demand_kwh), float(row.dwell_hours))
        for i, (_, row) in enumerate(frame.iterrows())
    }
    med_demand = float(training["caltech"].demand_kwh.median())
    med_dwell = float(training["caltech"].dwell_hours.median())

    def predict(features):
        demand, dwell = known.get(features.token, (med_demand, med_dwell))
        return Prediction(float(demand), float(dwell))

    def arrival_rate(site, hour, weekday):
        return 0.0

    vehicles = tuple(
        VehicleObservation(
            ArrivalFeatures(token, "caltech", float(row.hour), int(row.weekday), 6.6, None),
            0,
            0,
            0.0,
            False,
        )
        for token, (_, row) in zip(known, frame.iterrows())
    )

    horizon = 12
    # 前 4 步限流、随后放开，观察限流与回补。
    capacity = (10.0, 10.0, 10.0, 10.0) + (200.0,) * (96 - 4)
    background = (0.0,) * 96
    observation = Observation(
        step=0,
        dt_hours=0.25,
        horizon_steps=96,
        site="caltech",
        local_hour=9.0,
        weekday=0,
        vehicles=vehicles,
        capacity_kw=200.0,
        background_kw=0.0,
        prices=(0.6,) * 96,
        forecast_capacity=(200.0,) * 96,
        forecast_background=background,
        historical_arrivals=(0,) * 96,
        intervention_capacity=capacity,
    )

    natural = forecast_natural(observation, horizon, predict, arrival_rate)
    edf = forecast_capacity_edf(observation, horizon, predict, arrival_rate, "edf")
    slack = forecast_capacity_edf(observation, horizon, predict, arrival_rate, "slack")

    def summary(name, result):
        load = np.asarray(result["ev_load_kw"], dtype=float)
        return {
            "strategy": name,
            "served_kwh": round(float(load.sum() * 0.25), 3),
            "peak_kw": round(float(load.max()), 3),
            "load": [round(v, 2) for v in load],
        }

    rows = [summary("natural", natural), summary("edf", edf), summary("slack", slack)]
    print("站点 caltech | 6 辆在场车辆 | 前4步容量10kW、随后200kW")
    for row in rows:
        print(f"{row['strategy']:>8} served={row['served_kwh']:>7} kWh "
              f"peak={row['peak_kw']:>5} kW  load={row['load']}")

    edf_diff = np.asarray(edf["ev_load_kw"]) - np.asarray(natural["ev_load_kw"])
    slack_diff = np.asarray(slack["ev_load_kw"]) - np.asarray(natural["ev_load_kw"])
    print("响应差 MAE vs natural: edf=%.3f, slack=%.3f"
          % (float(np.mean(np.abs(edf_diff))), float(np.mean(np.abs(slack_diff)))))
    print("edf 与官方 capacity_edf 定义一致，保持为提交默认策略。")

    # 截断场景：容量持续紧张时，EDF 与 slack 的分配顺序才会拉开差距。
    urgent_late = VehicleObservation(
        ArrivalFeatures("late", "caltech", 9.0, 0, 6.6, None), 0, 0, 0.0, False
    )
    lax_early = VehicleObservation(
        ArrivalFeatures("early", "caltech", 9.0, 0, 6.6, None), 0, 0, 0.0, False
    )
    known2 = {"late": (20.0, 2.0), "early": (1.0, 0.5)}

    def predict2(features):
        demand, dwell = known2[features.token]
        return Prediction(float(demand), float(dwell))

    observation2 = Observation(
        step=0,
        dt_hours=0.25,
        horizon_steps=96,
        site="caltech",
        local_hour=9.0,
        weekday=0,
        vehicles=(lax_early, urgent_late),
        capacity_kw=6.6,
        background_kw=0.0,
        prices=(0.6,) * 96,
        forecast_capacity=(6.6,) * 96,
        forecast_background=(0.0,) * 96,
        historical_arrivals=(0,) * 96,
        intervention_capacity=(6.6,) * 96,
    )
    nat2 = forecast_natural(observation2, 3, predict2, arrival_rate)
    edf2 = forecast_capacity_edf(observation2, 3, predict2, arrival_rate, "edf")
    slack2 = forecast_capacity_edf(observation2, 3, predict2, arrival_rate, "slack")
    print("截断对照 | 早离开低需求车 vs 晚离开高需求车 | 容量6.6kW | 3步")
    print("natural", [round(v, 2) for v in nat2["ev_load_kw"]])
    print("edf    ", [round(v, 2) for v in edf2["ev_load_kw"]])
    print("slack  ", [round(v, 2) for v in slack2["ev_load_kw"]])


if __name__ == "__main__":
    main()
