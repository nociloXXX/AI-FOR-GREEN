"""本地完成率代理：用公开训练标签按 act 跑容量分配，数 3% 容差完成率。

这不是官方评测器，只用于验证决策策略（module_5_decision.act_edf）在容量
约束下能否把每辆车送到自身目标电量的 97% 以上。predict 使用“完美预测”
（直接取训练标签的需求/停留），以隔离决策模块，不把预测误差混进来。
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from acn_causal.student_api import (
    ArrivalFeatures,
    Observation,
    Prediction,
    VehicleObservation,
)
from module_5_decision import act_edf


def load_training():
    training = {}
    for site in ("caltech", "jpl", "office1"):
        rows = json.loads(
            Path(f"data/training_{site}.json").read_text(encoding="utf-8")
        )
        training[site] = rows
    return training


def busiest_day(rows):
    counts = defaultdict(int)
    for row in rows:
        counts[row["day"]] += 1
    day = max(counts, key=counts.get)
    return [row for row in rows if row["day"] == day]


def build_episode(rows, dt=0.25, steps=96):
    episode = []
    for idx, row in enumerate(rows):
        arrive = max(0, min(int(float(row["hour"]) * 4), steps - 1))
        dwell_steps = max(1, int(math.ceil(float(row["dwell_hours"]) / dt)))
        episode.append(
            {
                "token": f"v{idx}",
                "site": None,  # 填充时由外部覆盖
                "arrive": arrive,
                "depart": min(arrive + dwell_steps, steps),
                "target": max(float(row["demand_kwh"]), 0.0),
                "limit": max(float(row["charger_limit_kw"]), 0.0),
                "hour": float(row["hour"]),
                "weekday": int(row["weekday"]),
            }
        )
    return episode


def peak_concurrent_kw(episode, steps=96):
    demand = [0.0] * steps
    for veh in episode:
        for t in range(veh["arrive"], veh["depart"]):
            demand[t] += veh["limit"]
    return max(demand) if demand else 0.0


def simulate(episode, site, ev_capacity_kw, dt=0.25, steps=96):
    lookup = {v["token"]: v for v in episode}
    by_arrival = defaultdict(list)
    for veh in episode:
        by_arrival[veh["arrive"]].append(veh)

    def predict_fn(features):
        veh = lookup[features.token]
        return Prediction(veh["target"], (veh["depart"] - veh["arrive"]) * dt)

    active = []
    completed = 0
    total = len(episode)
    prices = tuple(
        [0.6] * 24 + [0.9] * 8 + [1.2] * 16 + [0.8] * 8 + [1.1] * 16 + [1.5] * 16 + [0.7] * 8
    )
    prices = prices[:steps]
    if len(prices) < steps:
        prices = prices + prices[-1:] * (steps - len(prices))

    for t in range(steps):
        for veh in by_arrival.get(t, []):
            active.append(
                {
                    **veh,
                    "delivered": 0.0,
                    "complete": veh["target"] <= 1e-9,
                    "elapsed": 0,
                }
            )

        vehicles = []
        for a in active:
            if a["arrive"] <= t < a["depart"]:
                features = ArrivalFeatures(
                    a["token"], site, a["hour"], a["weekday"], a["limit"], None
                )
                vehicles.append(
                    VehicleObservation(
                        features,
                        a["arrive"],
                        t - a["arrive"],
                        a["delivered"],
                        a["complete"],
                    )
                )

        observation = Observation(
            step=t,
            dt_hours=dt,
            horizon_steps=steps,
            site=site,
            local_hour=(t * dt) % 24.0,
            weekday=0,
            vehicles=tuple(vehicles),
            capacity_kw=ev_capacity_kw,
            background_kw=0.0,
            prices=prices,
            forecast_capacity=(ev_capacity_kw,) * steps,
            forecast_background=(0.0,) * steps,
            historical_arrivals=(0,) * steps,
            intervention_capacity=(),
        )
        power = act_edf(observation, predict_fn)

        for a in active:
            if not (a["arrive"] <= t < a["depart"]):
                continue
            granted = power.get(a["token"], 0.0)
            need = max(a["target"] - a["delivered"], 0.0)
            a["delivered"] += min(granted * dt, need)
            a["elapsed"] += 1
            if a["target"] - a["delivered"] <= 1e-9:
                a["complete"] = True

        remaining = []
        for a in active:
            if a["depart"] == t + 1:
                # 离开时判定：剩余未交付电量 <= 自身目标电量的 3%。
                if a["target"] - a["delivered"] <= 0.03 * a["target"] + 1e-9:
                    completed += 1
            else:
                remaining.append(a)
        active = remaining

    # 场景结束时仍连接且目标为正的车辆，也按同口径判定一次。
    for a in active:
        if a["target"] - a["delivered"] <= 0.03 * a["target"] + 1e-9:
            completed += 1
    return completed / total if total else 0.0, completed, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=["caltech", "jpl", "office1"], default=None)
    parser.add_argument("--capacity", type=float, default=None)
    args = parser.parse_args()

    training = load_training()
    sites = [args.site] if args.site else ["caltech", "jpl", "office1"]
    thresholds = {"caltech": 0.85, "jpl": 0.65, "office1": 0.85}

    for site in sites:
        rows = busiest_day(training[site])
        episode = build_episode(rows)
        for veh in episode:
            veh["site"] = site
        peak = peak_concurrent_kw(episode)

        if args.capacity is not None:
            capacities = [args.capacity]
        else:
            capacities = [round(peak * f, 1) for f in (0.35, 0.5, 0.65, 0.8, 1.0, 1.25)]

        print(f"\n[{site}] 车辆={len(episode)} 峰值并充={peak:.1f} kW")
        for cap in capacities:
            rate, done, total = simulate(episode, site, cap)
            ok = "OK" if rate + 1e-9 >= thresholds[site] else "FAIL"
            print(
                f"  容量={cap:7.1f} kW  完成率={rate:.3f} ({done}/{total})  "
                f"门槛{thresholds[site]:.2f} -> {ok}"
            )


if __name__ == "__main__":
    main()
