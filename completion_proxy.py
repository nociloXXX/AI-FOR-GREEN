"""本地完成率代理验收闸门（P0-A）。

把公开训练标签重放成逐日 96 步场景，覆盖四种条件：
  normal            原车池，正常容量
  capacity_limited  容量降到峰值并充功率的 0.4 倍
  arrival_burst     车池复制一份（到达高峰）
  demand_shift      每车目标电量 x1.35（需求偏移）

每站取会话数最多的若干天，逐日（逐场景）计算完成率：
  完成 = 剩余未交付电量 <= 0.03 * 该车自身目标电量
必须每一天都满足该站门槛，任一场景 FAIL 即整体 FAIL。

同时做 price_deferral=True/False 的 A/B：启用价格延后不得降低完成率。
支持完美预测（隔离决策）与真实 predict（接 submission.build_model）两种口径。

背景负荷为本地合成的站级日曲线，非官方曲线，仅用于验证 act 在
“容量扣除背景后才是可用功率”下的行为。
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from acn_causal.student_api import ArrivalFeatures, Observation, Prediction, VehicleObservation
from module_5_decision import act_edf


SITES = ("caltech", "jpl", "office1")
THRESHOLDS = {"caltech": 0.85, "jpl": 0.65, "office1": 0.85}
DT = 15 / 60.0
STEPS = 96
CAPACITY_MULT = {"normal": 1.0, "capacity_limited": 0.4, "arrival_burst": 1.0, "demand_shift": 1.0}

# 官方统一分时电价（USD/kWh），15 分钟一步。
TOU_PRICES = tuple(
    [0.6] * 24 + [0.9] * 8 + [1.2] * 16 + [0.8] * 8 + [1.1] * 16 + [1.5] * 16 + [0.7] * 8
)[:STEPS]


def load_training():
    training = {}
    for site in SITES:
        rows = json.loads(Path(f"data/training_{site}.json").read_text(encoding="utf-8"))
        training[site] = rows
    return training


def _finite(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _background_curve(base_peak_kw, steps=STEPS):
    """合成一条站级日负荷曲线，峰值约为 base_peak_kw 的 20%，夜间较低。"""
    curve = []
    for step in range(steps):
        hour = (step * DT) % 24.0
        shape = max(0.0, math.sin(math.pi * (hour - 7.0) / 12.0))
        curve.append(base_peak_kw * 0.20 * shape)
    return tuple(curve)


def _build_day_vehicles(rows, dt=DT, steps=STEPS):
    vehicles = []
    for idx, row in enumerate(rows):
        arrive = max(0, min(int(math.floor(float(row["hour"]) * 4)), steps - 1))
        dwell_steps = max(1, int(math.ceil(float(row["dwell_hours"]) / dt)))
        depart = min(arrive + dwell_steps, steps)
        requested = _finite(row.get("last_requested_energy_kwh"), math.nan)
        vehicles.append(
            {
                "token": f"v{idx}",
                "arrive": arrive,
                "depart": depart,
                "target": max(float(row["demand_kwh"]), 0.0),
                "limit": max(float(row["charger_limit_kw"]), 1e-6),
                "hour": float(row["hour"]),
                "weekday": int(row["weekday"]),
                "requested": requested,
            }
        )
    return vehicles


def _peak_concurrent_kw(vehicles, steps=STEPS):
    demand = [0.0] * steps
    for veh in vehicles:
        for t in range(veh["arrive"], veh["depart"]):
            demand[t] += veh["limit"]
    return max(demand) if demand else 0.0


def _apply_condition(vehicles, condition):
    if condition == "arrival_burst":
        out = []
        for veh in vehicles:
            out.append(veh)
            copy = dict(veh)
            copy["token"] = veh["token"] + "-b"
            out.append(copy)
        return out
    if condition == "demand_shift":
        return [{**veh, "target": veh["target"] * 1.35} for veh in vehicles]
    return list(vehicles)


def _simulate(vehicles, site, ev_capacity_kw, background, predict_fn, price_deferral):
    by_arrival = defaultdict(list)
    for veh in vehicles:
        by_arrival[veh["arrive"]].append(veh)

    delivered = {veh["token"]: 0.0 for veh in vehicles}
    active = []
    completed = 0
    total = len(vehicles)
    forecast_capacity = tuple(ev_capacity_kw + bg for bg in background)

    for t in range(STEPS):
        for veh in by_arrival.get(t, []):
            active.append(veh)

        observations = []
        for veh in active:
            if not (veh["arrive"] <= t < veh["depart"]):
                continue
            charged = delivered[veh["token"]]
            features = ArrivalFeatures(
                veh["token"],
                site,
                veh["hour"],
                veh["weekday"],
                veh["limit"],
                None if not math.isfinite(veh["requested"]) else veh["requested"],
            )
            observations.append(
                VehicleObservation(
                    features,
                    veh["arrive"],
                    t - veh["arrive"],
                    charged,
                    veh["target"] - charged <= 1e-9,
                )
            )

        observation = Observation(
            step=t,
            dt_hours=DT,
            horizon_steps=STEPS,
            site=site,
            local_hour=(t * DT) % 24.0,
            weekday=0,
            vehicles=tuple(observations),
            capacity_kw=ev_capacity_kw + background[t],
            background_kw=background[t],
            prices=TOU_PRICES,
            forecast_capacity=forecast_capacity,
            forecast_background=background,
            historical_arrivals=(0,) * STEPS,
            intervention_capacity=(),
        )
        power = act_edf(observation, predict_fn, price_deferral=price_deferral)

        leaving = []
        for veh in active:
            if not (veh["arrive"] <= t < veh["depart"]):
                continue
            granted = _finite(power.get(veh["token"], 0.0))
            need = max(veh["target"] - delivered[veh["token"]], 0.0)
            delivered[veh["token"]] += min(granted * DT, need)
            if veh["depart"] == t + 1:
                leaving.append(veh)

        for veh in leaving:
            remaining = max(veh["target"] - delivered[veh["token"]], 0.0)
            if remaining <= 0.03 * veh["target"] + 1e-9:
                completed += 1

        # 离开判断发生在上面；这里只保留下一步仍会留在场站的车。
        active = [veh for veh in active if veh["depart"] > t + 1]

    for veh in active:
        remaining = max(veh["target"] - delivered[veh["token"]], 0.0)
        if remaining <= 0.03 * veh["target"] + 1e-9:
            completed += 1

    return completed / total if total else 1.0, completed, total


def _perfect_predict(vehicles):
    lookup = {veh["token"]: veh for veh in vehicles}

    def predict_fn(features):
        veh = lookup[features.token]
        return Prediction(veh["target"], (veh["depart"] - veh["arrive"]) * DT)

    return predict_fn


def _real_predict(training):
    import pandas as pd

    from submission import build_model

    frames = {}
    for site in SITES:
        frame = pd.DataFrame(training[site])
        frame["day"] = pd.to_datetime(frame["day"])
        frames[site] = frame
    model = build_model(frames)
    return lambda features: model.predict(features)


def _run_condition(site, condition, day_vehicles, base_peak, real_predict):
    vehicles = _apply_condition(day_vehicles, condition)
    ev_capacity = base_peak * CAPACITY_MULT[condition]
    background = _background_curve(base_peak)

    predict_fns = {"perfect": _perfect_predict(vehicles)}
    if real_predict is not None:
        predict_fns["real"] = real_predict

    results = {}
    for predict_name, predict_fn in predict_fns.items():
        true_rate, _, _ = _simulate(vehicles, site, ev_capacity, background, predict_fn, True)
        false_rate, _, _ = _simulate(vehicles, site, ev_capacity, background, predict_fn, False)
        threshold = THRESHOLDS[site]
        ok = true_rate + 1e-9 >= threshold and false_rate + 1e-9 >= threshold
        defer_safe = true_rate + 1e-9 >= false_rate
        results[predict_name] = {
            "completion_defer_true": round(true_rate, 4),
            "completion_defer_false": round(false_rate, 4),
            "threshold": threshold,
            "ok": bool(ok),
            "defer_safe": bool(defer_safe),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=SITES, default=None)
    parser.add_argument("--days", type=int, default=4)
    parser.add_argument("--predict", choices=["perfect", "real", "both"], default="both")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    training = load_training()
    sites = [args.site] if args.site else list(SITES)
    conditions = ["normal", "capacity_limited", "arrival_burst", "demand_shift"]

    real_predict = None
    if args.predict in ("real", "both"):
        try:
            real_predict = _real_predict(training)
            print("真实 predict 已加载（submission.build_model）。")
        except Exception as exc:
            print(f"真实 predict 不可用，只跑完美预测：{exc}")
            if args.predict == "real":
                return 2

    report = {"kind": "completion-proxy", "days": args.days, "sites": {}}
    all_ok = True

    for site in sites:
        by_day = defaultdict(list)
        for row in training[site]:
            by_day[row["day"]].append(row)
        chosen = sorted(by_day.values(), key=len, reverse=True)[: max(1, args.days)]

        site_report = {"days": [], "failed_scenarios": []}
        for day_rows in chosen:
            day_vehicles = _build_day_vehicles(day_rows)
            if not day_vehicles:
                continue
            base_peak = _peak_concurrent_kw(day_vehicles)
            if base_peak <= 1e-6:
                continue

            day = day_rows[0]["day"]
            day_result = {
                "day": day,
                "vehicles": len(day_vehicles),
                "peak_kw": round(base_peak, 1),
                "conditions": {},
            }
            for condition in conditions:
                cond = _run_condition(site, condition, day_vehicles, base_peak, real_predict)
                day_result["conditions"][condition] = cond
                for predict_name, res in cond.items():
                    if not res["ok"] or not res["defer_safe"]:
                        all_ok = False
                        site_report["failed_scenarios"].append(
                            {"day": day, "condition": condition, "predict": predict_name, **res}
                        )

            site_report["days"].append(day_result)
            print(f"[{site}] {day} vehicles={len(day_vehicles)} peak={base_peak:.1f} kW")
            for condition in conditions:
                for predict_name, res in day_result["conditions"][condition].items():
                    flag = "OK" if res["ok"] and res["defer_safe"] else "FAIL"
                    print(
                        f"  {condition:>16} {predict_name:>7} "
                        f"defer_T={res['completion_defer_true']:.3f} "
                        f"defer_F={res['completion_defer_false']:.3f} "
                        f"threshold={res['threshold']:.2f} -> {flag}"
                    )

        report["sites"][site] = site_report

    verdict = "PASS" if all_ok else "FAIL"
    print("-" * 60)
    print(f"完成率闸门: {verdict}")
    if not all_ok:
        print("存在未达门槛或 price_deferral 拉低完成率的场景，不应提交。")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"verdict": verdict, **report}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("结果已写入:", args.out)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
