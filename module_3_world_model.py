"""模块3：世界模型。

基于预测的需求/停留与期望到达，做物理前向推演，输出未来负荷与活跃车辆数。
natural 条件不施加站点容量限制。
"""
from __future__ import annotations

import numpy as np

from acn_causal.student_api import ArrivalFeatures


def rollout(observation, horizon_steps, predict_fn, arrival_rate_fn, capacity_fn):
    """通用推演：车辆状态为 [剩余需求kWh, 剩余停留步数, 桩功率上限kW]。"""
    cars = []
    for vehicle in observation.vehicles:
        if vehicle.charging_complete:
            continue
        pred = predict_fn(vehicle.features)
        cars.append(
            [
                max(pred.demand_kwh - vehicle.delivered_kwh, 0.0),
                max(
                    1,
                    int(
                        np.ceil(
                            pred.dwell_hours / observation.dt_hours
                            - vehicle.elapsed_steps
                        )
                    ),
                ),
                vehicle.features.charger_limit_kw,
            ]
        )

    loads = []
    active = []
    acc = 0.0
    for step in range(horizon_steps):
        hour = (observation.local_hour + step * observation.dt_hours) % 24.0
        weekday = (
            observation.weekday
            + int((observation.local_hour + step * observation.dt_hours) // 24)
        ) % 7
        acc += arrival_rate_fn(observation.site, hour, weekday)
        while acc >= 1.0:
            pred = predict_fn(
                ArrivalFeatures("forecast", observation.site, hour, weekday, 6.6)
            )
            cars.append(
                [
                    pred.demand_kwh,
                    max(1, int(np.ceil(pred.dwell_hours / observation.dt_hours))),
                    6.6,
                ]
            )
            acc -= 1.0

        valid = [car for car in cars if car[0] > 1e-8 and car[1] > 0]
        active.append(len(valid))
        cap = capacity_fn(step)
        load = 0.0
        for car in sorted(valid, key=lambda item: item[1]):
            power = min(car[2], car[0] / observation.dt_hours, cap)
            car[0] -= power * observation.dt_hours
            cap -= power
            load += power
        for car in cars:
            car[1] -= 1
        loads.append(load)
    return {"ev_load_kw": loads, "active_count": active}


def forecast_natural(observation, horizon_steps, predict_fn, arrival_rate_fn):
    """自然充电：不施加站点容量限制。"""
    return rollout(
        observation,
        horizon_steps,
        predict_fn,
        arrival_rate_fn,
        lambda _step: float("inf"),
    )
