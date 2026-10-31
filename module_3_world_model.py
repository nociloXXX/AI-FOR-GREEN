"""模块3：世界模型。

基于预测的需求/停留与期望到达，从当前观测状态出发做物理前向推演，输出
未来 ``horizon_steps`` 步的 EV 负荷与活跃车辆数。``natural`` 条件不施加
站点容量限制；``capacity_edf`` 按公开容量扣除背景负荷后在容量内按预测
离开最早优先分配。两种条件共享同一套需求、到达与离开假设。

在线约束：本模块只读取 ``Observation`` 的当前状态与公开序列，未来车辆只
按 ``arrival_rate_fn`` 的期望到达数补入，绝不读取未来到达、离开或真值。
"""
from __future__ import annotations

import numpy as np

from acn_causal.student_api import ArrivalFeatures, Observation, Prediction


def _remaining_steps(pred: Prediction, dt_hours: float, elapsed_steps: int) -> int:
    """预测剩余停留步数，至少保留一步，避免零除与车辆提前消失。"""
    steps = int(np.ceil(pred.dwell_hours / max(dt_hours, 1e-9) - elapsed_steps))
    return max(1, steps)


def rollout(observation, horizon_steps, predict_fn, arrival_rate_fn, capacity_fn):
    """通用前向推演。

    车辆状态用 ``[剩余需求kWh, 剩余停留步数, 桩功率上限kW]`` 表示；每步先
    按期望到达补入新车，再统计仍在场且未完成的车数，按容量分配功率，最后
    全体车辆停留步数减一。``capacity_fn(step)`` 返回本步可用充电功率，传
    ``float("inf")`` 即自然满功率充电。
    """
    dt = max(float(observation.dt_hours), 1e-9)
    cars = []
    for vehicle in observation.vehicles:
        if vehicle.charging_complete:
            continue
        pred = predict_fn(vehicle.features)
        cars.append(
            [
                max(pred.demand_kwh - vehicle.delivered_kwh, 0.0),
                _remaining_steps(pred, dt, vehicle.elapsed_steps),
                vehicle.features.charger_limit_kw,
            ]
        )

    loads = []
    active = []
    acc = 0.0
    for step in range(horizon_steps):
        hour = (observation.local_hour + step * dt) % 24.0
        weekday = (
            observation.weekday
            + int((observation.local_hour + step * dt) // 24)
        ) % 7
        acc += arrival_rate_fn(observation.site, hour, weekday)
        while acc >= 1.0:
            pred = predict_fn(
                ArrivalFeatures("forecast", observation.site, hour, weekday, 6.6)
            )
            cars.append(
                [
                    pred.demand_kwh,
                    _remaining_steps(pred, dt, 0),
                    6.6,
                ]
            )
            acc -= 1.0

        valid = [car for car in cars if car[0] > 1e-8 and car[1] > 0]
        active.append(int(len(valid)))
        cap = capacity_fn(step)
        load = 0.0
        for car in sorted(valid, key=lambda item: item[1]):
            power = min(car[2], car[0] / dt, cap)
            car[0] -= power * dt
            cap -= power
            load += power
        for car in cars:
            car[1] -= 1
        loads.append(float(load))
    return {
        "ev_load_kw": [max(0.0, float(v)) for v in loads],
        "active_count": active,
    }


def forecast_natural(observation, horizon_steps, predict_fn, arrival_rate_fn):
    """自然充电推演：不施加站点容量限制，按桩功率上限充到完成或离开。"""
    return rollout(
        observation,
        horizon_steps,
        predict_fn,
        arrival_rate_fn,
        lambda _step: float("inf"),
    )


def forecast_capacity_edf(observation, horizon_steps, predict_fn, arrival_rate_fn):
    """容量 EDF 推演：公开容量扣除背景负荷后，按预测离开最早优先充电。"""
    cap_seq = observation.intervention_capacity or observation.forecast_capacity
    background_seq = observation.forecast_background

    def capacity_fn(step):
        if not cap_seq:
            return 0.0
        if not background_seq:
            background = 0.0
        else:
            background = background_seq[min(step, len(background_seq) - 1)]
        return max(float(cap_seq[min(step, len(cap_seq) - 1)]) - float(background), 0.0)

    return rollout(
        observation,
        horizon_steps,
        predict_fn,
        arrival_rate_fn,
        capacity_fn,
    )


class WorldModel:
    """供提交入口使用的世界模型门面。

    ``forecast`` 统一返回 ``{"ev_load_kw": [...], "active_count": [...]}``，
    两个序列长度都等于 ``horizon_steps``；natural 与 capacity_edf 从同一
    状态出发，只改变容量条件与分配顺序。
    """

    def __init__(self, predict_fn, arrival_rate_fn):
        self.predict_fn = predict_fn
        self.arrival_rate_fn = arrival_rate_fn

    def forecast(self, observation, horizon_steps, intervention):
        if intervention == "capacity_edf":
            return forecast_capacity_edf(
                observation, horizon_steps, self.predict_fn, self.arrival_rate_fn
            )
        return forecast_natural(
            observation, horizon_steps, self.predict_fn, self.arrival_rate_fn
        )
