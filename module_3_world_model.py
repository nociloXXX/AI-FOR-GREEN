"""模块3：世界模型。

基于预测的需求/停留与期望到达，从当前观测状态出发做物理前向推演，输出
未来 ``horizon_steps`` 步的 EV 负荷与活跃车辆数。``natural`` 条件不施加
站点容量限制；``capacity_edf`` 按公开容量扣除背景负荷后在容量内按预测
离开最早优先分配。两种条件共享同一套需求、到达与离开假设。

在线约束：本模块只读取 ``Observation`` 的当前状态与公开序列，未来车辆只
按 ``arrival_rate_fn`` 的期望到达数补入，绝不读取未来到达、离开或真值。
"""
from __future__ import annotations

import math

import numpy as np

from acn_causal.student_api import ArrivalFeatures, Prediction


def _finite(value, default=0.0, allow_inf=False):
    """把数值转成有限 float；NaN/Inf/非法值退回 default（capacity 可放行 +inf）。"""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(value):
        return float(default)
    if allow_inf and value == math.inf:
        return value
    if not math.isfinite(value):
        return float(default)
    return value


def _remaining_steps(pred: Prediction, dt_hours: float, elapsed_steps: int) -> int:
    """预测剩余停留步数，至少保留一步，避免零除与车辆提前消失。"""
    dwell = _finite(getattr(pred, "dwell_hours", 0.0), default=dt_hours)
    elapsed = max(0, int(_finite(elapsed_steps, default=0.0)))
    steps = int(np.ceil(dwell / max(dt_hours, 1e-9) - elapsed))
    return max(1, steps)


def rollout(observation, horizon_steps, predict_fn, arrival_rate_fn, capacity_fn):
    """通用前向推演。

    车辆状态用 ``[剩余需求kWh, 剩余停留步数, 桩功率上限kW]`` 表示；每步先
    按期望到达补入新车，再统计仍在场且未完成的车数，按容量分配功率，最后
    全体车辆停留步数减一。``capacity_fn(step)`` 返回本步可用充电功率，传
    ``float("inf")`` 即自然满功率充电。
    """
    dt = max(_finite(observation.dt_hours, default=0.25), 1e-9)
    horizon = max(0, int(_finite(horizon_steps, default=0.0)))
    cars = []
    for vehicle in observation.vehicles:
        if vehicle.charging_complete:
            continue
        pred = predict_fn(vehicle.features)
        demand = max(
            _finite(getattr(pred, "demand_kwh", 0.0), default=0.0)
            - _finite(vehicle.delivered_kwh, default=0.0),
            0.0,
        )
        limit = max(
            _finite(vehicle.features.charger_limit_kw, default=0.0),
            0.0,
        )
        cars.append(
            [
                demand,
                _remaining_steps(pred, dt, vehicle.elapsed_steps),
                limit,
            ]
        )

    loads = []
    active = []
    acc = 0.0
    for step in range(horizon):
        hour = (observation.local_hour + step * dt) % 24.0
        weekday = (
            observation.weekday
            + int((observation.local_hour + step * dt) // 24)
        ) % 7
        rate = max(_finite(arrival_rate_fn(observation.site, hour, weekday)), 0.0)
        acc += rate
        while acc >= 1.0:
            pred = predict_fn(
                ArrivalFeatures("forecast", observation.site, hour, weekday, 6.6)
            )
            demand = max(_finite(getattr(pred, "demand_kwh", 0.0)), 0.0)
            cars.append(
                [
                    demand,
                    _remaining_steps(pred, dt, 0),
                    6.6,
                ]
            )
            acc -= 1.0

        valid = [car for car in cars if car[0] > 1e-8 and car[1] > 0]
        active.append(int(len(valid)))
        cap = _finite(capacity_fn(step), default=0.0, allow_inf=True)
        load = 0.0
        for car in sorted(valid, key=lambda item: item[1]):
            power = min(car[2], car[0] / dt, cap)
            car[0] -= power * dt
            cap -= power
            load += power
        for car in cars:
            car[1] -= 1
        loads.append(float(load))
    loads = [max(_finite(v), 0.0) for v in loads]
    loads += [0.0] * (horizon - len(loads))
    active += [0] * (horizon - len(active))
    return {"ev_load_kw": loads, "active_count": active}


def forecast_natural(observation, horizon_steps, predict_fn, arrival_rate_fn):
    """自然充电推演：不施加站点容量限制，按桩功率上限充到完成或离开。"""
    return rollout(
        observation,
        horizon_steps,
        predict_fn,
        arrival_rate_fn,
        lambda _step: float("inf"),
    )
