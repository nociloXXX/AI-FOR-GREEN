"""模块5：充电决策。

act 按最早离开优先（EDF）在可用容量内为当前连接且未完成的车辆分配请求
功率。行动限额只来自 capacity_kw 与 forecast_capacity，绝不使用
intervention_capacity；每辆车不超过其 charger_limit_kw。
"""
from __future__ import annotations

import math


def _finite(value, default=0.0):
    """把输入转成有限 float；NaN/Inf/非法值一律退回 default，避免被拒收。"""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _ev_capacity_kw(observation):
    """当前步可用的电动车功率（kW）。

    行动限额是 capacity_kw 与 forecast_capacity 当前值的最小者，再扣除背景
    负荷。intervention_capacity 是公开反事实容量，不能当行动限额。
    """
    capacity = _finite(observation.capacity_kw, 0.0)
    forecast = getattr(observation, "forecast_capacity", ())
    if forecast is not None and len(forecast) > 0:
        capacity = min(capacity, _finite(forecast[0], capacity))
    background = _finite(observation.background_kw, 0.0)
    return max(capacity - background, 0.0)


def act_edf(observation, predict_fn):
    """返回 {token: 请求功率 kW}，只操作当前连接且未完成的车辆。"""
    dt = _finite(observation.dt_hours, 0.25)
    if dt <= 0:
        dt = 0.25

    request = []
    for vehicle in observation.vehicles:
        if vehicle.charging_complete:
            continue
        limit = _finite(vehicle.features.charger_limit_kw, 0.0)
        if limit <= 0:
            continue

        pred = predict_fn(vehicle.features)
        remaining = max(
            _finite(pred.demand_kwh, 0.0) - _finite(vehicle.delivered_kwh, 0.0),
            0.0,
        )
        # 预测说快充完但车辆仍未完成时，继续按桩上限推进以发现真实剩余需求。
        if remaining < 0.05:
            remaining = limit * dt

        steps_left = max(
            _finite(pred.dwell_hours, dt) / dt
            - _finite(vehicle.elapsed_steps, 0.0),
            1.0,
        )
        wanted = min(limit, remaining / dt)
        wanted = min(max(wanted, 0.0), limit)
        request.append((steps_left, vehicle.features.token, wanted))

    power = {}
    capacity = _ev_capacity_kw(observation)
    for _deadline, token, wanted in sorted(request):
        granted = min(wanted, capacity)
        power[token] = granted
        capacity -= granted
    return power
