"""模块5：充电决策。

act 采用“未完成即满充 + 最小松弛优先（LLF）削减”：
- 任何 charging_complete=False 的车一律申请满功率 charger_limit_kw；
- 预测误差只影响排序（laxity），不再限制单辆功率上限；
- 站点容量不够时按 LLF 削减，laxity 越小越优先。

行动限额只来自 capacity_kw 与 forecast_capacity 的当前值，扣除背景负荷；
不使用 intervention_capacity。
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

    行动限额取 capacity_kw 与 forecast_capacity 当前值的最小者，再扣除背景
    负荷。intervention_capacity 是公开反事实容量，不能当行动限额。
    """
    capacity = _finite(observation.capacity_kw, 0.0)
    forecast = getattr(observation, "forecast_capacity", ())
    if forecast is not None and len(forecast) > 0:
        capacity = min(capacity, _finite(forecast[0], capacity))
    background = _finite(observation.background_kw, 0.0)
    return max(capacity - background, 0.0)


def act_edf(observation, predict_fn, price_deferral=None, dwell_fn=None):
    """返回 {token: 请求功率 kW}。

    未完成车辆一律请求满功率，容量不足时按 LLF 削减。``price_deferral`` 与
    ``dwell_fn`` 参数仅为接口兼容保留，当前不使用。
    """
    dt = _finite(observation.dt_hours, 0.25)
    if dt <= 0:
        dt = 0.25

    cars = []
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
        dwell_hours = _finite(pred.dwell_hours, dt)
        steps_left = max(
            1,
            math.ceil(
                dwell_hours / dt - _finite(vehicle.elapsed_steps, 0.0)
            ),
        )
        time_to_finish = remaining / (limit * dt)
        laxity = steps_left - time_to_finish

        cars.append(
            {
                "token": vehicle.features.token,
                "laxity": laxity,
                "steps_left": steps_left,
                "limit": limit,
            }
        )

    if not cars:
        return {}

    capacity = _ev_capacity_kw(observation)
    order = sorted(cars, key=lambda car: (car["laxity"], car["steps_left"]))
    power = {}
    for car in order:
        granted = min(car["limit"], capacity)
        power[car["token"]] = granted
        capacity -= granted
    return power
