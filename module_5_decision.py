"""模块5：充电决策。

act 按 least-slack（最小松弛时间）优先，在可用容量内为当前连接且未完成的
车辆分配请求功率，并结合分时电价做成本前冲/延后。行动限额只来自
capacity_kw 与 forecast_capacity，绝不使用 intervention_capacity；每辆车
不超过其 charger_limit_kw。
"""
from __future__ import annotations

import math
import statistics


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


def _cheap_now(prices):
    """当前电价是否不高于今日剩余时段的电价中位数。"""
    if not prices:
        return True
    future = [float(p) for p in prices[1:]]
    if not future:
        return True
    try:
        reference = statistics.median(future)
    except statistics.StatisticsError:
        reference = float(prices[0])
    return float(prices[0]) <= reference


# 电价延后只在车辆有至少这一步的富余时启用，完成率优先于成本。
_PRICE_DEFER_MARGIN_STEPS = 1.0


def act_edf(observation, predict_fn):
    """返回 {token: 请求功率 kW}，只操作当前连接且未完成的车辆。"""
    dt = _finite(observation.dt_hours, 0.25)
    if dt <= 0:
        dt = 0.25

    prices = getattr(observation, "prices", ()) or ()
    cheap = _cheap_now(prices)

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
        steps_left = max(
            _finite(pred.dwell_hours, dt) / dt
            - _finite(vehicle.elapsed_steps, 0.0),
            1.0,
        )

        # 预测说快充完但车辆仍未完成：保持桩上限推进，发现真实剩余需求。
        if remaining < 0.05:
            request.append((0.0, steps_left, vehicle.features.token, limit))
            continue

        # slack=剩余停留步数-剩余需求/(桩上限*dt)。负值表示按时满功率也
        # 充不完，必须优先；正值表示有弹性，可前冲或延后。
        demand_steps = remaining / (limit * dt)
        slack = steps_left - demand_steps
        just_in_time = remaining / (steps_left * dt)

        if slack <= 0 or slack < _PRICE_DEFER_MARGIN_STEPS or cheap:
            wanted = limit
        else:
            wanted = min(just_in_time, limit)
        wanted = min(max(wanted, 0.0), limit)
        request.append((slack, steps_left, vehicle.features.token, wanted))

    power = {}
    capacity = _ev_capacity_kw(observation)
    for _slack, _steps_left, token, wanted in sorted(request):
        granted = min(wanted, capacity)
        power[token] = granted
        capacity -= granted
    return power
