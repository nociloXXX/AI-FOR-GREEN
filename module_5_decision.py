"""模块5：充电决策。

act 采用最小松弛优先（least-laxity-first）：
- 每辆在场未完成车计算 laxity = steps_left - time_to_finish，laxity 越小越紧急；
  time_to_finish = remaining_energy / (charger_limit_kw·dt)。
- 完成基线 = 尽快充（min(桩上限, 剩余需求/dt)），laxity <= safety_margin 的车
  立即满充；价格延后只作用于 laxity > safety_margin 的正松弛车辆，把它们的
  功率降到“在预测停留内完成”的必要功率，让位削峰，绝不延后紧车辆。

行动限额只来自 capacity_kw 与 forecast_capacity，绝不使用
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


def _green_signal(observation, horizon_steps=8):
    """返回 (当前是否便宜, 未来近期容量是否比现在更紧)。"""
    prices = getattr(observation, "prices", ()) or ()
    if prices:
        future = [float(p) for p in prices[1:]]
        cheap = not future or float(prices[0]) <= min(future)
    else:
        cheap = True

    current_ev = _ev_capacity_kw(observation)
    fc = getattr(observation, "forecast_capacity", ()) or ()
    fb = getattr(observation, "forecast_background", ()) or ()
    tight_ahead = False
    if fc:
        horizon = min(len(fc) - 1, max(1, int(horizon_steps)))
        headrooms = []
        for i in range(1, horizon + 1):
            cap = _finite(fc[i], current_ev)
            bg = _finite(fb[i], 0.0) if i < len(fb) else 0.0
            headrooms.append(max(cap - bg, 0.0))
        if headrooms:
            tight_ahead = min(headrooms) < current_ev
    return cheap, tight_ahead


# laxity 不超过该步数视为“现在已紧”：立即满充、禁止价格延后。
_SAFETY_MARGIN_STEPS = 1.0

# dwell 点预测有约 1.6h 误差：被高估时会把车辆误判成“有松弛”而延后、少充。
# 这里保守地把 dwell 估计缩短（假设车辆可能提前离开），完成率优先。
DWELL_SAFETY_FACTOR = 0.8

# 价格延后总开关：False 时 act 退化为纯完成率优先的尽快充，便于 A/B。
PRICE_DEFERRAL_ENABLED = True


def act_edf(observation, predict_fn, price_deferral=None):
    """返回 {token: 请求功率 kW}，只操作当前连接且未完成的车辆。"""
    dt = _finite(observation.dt_hours, 0.25)
    if dt <= 0:
        dt = 0.25

    cheap, tight_ahead = _green_signal(observation)
    defer_enabled = (
        PRICE_DEFERRAL_ENABLED if price_deferral is None else bool(price_deferral)
    )

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
        dwell_safe_hours = max(
            _finite(pred.dwell_hours, dt) * DWELL_SAFETY_FACTOR,
            dt,
        )
        steps_left = max(
            1,
            math.ceil(
                dwell_safe_hours / dt
                - _finite(vehicle.elapsed_steps, 0.0)
            ),
        )

        # 预测说快充完但车辆仍未完成：保持桩上限推进，发现真实剩余需求。
        discover = remaining < 0.05
        if discover:
            laxity = 0.0
            wanted = limit
            base = limit
            deferrable = False
        else:
            time_to_finish = remaining / (limit * dt)
            laxity = steps_left - time_to_finish
            necessary = min(limit, remaining / (steps_left * dt))
            base = min(limit, remaining / dt)
            if laxity <= _SAFETY_MARGIN_STEPS:
                # 现在已紧：立即满充，不参与价格延后。
                wanted = limit
                deferrable = False
            else:
                wanted = base
                deferrable = True

        cars.append(
            {
                "token": vehicle.features.token,
                "laxity": laxity,
                "wanted": wanted,
                "base": base,
                "necessary": necessary if not discover else limit,
                "deferrable": deferrable,
                "steps_left": steps_left,
            }
        )

    if not cars:
        return {}

    capacity = _ev_capacity_kw(observation)
    # 全局“现在已紧”硬条件：当前可用容量不足以让所有未完成车辆按自然满充
    # 功率充电时，整个步禁用价格延后，避免把本就不足的容量留成浪费。
    natural_demand = sum(car["base"] for car in cars)
    now_tight = natural_demand > capacity

    # 第二阶段（价格延后，可开关）：只把正松弛车辆从完成基线降到必要功率；
    # 高价且未来容量富余的步不前冲，让位削峰。绝不延后 laxity <= margin 的车。
    if defer_enabled:
        for car in cars:
            if car["deferrable"] and not (cheap or tight_ahead or now_tight):
                car["wanted"] = car["necessary"]

    # LLF 分配：laxity 小者优先，每辆最多给 wanted，直到可用容量用完。
    order = sorted(cars, key=lambda car: (car["laxity"], car["steps_left"]))
    power = {}
    for car in order:
        granted = min(car["wanted"], capacity)
        power[car["token"]] = granted
        capacity -= granted
    return power
