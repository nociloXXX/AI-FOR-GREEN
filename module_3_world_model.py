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


def _step_hour_weekday(observation, step, dt):
    """安全计算推演步的当地时刻与星期，非法输入一律归一。"""
    local_hour = _finite(observation.local_hour, default=0.0)
    weekday = int(_finite(observation.weekday, default=0.0))
    hour = (local_hour + step * dt) % 24.0
    return hour, (weekday + int((local_hour + step * dt) // 24)) % 7


def _remaining_steps(pred: Prediction, dt_hours: float, elapsed_steps: int) -> int:
    """预测剩余停留步数，至少保留一步，避免零除与车辆提前消失。"""
    dwell = _finite(getattr(pred, "dwell_hours", 0.0), default=dt_hours)
    elapsed = max(0, int(_finite(elapsed_steps, default=0.0)))
    steps = int(np.ceil(dwell / max(dt_hours, 1e-9) - elapsed))
    return max(1, steps)


def _poisson_quantile(rate, level):
    """泊松计数分布的 ``level`` 分位数：P(X <= k) >= level 的最小 k。"""
    rate = max(_finite(rate), 0.0)
    level = min(max(_finite(level, default=0.5), 0.0), 1.0)
    if rate <= 1e-12:
        return 0
    # 3 个标准差外几乎无概率，给大 rate 一个确定上界，避免无界循环。
    upper = int(rate + 10.0 * math.sqrt(rate) + 12)
    pmf = math.exp(-rate)
    cdf = pmf
    k = 0
    while k < upper:
        if cdf >= level:
            return k
        k += 1
        pmf *= rate / k
        cdf += pmf
    return upper


def _arrival_counts(observation, horizon_steps, arrival_rate_fn, quantiles):
    """按每步公开到达率生成各分位情景下的整数到达数。"""
    dt = max(_finite(observation.dt_hours, default=0.25), 1e-9)
    horizon = max(0, int(_finite(horizon_steps, default=0.0)))
    counts = {level: [0] * horizon for level in quantiles}
    for step in range(horizon):
        hour, weekday = _step_hour_weekday(observation, step, dt)
        rate = max(_finite(arrival_rate_fn(observation.site, hour, weekday)), 0.0)
        for level in quantiles:
            counts[level][step] = _poisson_quantile(rate, level)
    return counts


def _rollout_with_counts(
    observation,
    horizon_steps,
    predict_fn,
    capacity_fn,
    arrival_counts,
    priority_key=None,
    injected_predict=None,
):
    """给定未来每步整数到达数，执行一条确定性物理推演路径。"""
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
    for step in range(horizon):
        hour, weekday = _step_hour_weekday(observation, step, dt)
        for _ in range(int(arrival_counts[step])):
            features = ArrivalFeatures("forecast", observation.site, hour, weekday, 6.6)
            pred = (injected_predict or predict_fn)(features)
            demand = max(_finite(getattr(pred, "demand_kwh", 0.0)), 0.0)
            cars.append(
                [
                    demand,
                    _remaining_steps(pred, dt, 0),
                    6.6,
                ]
            )

        valid = [car for car in cars if car[0] > 1e-8 and car[1] > 0]
        active.append(int(len(valid)))
        cap = _finite(capacity_fn(step), default=0.0, allow_inf=True)
        load = 0.0
        key = priority_key or (lambda item: item[1])
        for car in sorted(valid, key=key):
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


def rollout(
    observation,
    horizon_steps,
    predict_fn,
    arrival_rate_fn,
    capacity_fn,
    priority_key=None,
    arrival_quantiles=(0.25, 0.5, 0.75),
    demand_quantiles=(0.5, 0.75, 0.9),
    predict_quantiles_fn=None,
    sample_fn=None,
    seed=0,
):
    """通用前向推演。

    车辆状态用 ``[剩余需求kWh, 剩余停留步数, 桩功率上限kW]`` 表示；每步先
    按公开到达率的分位情景补入新车，再统计仍在场且未完成的车数，按容量分配
    功率，最后全体车辆停留步数减一。``capacity_fn(step)`` 返回本步可用充电
    功率，传 ``float("inf")`` 即自然满功率充电。``priority_key`` 省略时按
    剩余停留步数升序（最早离开优先，EDF）分配。

    ``arrival_quantiles`` 给多个分位时，把各情景轨迹求平均作为稳健单路径；
    传空元组则退回旧的“期望到达累加器”口径。
    """
    if sample_fn is not None:
        # 条件密度/GMM 注入：单条采样路径。natural 与 capacity_edf 由调用方
        # 传同一个 sample_fn 与 seed，保证注入序列逐车一致。
        import random as _random

        rng = _random.Random(seed)
        dt = max(_finite(observation.dt_hours, default=0.25), 1e-9)
        horizon = max(0, int(_finite(horizon_steps, default=0.0)))
        counts = _arrival_counts(observation, horizon, arrival_rate_fn, (0.5,))[0.5]
        injected_predict = lambda features: sample_fn(features, rng)  # noqa: E731
        return _rollout_with_counts(
            observation,
            horizon,
            predict_fn,
            capacity_fn,
            counts,
            priority_key,
            injected_predict=injected_predict,
        )

    if not arrival_quantiles:
        # 旧口径：累积期望到达，满 1 补一辆。保留给需要严格期望路径的调用方。
        dt = max(_finite(observation.dt_hours, default=0.25), 1e-9)
        horizon = max(0, int(_finite(horizon_steps, default=0.0)))
        counts = [0] * horizon
        acc = 0.0
        for step in range(horizon):
            hour, weekday = _step_hour_weekday(observation, step, dt)
            acc += max(_finite(arrival_rate_fn(observation.site, hour, weekday)), 0.0)
            while acc >= 1.0:
                counts[step] += 1
                acc -= 1.0
        return _rollout_with_counts(
            observation, horizon, predict_fn, capacity_fn, counts, priority_key
        )

    arrival_counts = _arrival_counts(
        observation, horizon_steps, arrival_rate_fn, arrival_quantiles
    )
    levels = tuple(demand_quantiles or (0.5,))
    scenarios = [
        _rollout_with_counts(
            observation,
            horizon_steps,
            predict_fn,
            capacity_fn,
            arrival_counts[arrival_level],
            priority_key,
            injected_predict=(
                None
                if predict_quantiles_fn is None
                else (lambda features, q=dq: predict_quantiles_fn(features)[q])
            ),
        )
        for arrival_level in arrival_quantiles
        for dq in levels
    ]
    horizon = max(0, int(_finite(horizon_steps, default=0.0)))
    loads = [
        sum(scenario["ev_load_kw"][i] for scenario in scenarios) / len(scenarios)
        for i in range(horizon)
    ]
    active = [
        int(round(sum(scenario["active_count"][i] for scenario in scenarios) / len(scenarios)))
        for i in range(horizon)
    ]
    return {"ev_load_kw": loads, "active_count": active}


def forecast_natural(
    observation,
    horizon_steps,
    predict_fn,
    arrival_rate_fn,
    predict_quantiles_fn=None,
    sample_fn=None,
    seed=0,
):
    """自然充电推演：不施加站点容量限制，按桩功率上限充到完成或离开。"""
    return rollout(
        observation,
        horizon_steps,
        predict_fn,
        arrival_rate_fn,
        lambda _step: float("inf"),
        predict_quantiles_fn=predict_quantiles_fn,
        sample_fn=sample_fn,
        seed=seed,
    )
