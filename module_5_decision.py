"""模块5：充电决策。

act 用 TOPSIS 多指标评分（剩余需求、停留紧急性、电价价差）确定充电优先级，
在可用容量内为当前连接且未完成的车辆分配功率；超过容量时按优先级实时削减。
行动限额只来自 capacity_kw 与 forecast_capacity，绝不使用
intervention_capacity；每辆车不超过其 charger_limit_kw。
"""
from __future__ import annotations

import math

import numpy as np


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
    """返回 (当前是否便宜, 未来近期容量是否比现在更紧)。

    便宜：当前价不高于今日剩余时段电价中位数。容量变紧：未来 horizon 步内
    的最小 EV 可用容量（forecast_capacity - forecast_background）低于当前值，
    此时即使电价不低也应提前多充，避免未来被容量卡住。
    """
    prices = getattr(observation, "prices", ()) or ()
    if prices:
        future = np.asarray([float(p) for p in prices[1:]], dtype=float)
        cheap = future.size == 0 or float(prices[0]) <= float(np.median(future))
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


def _price_advantage(prices, steps_left):
    """充电“现在 vs 延后”的价差：未来窗口价格中位数减当前价，正=现在更划算。"""
    if not prices:
        return 0.0
    current = float(prices[0])
    horizon = max(1, int(steps_left))
    future = np.asarray([float(p) for p in prices[1 : 1 + horizon]], dtype=float)
    if future.size == 0:
        return 0.0
    return float(np.median(future)) - current


# slack 粗排序步长：所有车先按 0.5 步分桶保住 least-slack 的完成率，
# 桶内再用 TOPSIS（剩余需求、停留、电价）精排做多指标权衡。
_SLACK_BUCKET_STEPS = 0.5

# TOPSIS 指标权重：剩余需求、停留紧急性、电价价差。前两项决定完成率，
# 权重更高；电价只影响削峰与成本，权重较低。
_TOPSIS_WEIGHTS = np.array([0.45, 0.35, 0.20], dtype=float)


def _topsis_scores(matrix, weights):
    """对 n 辆车 x 3 个“越大越优先”指标做 TOPSIS 评分，返回 [0,1] 分数。"""
    n = matrix.shape[0]
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.array([1.0])

    norms = np.sqrt((matrix ** 2).sum(axis=0))
    norms = np.where(norms > 1e-12, norms, 1.0)
    weighted = (matrix / norms) * weights

    best = weighted.max(axis=0)
    worst = weighted.min(axis=0)
    d_best = np.sqrt(((weighted - best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((weighted - worst) ** 2).sum(axis=1))
    denom = d_best + d_worst
    return np.where(denom > 1e-12, d_worst / np.maximum(denom, 1e-12), 0.5)


def act_edf(observation, predict_fn):
    """返回 {token: 请求功率 kW}，只操作当前连接且未完成的车辆。"""
    dt = _finite(observation.dt_hours, 0.25)
    if dt <= 0:
        dt = 0.25

    prices = getattr(observation, "prices", ()) or ()
    cheap, tight_ahead = _green_signal(observation)

    rows = []
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
        discover = remaining < 0.05
        if discover:
            score_remaining = limit * dt
            slack = 0.0
            wanted = limit
        else:
            score_remaining = remaining
            demand_steps = remaining / (limit * dt)
            slack = steps_left - demand_steps
            just_in_time = remaining / (steps_left * dt)
            if slack <= 0:
                wanted = limit
            elif cheap or tight_ahead:
                wanted = limit
            else:
                wanted = min(just_in_time, limit)
            wanted = min(max(wanted, 0.0), limit)

        rows.append(
            {
                "token": vehicle.features.token,
                "wanted": wanted,
                "steps_left": steps_left,
                "slack": slack,
                "criteria": (
                    score_remaining,
                    1.0 / steps_left,
                    _price_advantage(prices, steps_left),
                ),
            }
        )

    if not rows:
        return {}

    matrix = np.array([row["criteria"] for row in rows], dtype=float)
    scores = _topsis_scores(matrix, _TOPSIS_WEIGHTS)

    def sort_key(idx):
        row = rows[idx]
        bucket = math.floor(row["slack"] / _SLACK_BUCKET_STEPS)
        return (bucket, -float(scores[idx]), row["steps_left"])

    order = sorted(range(len(rows)), key=sort_key)

    power = {}
    capacity = _ev_capacity_kw(observation)
    for idx in order:
        token = rows[idx]["token"]
        granted = min(rows[idx]["wanted"], capacity)
        power[token] = granted
        capacity -= granted
    return power
