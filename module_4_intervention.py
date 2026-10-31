"""模块4：干预推演。

capacity_edf 反事实：用公开干预容量扣除背景负荷后的 EV 可用容量，
按最早离开优先分配，推演限流与后续回补。两种条件从同一状态出发，
只改变容量条件，不改变需求、到达、离开或已交付电量。
"""
from __future__ import annotations

from module_3_world_model import rollout


def forecast_capacity_edf(
    observation, horizon_steps, predict_fn, arrival_rate_fn
):
    cap_seq = observation.intervention_capacity or observation.forecast_capacity
    background_seq = observation.forecast_background

    def capacity_fn(step):
        cap = cap_seq[min(step, len(cap_seq) - 1)]
        background = background_seq[min(step, len(background_seq) - 1)]
        return max(cap - background, 0.0)

    return rollout(
        observation,
        horizon_steps,
        predict_fn,
        arrival_rate_fn,
        capacity_fn,
    )
