"""模块4：干预推演。

capacity_edf 反事实：用公开 intervention_capacity 减 forecast_background 后的
EV 可用容量，按最早离开优先（EDF）分配，推演限流与后续回补。两种条件从同
一状态出发，只改变容量条件，不改变需求、到达、离开或已交付电量。

``strategy="edf"`` 与官方反事实定义一致，是提交默认策略；``strategy="slack"``
是供离线对照的备选优先级（按“剩余停留步数 - 按上限充满所需步数”的 slack 升
序，即最可能完不成的车优先），不改变需求/到达/离开假设，只改变容量内的分配
顺序。
"""
from __future__ import annotations

from module_3_world_model import rollout


def _ev_capacity_sequence(observation, horizon_steps):
    """构造反事实情景下每个推演步的 EV 可用容量（kW）。

    ``intervention_capacity`` 是反事实总站点容量（含背景负荷），扣除预测
    背景负荷后才是电动车可用功率。只取前 ``horizon_steps`` 步，序列不足时
    按末值沿用；容量恒取非负。
    """
    total = observation.intervention_capacity
    if total is None or len(total) == 0:
        total = observation.forecast_capacity
    if total is None or len(total) == 0:
        # 评测恒会提供容量序列；这里只为本地/异常输入兜底。
        return [0.0] * horizon_steps

    background = observation.forecast_background
    capacities = []
    for step in range(horizon_steps):
        cap = float(total[min(step, len(total) - 1)])
        bg = (
            float(background[min(step, len(background) - 1)])
            if background is not None and len(background) > 0
            else 0.0
        )
        capacities.append(max(cap - bg, 0.0))
    return capacities


def _priority_key(strategy, dt_hours):
    """返回 rollouts 的车辆排序键；``strategy="slack"`` 提供非 EDF 的对照。"""
    if strategy == "slack":
        dt = max(float(dt_hours), 1e-9)

        def key(car):
            energy, steps, limit = car
            needed = energy / max(float(limit) * dt, 1e-9)
            return (steps - needed, steps)

        return key
    # 默认 EDF：按剩余停留步数升序（最早离开优先）。
    return None


def forecast_capacity_edf(
    observation,
    horizon_steps,
    predict_fn,
    arrival_rate_fn,
    strategy="edf",
    predict_quantiles_fn=None,
    sample_fn=None,
    seed=0,
    future_cohorts_fn=None,
    survival_fn=None,
    arrival_calibration_fn=None,
    occupancy_multiplier_fn=None,
):
    """capacity_edf 反事实推演：同一状态，仅容量条件改为容量内限流。

    与 ``module_3_world_model.forecast_natural`` 共用同一个 ``rollout``，车辆
    初始状态、需求/停留预测与未来到达序列完全一致；唯一差异是本函数传入的
    每步 EV 可用容量。被限流的电量保留在剩余需求中，由后续步回补，因此不会
    复制 natural 轨迹，能给出限流与回补的响应差。``strategy`` 只改变容量内
    的车辆分配顺序，不影响容量序列与需求/到达/离开假设。
    """
    capacities = _ev_capacity_sequence(observation, horizon_steps)
    return rollout(
        observation,
        horizon_steps,
        predict_fn,
        arrival_rate_fn,
        lambda step: capacities[step],
        priority_key=_priority_key(strategy, observation.dt_hours),
        predict_quantiles_fn=predict_quantiles_fn,
        sample_fn=sample_fn,
        seed=seed,
        future_cohorts_fn=future_cohorts_fn,
        survival_fn=survival_fn,
        arrival_calibration_fn=arrival_calibration_fn,
        occupancy_multiplier_fn=occupancy_multiplier_fn,
    )
