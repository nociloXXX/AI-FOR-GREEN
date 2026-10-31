"""干预推演：基于世界模型对候选调度动作做前向推演与评分。

本模块只回答“如果采取这个干预，会发生什么”，不负责选择动作。充电决策
模块会生成多个候选干预，交给这里逐个推演，再按收益选优。

在线约束：推演只使用预测模块给出的预测值和当前世界模型状态，不读取未来
真实数据。``step_world`` 只推进一个时间步，``simulate_intervention`` 用
预测负荷合成“本步到达”的需求，再按候选调度逐段推演场站状态；任何接口
都不接受未来事件表。
"""
from __future__ import annotations

import copy

import numpy as np

from . import config as C


def step_world(
    state,
    dispatch_kw,
    step_hours=None,
    arrivals=(),
    price_yuan_per_kwh=None,
    power_limit_kw=None,
):
    """用世界模型推演一个时间步。

    ``dispatch_kw`` 是本站本步允许调用的总功率预算（kW）。它作为每根桩的
    期望功率上限传入 ``StationState.step``，由状态机内部的公平分配把总功率
    压到 ``min(dispatch_kw, 变压器容量)`` 以内，并逐桩扣除电量、释放完成
    的会话、再从队列补位。

    ``arrivals`` 只接受“本步到达”的会话，调用方不能把未来到达传进来。
    """
    step_h = float(step_hours) if step_hours is not None else C.STEP_MINUTES / 60.0
    budget = max(0.0, float(dispatch_kw))
    limit = min(budget, float(state.capacity_kw))
    if power_limit_kw is not None:
        limit = min(limit, max(0.0, float(power_limit_kw)))
    return state.step(
        arrivals=arrivals,
        per_pile_power_kw=budget,
        step_hours=step_h,
        power_limit_kw=limit,
        price_yuan_per_kwh=price_yuan_per_kwh,
    )


def _forward_simulate(state, pred, dispatch, capacity_kw, step_hours):
    """把候选调度按世界模型逐段推演，返回端到端物理指标。

    预测负荷 ``pred[t]`` 在本步被合成一个充电需求会话（电量守恒，峰值功率
    等于该步预测负荷），再以 ``dispatch[t]`` 为功率预算调用 ``step_world``。
    未在本步充完的电量保留在桩上或队列里，由后续步继续充，因此这里测到的
    “未满足”是推演终点真正没充上的电量，而不是每个时间片的瞬时缺口。
    """
    sim = copy.deepcopy(state)
    sim.capacity_kw = float(capacity_kw)

    total_demand_kwh = 0.0
    served_kwh = 0.0
    completions = 0
    max_queue = int(sim.queue_length)

    for t in range(pred.size):
        demand_kw = float(pred[t])
        demand_kwh = demand_kw * step_hours
        arrivals = []
        if demand_kwh > 1e-12:
            arrivals.append(
                {
                    "energy_kwh": demand_kwh,
                    "max_power_kw": max(demand_kw, 1e-6),
                }
            )
            total_demand_kwh += demand_kwh

        result = step_world(
            sim, float(dispatch[t]), step_hours=step_hours, arrivals=arrivals
        )
        served_kwh += float(result.served_energy_kwh)
        completions += int(result.completions)
        max_queue = max(max_queue, int(sim.queue_length))

    unserved_kwh = float(
        sum(pile.remaining_energy_kwh for pile in sim.piles)
        + sum(session.energy_kwh for session in sim.queue)
    )
    return {
        "total_demand_kwh": round(total_demand_kwh, 3),
        "served_energy_kwh": round(served_kwh, 3),
        "unserved_energy_kwh": round(unserved_kwh, 3),
        "completions": completions,
        "max_queue_length": max_queue,
        "utilization": round(sim.utilization, 6),
    }


def simulate_intervention(pred, dispatch, capacity_kw, step_hours=None, state=None):
    """对一段候选调度做前向推演评分，返回越限、削峰、弃充与收益。

    当给出 ``state`` 时，按世界模型逐段推演，并用推演终点真正没充上的电量
    作为弃充口径；否则退回纯曲线口径，用逐段正缺口 ``pred - dispatch`` 估算
    弃充。两种口径都只依赖预测值与当前状态，不读取未来真实数据。
    """
    pred = np.asarray(pred, dtype=float).reshape(-1)
    dispatch = np.asarray(dispatch, dtype=float).reshape(-1)
    if pred.size != dispatch.size:
        raise ValueError("预测负荷与调度长度不一致")

    step_h = float(step_hours) if step_hours is not None else C.STEP_MINUTES / 60.0
    cap = float(capacity_kw)
    if pred.size == 0:
        return {
            "violations": 0,
            "peak_before_kw": 0.0,
            "peak_after_kw": 0.0,
            "overflow_kwh": 0.0,
            "curtailment_kwh": 0.0,
            "unmet_kwh": 0.0,
            "score": 0.0,
        }

    violations = int((dispatch > cap + 1e-6).sum())
    peak_before = float(pred.max())
    peak_after = float(dispatch.max())
    overflow_kwh = float(np.maximum(pred - cap, 0.0).sum() * step_h)

    if state is not None:
        world = _forward_simulate(state, pred, dispatch, cap, step_h)
        curtailment_kwh = float(world["unserved_energy_kwh"])
    else:
        world = {}
        curtailment_kwh = float(np.maximum(pred - dispatch, 0.0).sum() * step_h)

    curtail_penalty = float(getattr(C, "INTERVENTION_CURTAIL_PENALTY", 2.0))
    violation_penalty = float(getattr(C, "INTERVENTION_VIOLATION_PENALTY", 1000.0))
    score = (
        overflow_kwh
        - curtail_penalty * curtailment_kwh
        - violation_penalty * violations
    )

    result = {
        "violations": violations,
        "peak_before_kw": round(peak_before, 3),
        "peak_after_kw": round(peak_after, 3),
        "overflow_kwh": round(overflow_kwh, 3),
        "curtailment_kwh": round(curtailment_kwh, 3),
        "unmet_kwh": round(curtailment_kwh, 3),
        "score": round(score, 3),
    }
    result.update(world)
    if state is not None:
        snap = state.snapshot()
        result["station_id"] = snap["station_id"]
        result["num_piles"] = snap["num_piles"]
        result["available_capacity_kw"] = snap["available_capacity_kw"]
    return result


def evaluate_candidates(pred, candidates, capacity_kw, step_hours=None, state=None):
    """对候选干预逐个推演，返回收益最高的调度与其评分表。"""
    items = candidates.items() if isinstance(candidates, dict) else candidates
    items = list(items)
    if not items:
        raise ValueError("候选干预列表为空，无法选优")

    best_dispatch = None
    best = None
    table = []
    for name, dispatch in items:
        result = simulate_intervention(pred, dispatch, capacity_kw, step_hours, state)
        result["name"] = name
        table.append(result)
        if best is None or result["score"] > best["score"]:
            best = result
            best_dispatch = np.asarray(dispatch, dtype=float)
    best["candidates"] = table
    return best_dispatch, best
