"""干预推演：基于世界模型对候选调度动作做前向推演与评分。

本模块只回答“如果采取这个干预，会发生什么”，不负责选择动作。充电决策
模块会生成多个候选干预，交给这里逐个推演，再按收益选优。

在线约束：推演只使用预测模块给出的预测值和当前世界模型状态，不读取未来
真实数据。默认候选中的 mpc 只使用预测值做小步长前瞻。
"""
from __future__ import annotations

import numpy as np

from . import config as C


def step_world(state, dispatch_kw, step_hours=None):
    """用世界模型推演一个时间步。

    把场站级调度功率均摊到当前占用桩，调用 ``StationState.step`` 得到
    本步服务电量、弃充电量与排队长度等结果。
    """
    step_h = float(step_hours) if step_hours is not None else C.STEP_MINUTES / 60.0
    per_pile = float(dispatch_kw) / max(1, state.occupied)
    return state.step(per_pile_power_kw=per_pile, step_hours=step_h)


def simulate_intervention(pred, dispatch, capacity_kw, step_hours=None, state=None):
    """对一段候选调度做推演评分，返回越限、削峰、弃充与收益。"""
    pred = np.asarray(pred, dtype=float).reshape(-1)
    dispatch = np.asarray(dispatch, dtype=float).reshape(-1)
    if pred.size != dispatch.size:
        raise ValueError("预测负荷与调度长度不一致")

    step_h = float(step_hours) if step_hours is not None else C.STEP_MINUTES / 60.0
    cap = float(capacity_kw)

    violations = int((dispatch > cap + 1e-6).sum())
    peak_before = float(pred.max()) if pred.size else 0.0
    peak_after = float(dispatch.max()) if dispatch.size else 0.0
    overflow_kwh = float(np.maximum(pred - cap, 0.0).sum() * step_h)
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
    if state is not None:
        snap = state.snapshot()
        result["station_id"] = snap["station_id"]
        result["num_piles"] = snap["num_piles"]
        result["available_capacity_kw"] = snap["available_capacity_kw"]
    return result


def evaluate_candidates(pred, candidates, capacity_kw, step_hours=None, state=None):
    """对候选干预逐个推演，返回收益最高的调度与其评分表。"""
    best_dispatch = None
    best = None
    table = []
    for name, dispatch in candidates:
        result = simulate_intervention(pred, dispatch, capacity_kw, step_hours, state)
        result["name"] = name
        table.append(result)
        if best is None or result["score"] > best["score"]:
            best = result
            best_dispatch = np.asarray(dispatch, dtype=float)
    best["candidates"] = table
    return best_dispatch, best
