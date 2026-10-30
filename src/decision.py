"""有序充电与调度决策。

职责：基于状态建模与预测模块的输出，决定每根桩的功率分配、排队顺序与
充放电动作。策略按复杂度分层，低阶策略可解释、依赖少，保证链路能闭环：

- causal_load_shift    在线安全的削峰填谷，只向后看，不使用任何未来信息（默认）
- greedy_priority      规则/贪心兜底（先到先充、低 SOC 优先、最早离场优先）
- lp_load_shift        连续功率分配的“注水法”LP，全量离线分析专用，禁止在线使用
- milp_schedule        开关桩/整数变量，优先 OR-Tools/PuLP，缺失时回退贪心
- mpc_schedule         滚动优化，仅使用预测模块给到的未来一小段“预测值”，非真实数据
- battery_arbitrage    储能峰谷套利 + 削峰，阈值改为因果滚动均值，带 SOC 与效率约束

硬约束统一为：单桩功率不超上限、场站总功率不超变压器容量、SOC 不越界、
能量服务总量不超需求。每个策略都输出可解释的分配依据。

在线评测隐藏未来数据：默认策略 ``causal_load_shift`` 与 ``greedy_priority``、
``battery_arbitrage`` 均为因果实现，第 t 步的决策只依赖 0..t 已见数据。
``lp_load_shift`` 需要整段曲线，只用于离线报告，上线前不要接到在线链路。
"""
from collections import defaultdict

import numpy as np

from . import config as C


def _water_fill(load, capacity_kw):
    """连续 LP 的闭式解：把负荷曲线投影到 [0, capacity] 且总量守恒。

    对 ``min sum((d_i - p_i)^2)`` 做 KKT 求导可得 ``d_i = clip(p_i + mu, 0, c)``，
    ``mu`` 用二分搜索求解，等价于标准“注水法”。仅当总需求超过时段总容量时才
    允许少量弃充，此时输出退化为全部压在容量线上。
    """
    p = np.asarray(load, dtype=float).reshape(-1)
    if p.size == 0:
        return np.zeros(0)
    cap = float(capacity_kw)
    total = float(p.sum())
    target = min(total, p.size * cap)

    lo = float(-p.max())
    hi = float(cap - p.min())
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(np.clip(p + mid, 0.0, cap).sum()) > target:
            hi = mid
        else:
            lo = mid
    mu = 0.5 * (lo + hi)
    dispatch = np.clip(p + mu, 0.0, cap)

    # 二分精度足够时该残差约为 1e-12，这里再做一次精确找平，保证总量守恒。
    residual = target - float(dispatch.sum())
    if abs(residual) > 1e-9:
        if residual > 0:
            slots = np.flatnonzero(dispatch < cap)
            order = np.argsort(p[slots])[::-1]
            for idx in slots[order]:
                step = min(residual, cap - dispatch[idx])
                dispatch[idx] += step
                residual -= step
                if residual <= 1e-12:
                    break
        else:
            slots = np.flatnonzero(dispatch > 0)
            order = np.argsort(p[slots])
            for idx in slots[order]:
                step = min(-residual, dispatch[idx])
                dispatch[idx] -= step
                residual += step
                if residual >= -1e-12:
                    break
    return dispatch


def _fill_cheapest_first(load, capacity_kw, price):
    """电价加权平移：把超容电量优先塞进电价最低的谷段，能量仍守恒。"""
    p = np.asarray(load, dtype=float).reshape(-1)
    price = np.asarray(price, dtype=float).reshape(-1)
    if price.size != p.size:
        price = np.full(p.size, float(np.median(price)))

    cap = float(capacity_kw)
    base = np.minimum(p, cap)
    headroom = np.maximum(cap - base, 0.0)
    remaining = float(np.maximum(p - cap, 0.0).sum())
    shifted = np.zeros_like(p)

    if remaining > 0 and headroom.sum() > 1e-9:
        for idx in np.argsort(price):
            room = headroom[idx]
            if room <= 1e-12:
                continue
            take = min(room, remaining)
            shifted[idx] = take
            remaining -= take
            if remaining <= 1e-12:
                break
    return base + shifted


def causal_load_shift(predicted_power_kw, capacity_kw, step_hours=None):
    """在线安全的削峰填谷：只向后看，不使用任何未来信息。

    第 t 步只依赖 0..t 的已见数据：当前负荷超过容量时，把超出电量记入
    待服务积压；当前有裕量时优先消化之前积压。能量守恒，仅当后续没有
    足够谷段时才把未消化的积压计为弃充。这是默认策略，可安全接入在线评测。
    """
    pred = np.asarray(predicted_power_kw, dtype=float).reshape(-1)
    cap = float(capacity_kw)
    step_h = float(step_hours) if step_hours else C.STEP_MINUTES / 60.0
    dispatch = np.zeros_like(pred)
    backlog_kwh = 0.0

    for t, load in enumerate(pred):
        if load > cap:
            dispatch[t] = cap
            backlog_kwh += (load - cap) * step_h
        else:
            headroom = cap - load
            serve_kw = min(backlog_kwh / step_h, headroom) if backlog_kwh > 0 else 0.0
            dispatch[t] = load + serve_kw
            backlog_kwh = max(backlog_kwh - serve_kw * step_h, 0.0)

    deferred_kwh = float(np.maximum(pred - cap, 0.0).sum() * step_h)
    info = {
        "strategy": "causal_load_shift",
        "peak_before_kw": float(pred.max()) if pred.size else 0.0,
        "peak_after_kw": float(dispatch.max()) if dispatch.size else 0.0,
        "shifted_kwh": float(deferred_kwh - backlog_kwh),
        "curtailment_kwh": float(backlog_kwh),
        "violations": int((dispatch > cap + 1e-6).sum()),
    }
    return dispatch, info


def lp_load_shift(predicted_power_kw, capacity_kw, price=None, step_hours=None):
    """连续功率分配（LP）：在容量约束内能量守恒地平移负荷。

    返回 ``(dispatch, info)``。无电价时用注水法平滑曲线；有电价时把峰段
    电量优先搬到最便宜的谷段，兼顾电费与碳排。注意：本函数按整段曲线
    求全局水位，属于离线全量分析，在线评测禁用。
    """
    pred = np.asarray(predicted_power_kw, dtype=float).reshape(-1)
    cap = float(capacity_kw)
    if price is not None:
        dispatch = _fill_cheapest_first(pred, cap, price)
    else:
        dispatch = _water_fill(pred, cap)

    step_h = float(step_hours) if step_hours else C.STEP_MINUTES / 60.0
    overflow = np.maximum(pred - cap, 0.0)
    info = {
        "strategy": "lp_load_shift",
        "peak_before_kw": float(pred.max()) if pred.size else 0.0,
        "peak_after_kw": float(dispatch.max()) if dispatch.size else 0.0,
        "shifted_kwh": float(overflow.sum() * step_h),
        "curtailment_kwh": float(np.maximum(pred - dispatch, 0.0).sum() * step_h),
        "violations": int((dispatch > cap + 1e-6).sum()),
    }
    return dispatch, info


def mpc_schedule(predicted_power_kw, capacity_kw, horizon_steps=None, price=None):
    """模型预测控制：滚动窗口只取第一步，用未来一小段预测值做有前瞻调度。

    这里的前瞻只来自预测模块给出的 ``predicted_power_kw``（预测值），不会
    读取任何未来真实数据；在线评测若要求零前瞻，直接使用默认的
    ``causal_load_shift`` 即可。
    """
    pred = np.asarray(predicted_power_kw, dtype=float).reshape(-1)
    horizon = max(1, int(horizon_steps or getattr(C, "MPC_HORIZON_STEPS", 4)))
    prices = np.asarray(price, dtype=float).reshape(-1) if price is not None else None

    dispatch = np.empty_like(pred)
    for t in range(pred.size):
        window = pred[t : t + horizon]
        wprice = prices[t : t + horizon] if prices is not None else None
        first, _ = lp_load_shift(window, capacity_kw, price=wprice)
        dispatch[t] = first[0]
    return dispatch


def peak_shaving(
    predicted_power_kw,
    capacity_kw,
    flexible_ratio=0.4,
    price=None,
    timestamps=None,
    shave_target=None,
    strategy=None,
    horizon_steps=None,
):
    """把超容负荷平移到谷段，保持对 pipeline 的 ``(dispatch, unmet)`` 接口稳定。

    ``flexible_ratio``/``timestamps`` 仅为兼容旧调用保留。默认策略是零前瞻的
    ``causal_load_shift``，保证在线评测不使用未来信息；``mpc_schedule`` 与
    ``lp_load_shift`` 仅作显式选择，后者只能离线使用。dispatch 始终不超过
    变压器容量。
    """
    pred = np.asarray(predicted_power_kw, dtype=float).reshape(-1)
    cap = float(capacity_kw)
    strategy = strategy or getattr(C, "DECISION_STRATEGY", "causal")

    if strategy == "mpc":
        dispatch = mpc_schedule(pred, cap, horizon_steps=horizon_steps, price=price)
    elif strategy == "lp_load_shift":
        dispatch, _ = lp_load_shift(pred, cap, price=price)
    else:
        dispatch, _ = causal_load_shift(pred, cap)

    if shave_target is not None:
        dispatch = np.minimum(dispatch, float(shave_target))
    dispatch = np.minimum(dispatch, cap)
    unmet = np.maximum(pred - dispatch, 0.0)
    return dispatch, unmet


PRIORITY_KEYS = {
    "fcfs": lambda s: (s["start_step"], s["idx"]),
    "low_soc_first": lambda s: (s["start_soc"], s["start_step"], s["idx"]),
    "earliest_deadline_first": lambda s: (s["end_step"], s["start_step"], s["idx"]),
}


def _session_row(session, idx):
    return {
        "idx": idx,
        "start_step": int(session["start_step"]),
        "end_step": int(session["end_step"]),
        "energy_kwh": float(session["energy_kwh"]),
        "max_power_kw": float(session["max_power_kw"]),
        "start_soc": float(session.get("start_soc", 0.0)),
        "remaining_energy_kwh": float(session["energy_kwh"]),
    }


def _empty_summary(priority):
    return {
        "served_kwh": 0.0,
        "unmet_kwh": 0.0,
        "violations": 0,
        "peak_kw": 0.0,
        "order": [],
        "priority": priority,
    }


def greedy_priority(sessions, capacity_kw, step_minutes=None, priority="low_soc_first"):
    """规则/贪心兜底：逐时段按优先级把容量分给在场车辆。

    ``sessions`` 每项需含 ``start_step``、``end_step``（左闭右开）、
    ``energy_kwh``、``max_power_kw``、``start_soc``。返回
    ``(plan, summary)``，plan 为 ``{step: {session_idx: power_kw}}``，
    summary 含服务量、未满足量、峰值、越限次数与排队顺序。
    """
    step_hours = (step_minutes or C.STEP_MINUTES) / 60.0
    rows = [_session_row(s, i) for i, s in enumerate(sessions)]
    if not rows:
        return {}, _empty_summary(priority)

    key = PRIORITY_KEYS.get(priority, PRIORITY_KEYS["low_soc_first"])
    order = [r["idx"] for r in sorted(rows, key=key)]
    t_min = min(r["start_step"] for r in rows)
    t_max = max(r["end_step"] for r in rows)

    plan = {}
    served = defaultdict(float)
    for t in range(t_min, t_max):
        active = [
            r
            for r in rows
            if r["start_step"] <= t < r["end_step"]
            and r["remaining_energy_kwh"] > 1e-9
        ]
        active.sort(key=key)
        remaining_cap = float(capacity_kw)
        for r in active:
            want = min(
                r["max_power_kw"],
                remaining_cap,
                r["remaining_energy_kwh"] / step_hours,
            )
            if want <= 1e-12:
                continue
            r["remaining_energy_kwh"] -= want * step_hours
            remaining_cap -= want
            served[r["idx"]] += want * step_hours
            plan.setdefault(t, {})[r["idx"]] = round(want, 4)

    peak_kw = max((sum(v.values()) for v in plan.values()), default=0.0)
    total_energy = sum(r["energy_kwh"] for r in rows)
    served_kwh = sum(served.values())
    summary = {
        "served_kwh": round(served_kwh, 3),
        "unmet_kwh": round(total_energy - served_kwh, 3),
        "violations": int(peak_kw > float(capacity_kw) + 1e-6),
        "peak_kw": round(peak_kw, 3),
        "order": order,
        "priority": priority,
    }
    return plan, summary


def _milp_pulp(sessions, capacity_kw, step_minutes, priority):
    import pulp

    step_hours = (step_minutes or C.STEP_MINUTES) / 60.0
    rows = [_session_row(s, i) for i, s in enumerate(sessions)]
    if not rows:
        return {}, _empty_summary(priority)

    t_min = min(r["start_step"] for r in rows)
    t_max = max(r["end_step"] for r in rows)
    steps = list(range(t_min, t_max))
    pairs = [(r["idx"], t) for r in rows for t in steps]

    prob = pulp.LpProblem("ordered_charging", pulp.LpMinimize)
    power = pulp.LpVariable.dicts("power", pairs, lowBound=0)
    on = pulp.LpVariable.dicts("on", pairs, cat="Binary")
    unmet = pulp.LpVariable.dicts("unmet", [r["idx"] for r in rows], lowBound=0)

    for r in rows:
        window = [t for t in steps if r["start_step"] <= t < r["end_step"]]
        prob += (
            pulp.lpSum(power[r["idx"], t] * step_hours for t in window)
            + unmet[r["idx"]]
            == r["energy_kwh"]
        )
    for t in steps:
        prob += (
            pulp.lpSum(
                power[r["idx"], t]
                for r in rows
                if r["start_step"] <= t < r["end_step"]
            )
            <= float(capacity_kw)
        )
    for r in rows:
        for t in steps:
            if r["start_step"] <= t < r["end_step"]:
                prob += power[r["idx"], t] <= r["max_power_kw"] * on[r["idx"], t]
            else:
                power[r["idx"], t].upBound = 0

    prob += 1000 * pulp.lpSum(list(unmet.values())) + pulp.lpSum(list(on.values()))
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    plan = {}
    served = defaultdict(float)
    for t in steps:
        step_plan = {}
        for r in rows:
            val = power[r["idx"], t].varValue or 0.0
            if val > 1e-6:
                step_plan[r["idx"]] = round(val, 4)
                served[r["idx"]] += val * step_hours
        if step_plan:
            plan[t] = step_plan

    key = PRIORITY_KEYS.get(priority, PRIORITY_KEYS["low_soc_first"])
    order = [r["idx"] for r in sorted(rows, key=key)]
    peak_kw = max((sum(v.values()) for v in plan.values()), default=0.0)
    summary = {
        "served_kwh": round(sum(served.values()), 3),
        "unmet_kwh": round(sum(r["energy_kwh"] for r in rows) - sum(served.values()), 3),
        "violations": int(peak_kw > float(capacity_kw) + 1e-6),
        "peak_kw": round(peak_kw, 3),
        "order": order,
        "priority": priority,
    }
    return plan, summary


def milp_schedule(sessions, capacity_kw, step_minutes=None, priority="low_soc_first"):
    """开关桩/整数变量调度，优先 OR-Tools/PuLP，缺失时回退规则贪心。

    返回 ``(plan, summary)``，summary 的 ``solver`` 字段记录实际求解器，
    便于路演解释该站为什么走整数求解还是规则兜底。
    """
    try:
        import pulp  # noqa: F401

        solver = "pulp"
    except ImportError:
        try:
            import ortools  # noqa: F401

            solver = "ortools"
        except ImportError:
            solver = "greedy_fallback"

    if solver == "pulp":
        plan, summary = _milp_pulp(sessions, capacity_kw, step_minutes, priority)
    else:
        plan, summary = greedy_priority(sessions, capacity_kw, step_minutes, priority)
    summary["solver"] = solver
    return plan, summary


def battery_arbitrage(
    predicted_power_kw,
    capacity_kwh,
    charge_kw,
    discharge_kw,
    soc0=0.5,
    price=None,
    efficiency=None,
    grid_capacity_kw=None,
    soc_min=None,
    soc_max=None,
):
    """储能峰谷套利：电价高于阈值放电、低于阈值充电，并顺带压峰。

    阈值用“截至当前步”的滚动均值，第 i 步只依赖 0..i 已见数据，不使用
    未来信息；始终满足 SOC 上下界与充放电效率，最后用场站容量把合成后的
    净负荷再压一次上限。接口兼容旧的
    ``(pred, capacity_kwh, charge_kw, discharge_kw, soc0)`` 调用。
    """
    pred = np.asarray(predicted_power_kw, dtype=float).reshape(-1)
    cap = float(capacity_kwh)
    eff = float(efficiency if efficiency is not None else C.BATTERY_EFFICIENCY)
    smin = float(soc_min if soc_min is not None else C.BATTERY_SOC_MIN)
    smax = float(soc_max if soc_max is not None else C.BATTERY_SOC_MAX)
    soc = cap * float(soc0)
    net = np.zeros_like(pred)

    if price is not None:
        prices = np.asarray(price, dtype=float).reshape(-1)
    else:
        prices = pred

    thresholds = np.cumsum(prices) / np.arange(1, pred.size + 1)

    for i, load in enumerate(pred):
        if prices[i] > thresholds[i]:
            out = min(float(discharge_kw), max(soc - smin * cap, 0.0))
            net[i] = -out
            soc -= out
        else:
            charge = min(float(charge_kw), max(smax * cap - soc, 0.0) / eff)
            net[i] = charge
            soc += charge * eff

    dispatch = pred + net
    if grid_capacity_kw is not None:
        dispatch = np.minimum(dispatch, float(grid_capacity_kw))
    dispatch = np.maximum(dispatch, 0.0)
    return dispatch, net, soc
