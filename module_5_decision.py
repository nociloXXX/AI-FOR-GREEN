"""模块5：充电决策。

act 按最早离开优先（EDF）在可用容量内为当前连接车辆分配请求功率，
只使用 capacity_kw 与 forecast_capacity 的实际行动限额。
"""
from __future__ import annotations


def act_edf(observation, predict_fn):
    request = []
    for vehicle in observation.vehicles:
        if vehicle.charging_complete:
            continue
        pred = predict_fn(vehicle.features)
        remaining = max(pred.demand_kwh - vehicle.delivered_kwh, 0.0)
        if remaining < 0.05:
            remaining = vehicle.features.charger_limit_kw * observation.dt_hours
        available = max(
            pred.dwell_hours / observation.dt_hours - vehicle.elapsed_steps, 1.0
        )
        request.append(
            (
                available,
                vehicle.features.token,
                min(
                    vehicle.features.charger_limit_kw,
                    remaining / observation.dt_hours,
                ),
            )
        )

    power = {}
    cap = max(observation.capacity_kw - observation.background_kw, 0.0)
    for _deadline, token, wanted in sorted(request):
        granted = min(wanted, cap)
        power[token] = granted
        cap -= granted
    return power
