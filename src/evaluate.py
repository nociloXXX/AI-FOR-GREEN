"""离线评测。开题后按官方公开的指标口径补全。"""
import numpy as np


def forecast_metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "mae": float(np.mean(np.abs(actual - pred))),
        "rmse": float(np.sqrt(np.mean((actual - pred) ** 2))),
    }


def decision_metrics(actual_power, dispatch, capacity_kw, step_hours):
    """离线真值口径的调度指标。

    这里使用 ``actual - dispatch`` 的正缺口乘步长作为“未满足电量”，是离线
    报告里唯一有真实标签的口径；在线决策侧则使用世界模型推演终点的剩余电量
    （见 intervention.simulate_intervention 的 ``unmet_kwh``）。
    """
    actual = np.asarray(actual_power, dtype=float)
    dispatch = np.asarray(dispatch, dtype=float)
    unmet_kwh = float(np.maximum(actual - dispatch, 0).sum() * step_hours)
    return {
        "peak_kw": float(dispatch.max()),
        "capacity_kw": float(capacity_kw),
        "unmet_kwh": round(unmet_kwh, 6),
        # 兼容旧字段名，与 unmet_kwh 同义。
        "curtailment_kwh": round(unmet_kwh, 6),
        "total_energy_kwh": float(actual.sum() * step_hours),
    }
