"""离线评测：预测与决策指标的唯一口径。pipeline 与 report 都从这里取数。"""
import numpy as np


def _as_array(values):
    return np.asarray(values, dtype=float).ravel()


def forecast_metrics(actual, pred):
    """预测精度：MAE、RMSE、MAPE（百分比）。"""
    actual = _as_array(actual)
    pred = _as_array(pred)
    if actual.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

    error = actual - pred
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))

    # MAPE 只对实际值明显非零的时间片计算，避免小负荷把百分比放大失真。
    nonzero = actual > 1e-6
    mape = (
        float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100.0)
        if nonzero.any()
        else 0.0
    )
    return {"mae": mae, "rmse": rmse, "mape": mape}


def decision_metrics(actual_power, dispatch, capacity_kw, step_hours):
    """离线真值口径的调度指标：削峰、越限次数、未满足电量与总电量。

    未满足电量用 ``actual - dispatch`` 的正缺口乘步长计算，是离线报告里唯一
    有真实标签的口径；在线决策侧使用世界模型推演终点的剩余电量（见
    intervention.simulate_intervention 的 ``unmet_kwh``），二者不要混用。
    """
    actual = _as_array(actual_power)
    dispatch = _as_array(dispatch)
    step_hours = float(step_hours)
    capacity_kw = float(capacity_kw)

    if actual.size == 0:
        return {
            "peak_before_kw": 0.0,
            "peak_after_kw": 0.0,
            "peak_reduction_pct": 0.0,
            "capacity_kw": capacity_kw,
            "before_violations": 0,
            "after_violations": 0,
            "unmet_kwh": 0.0,
            "curtailment_kwh": 0.0,
            "total_energy_kwh": 0.0,
        }

    peak_before_kw = float(actual.max())
    peak_after_kw = float(dispatch.max())
    peak_reduction_pct = (
        (peak_before_kw - peak_after_kw) / peak_before_kw * 100.0
        if peak_before_kw > 1e-6
        else 0.0
    )
    before_violations = int((actual > capacity_kw + 1e-6).sum())
    after_violations = int((dispatch > capacity_kw + 1e-6).sum())
    unmet_kwh = float(np.maximum(actual - dispatch, 0.0).sum() * step_hours)

    return {
        "peak_before_kw": peak_before_kw,
        "peak_after_kw": peak_after_kw,
        "peak_reduction_pct": peak_reduction_pct,
        "capacity_kw": capacity_kw,
        "before_violations": before_violations,
        "after_violations": after_violations,
        "unmet_kwh": unmet_kwh,
        # 兼容旧字段名，与 unmet_kwh 同义。
        "curtailment_kwh": unmet_kwh,
        "total_energy_kwh": float(actual.sum() * step_hours),
    }
