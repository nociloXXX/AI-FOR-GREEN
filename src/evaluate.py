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
    actual = np.asarray(actual_power, dtype=float)
    dispatch = np.asarray(dispatch, dtype=float)
    return {
        "peak_kw": float(dispatch.max()),
        "capacity_kw": float(capacity_kw),
        "curtailment_kwh": float(np.maximum(actual - dispatch, 0).sum() * step_hours),
        "total_energy_kwh": float(actual.sum() * step_hours),
    }
