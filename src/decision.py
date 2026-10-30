"""有序充电与调度决策。

这里给的是两个可运行的简单策略，用来先闭环。真实上分点在这里：
把峰值负荷平移到谷段、控制变压器不越限、降低电费与碳排。
"""
import numpy as np


def peak_shaving(predicted_power_kw, capacity_kw, flexible_ratio=0.4):
    """把超容部分尽量平摊到有裕量的时段，模拟时间平移。

    真正的调度应基于 state_model 的转移结果做约束优化或 MPC。
    """
    pred = np.asarray(predicted_power_kw, dtype=float)
    base = np.minimum(pred, capacity_kw)
    overflow = np.maximum(pred - capacity_kw, 0)
    headroom = np.maximum(capacity_kw - base, 0)

    total_overflow = float(overflow.sum())
    if headroom.sum() > 0 and total_overflow > 0:
        shifted = headroom * (total_overflow / headroom.sum())
    else:
        shifted = 0.0

    dispatch = np.minimum(base + shifted, capacity_kw)
    unmet = np.maximum(pred - dispatch, 0)
    return dispatch, unmet


def battery_arbitrage(
    predicted_power_kw, capacity_kwh, charge_kw, discharge_kw, soc0=0.5
):
    """峰谷套利：负荷高于均值时放电，低于均值时充电。"""
    pred = np.asarray(predicted_power_kw, dtype=float)
    mean = float(pred.mean())
    soc = capacity_kwh * soc0
    net = np.zeros_like(pred)

    for i, load in enumerate(pred):
        if load > mean:
            out = min(discharge_kw, soc)
            net[i] = -out
            soc -= out
        else:
            charge = min(charge_kw, capacity_kwh - soc)
            net[i] = charge
            soc += charge

    dispatch = np.maximum(pred + net, 0)
    return dispatch, net, soc
