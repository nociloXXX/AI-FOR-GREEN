"""KM 停留时长离线诊断（只标定，不接在线）。

用 numpy 手写 Kaplan-Meier 估计每个站点 dwell_hours 的生存曲线，并提取
q50/q75/q90 停留分位。训练标签都是完整观测（无删失），因此 KM 生存曲线退化为
经验生存曲线；保留 KM 乘积形式以对齐离线标定口径。本脚本只用于诊断，
不被 submission.py import。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


SITES = ("caltech", "jpl", "office1")


def kaplan_meier(times):
    """返回 (sorted_unique_times, survival)，survival 在 time 0 处为 1。"""
    t = np.sort(np.asarray(times, dtype=float))
    length = len(t)
    n = length
    times = [0.0]
    surv = [1.0]
    i = 0
    while i < length:
        j = i
        while j < length and t[j] == t[i]:
            j += 1
        d = j - i
        surv.append(surv[-1] * (1.0 - d / n))
        times.append(float(t[i]))
        n -= d
        i = j
    return np.asarray(times), np.asarray(surv)


def km_quantile(times, q):
    t, s = kaplan_meier(times)
    target = 1.0 - q
    below = s <= target
    idx = int(np.argmax(below)) if below.any() else len(t) - 1
    return float(t[idx])


def main():
    print("KM dwell 分位（小时）")
    for site in SITES:
        rows = json.load(open(f"data/training_{site}.json", encoding="utf-8"))
        dwell = np.asarray([r["dwell_hours"] for r in rows], dtype=float)
        qs = {q: km_quantile(dwell, q) for q in (0.5, 0.75, 0.9)}
        np_qs = {q: float(np.quantile(dwell, q)) for q in (0.5, 0.75, 0.9)}
        print(
            f"{site:8s} km={ {q: round(v, 3) for q, v in qs.items()} }  "
            f"np={ {q: round(v, 3) for q, v in np_qs.items()} }"
        )


if __name__ == "__main__":
    main()
