"""模块1：数据处理与特征构造。

负责把三站训练表整理成标签、到达率和模型特征。评测时只使用公开训练数据，
不读取隐藏场景、未来到达或真值标签。
"""
from __future__ import annotations

import numpy as np


def make_features(hour, weekday, pmax):
    """构造需求/停留回归特征：小时周期编码 + 星期 + 充电桩功率上限。"""
    hour = np.asarray(hour, dtype=float)
    weekday = np.asarray(weekday, dtype=float)
    pmax = np.asarray(pmax, dtype=float)
    return np.column_stack(
        [
            np.sin(hour * np.pi / 12.0),
            np.cos(hour * np.pi / 12.0),
            weekday,
            pmax,
        ]
    )


def summarize_site(training):
    """返回每个站点的需求/停留中位数标签与分小时到达率。"""
    labels = {}
    rates = {}
    for site, frame in training.items():
        labels[site] = (
            float(frame.demand_kwh.median()),
            float(frame.dwell_hours.median()),
        )
        span = max((frame.day.max() - frame.day.min()).days + 1, 1)
        rates[site] = (
            np.histogram(frame.hour, bins=np.arange(25))[0] / span / 4.0
        )
    return labels, rates


def build_weekday_hour_rates(training):
    """返回每个站点按星期x小时统计的到达率，单位：辆/15分钟步。"""
    import pandas as pd

    hour_rates = {}
    for site, frame in training.items():
        counts = np.zeros((7, 24))
        days = np.zeros(7)
        for day in pd.date_range(frame.day.min(), frame.day.max(), freq="D"):
            days[day.dayofweek] += 1
        for weekday, hour in zip(frame.weekday, frame.hour):
            counts[int(weekday), int(hour) % 24] += 1
        hour_rates[site] = counts / np.maximum(days[:, None], 1) / 4.0
    return hour_rates
