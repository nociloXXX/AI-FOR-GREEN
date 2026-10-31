"""模块1：训练表加载、数据处理与特征构造。

训练表是站点名到 pandas.DataFrame 的字典，列为：
    hour, weekday, charger_limit_kw, demand_kwh, dwell_hours, day,
    last_requested_energy_kwh

本模块只使用公开历史训练数据。到达率与特征只依据公开字段估计，不读取
隐藏场景、未来到达或真实标签。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_training(data_dir=None, sites=("caltech", "jpl", "office1")):
    """本地读取三站训练 JSON，返回 site -> DataFrame 的字典。

    评测沙箱不调用本函数（training 已由评测器注入 build_model），这里仅
    供本地验证与自评使用。
    """
    import json

    import pandas as pd

    data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent / "data"
    training = {}
    for site in sites:
        path = data_dir / f"training_{site}.json"
        if not path.exists():
            continue
        rows = json.load(open(path, encoding="utf-8"))
        frame = pd.DataFrame(rows)
        frame["day"] = pd.to_datetime(frame["day"], errors="coerce")
        for col in (
            "hour",
            "weekday",
            "charger_limit_kw",
            "demand_kwh",
            "dwell_hours",
        ):
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        # 固定 6.6 kW 桩是公开参数，缺失时按该值补齐。
        if "charger_limit_kw" in frame.columns:
            frame["charger_limit_kw"] = frame["charger_limit_kw"].fillna(6.6)
        frame = frame.dropna(
            subset=["day", "hour", "weekday", "demand_kwh", "dwell_hours"]
        ).reset_index(drop=True)
        training[site] = frame.sort_values(["day", "hour"]).reset_index(drop=True)
    return training


def make_features(hour, weekday, pmax, requested=None):
    """构造需求/停留回归特征。

    固定输出 9 列，训练与预测两端一致：24h/12h 周期编码、星期、是否周末、
    桩功率上限、是否有最后申报电量、最后申报电量本身。``requested`` 为
    last_requested_energy_kwh，缺失时按 0 编码并配 has_requested=0，不把
    缺失值当成 0 的真实含义。即使调用方只传 3 个参数，列数也保持不变。
    """
    hour = np.asarray(hour, dtype=float).reshape(-1)
    weekday = np.asarray(weekday, dtype=float).reshape(-1)
    pmax = np.asarray(pmax, dtype=float).reshape(-1)

    if requested is None:
        requested = np.full_like(hour, np.nan)
    else:
        requested = np.asarray(requested, dtype=float).reshape(-1)

    requested_clean = np.where(
        np.isfinite(requested) & (requested > 0.0), requested, 0.0
    )
    has_requested = np.where(
        np.isfinite(requested) & (requested > 0.0), 1.0, 0.0
    )
    hour = hour % 24.0
    return np.column_stack(
        [
            np.sin(hour * np.pi / 12.0),
            np.cos(hour * np.pi / 12.0),
            np.sin(hour * np.pi / 6.0),
            np.cos(hour * np.pi / 6.0),
            weekday,
            (weekday >= 5.0).astype(float),
            pmax,
            has_requested,
            requested_clean,
        ]
    )


def _smooth_hours(counts):
    """对 24 小时计数做环形 3 点平滑，保持一天总到达数不变。"""
    counts = np.asarray(counts, dtype=float)
    if counts.shape[-1] < 3:
        return counts.copy()
    left = np.roll(counts, 1, axis=-1)
    right = np.roll(counts, -1, axis=-1)
    return 0.25 * left + 0.5 * counts + 0.25 * right


def _day_and_weekday_counts(frame):
    """按日期范围统计每个星期出现的天数，返回 (总天数, 7 维星期计数)。"""
    import pandas as pd

    days = pd.to_datetime(frame["day"], errors="coerce").dropna()
    if days.empty:
        return 1.0, np.zeros(7, dtype=float)
    span = pd.date_range(days.min().normalize(), days.max().normalize(), freq="D")
    weekday_counts = np.bincount(span.dayofweek, minlength=7).astype(float)
    return max(float(len(span)), 1.0), weekday_counts


def summarize_site(training):
    """返回每个站点的需求/停留中位数标签与分小时到达率。"""
    labels = {}
    rates = {}
    for site, frame in training.items():
        if frame is None or len(frame) == 0:
            continue
        labels[site] = (
            float(frame.demand_kwh.median()),
            float(frame.dwell_hours.median()),
        )
        total_days, _ = _day_and_weekday_counts(frame)
        hours = np.asarray(frame.hour, dtype=float) % 24.0
        counts = np.histogram(hours, bins=24, range=(0.0, 24.0))[0].astype(float)
        rates[site] = _smooth_hours(counts) / max(total_days, 1.0) / 4.0
    return labels, rates


def build_weekday_hour_rates(training):
    """返回每个站点按星期x小时统计的到达率，单位：辆/15分钟步。"""
    hour_rates = {}
    for site, frame in training.items():
        if frame is None or len(frame) == 0:
            hour_rates[site] = np.zeros((7, 24), dtype=float)
            continue

        _, weekday_days = _day_and_weekday_counts(frame)
        counts = np.zeros((7, 24), dtype=float)
        hours = (np.asarray(frame.hour, dtype=float) % 24.0).astype(int)
        weekdays = np.asarray(frame.weekday, dtype=int) % 7
        np.add.at(counts, (weekdays, hours), 1.0)

        smoothed = _smooth_hours(counts)
        hour_rates[site] = smoothed / np.maximum(weekday_days[:, None], 1.0) / 4.0
    return hour_rates
