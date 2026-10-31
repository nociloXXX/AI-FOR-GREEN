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

SITES = ("caltech", "jpl", "office1")


def load_training(data_dir=None, sites=SITES):
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
        training[site] = clean_training_frame(frame)
    return training


def clean_training_frame(frame):
    """清洗单站训练表，只删行、不改标签值、不改变原行顺序。

    一行是一次充电连接。按 ACN-Data 的会话分析口径做物理域校验：
    ``hour`` 在 0-24、``weekday`` 为 0-6 的整数、``charger_limit_kw``
    为正、``demand_kwh`` 非负、``dwell_hours`` 为正；不合法的整行剔除。
    ``last_requested_energy_kwh`` 缺失保持 NaN，只保留正有限值，不填 0，
    也不用 ``demand_kwh`` 反推填补。
    """
    import pandas as pd

    if frame is None:
        return pd.DataFrame()
    df = frame.copy()
    required = ["hour", "weekday", "charger_limit_kw", "demand_kwh", "dwell_hours"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"训练表缺少字段: {missing}")

    hour = pd.to_numeric(df["hour"], errors="coerce")
    weekday = pd.to_numeric(df["weekday"], errors="coerce")
    charger = pd.to_numeric(df["charger_limit_kw"], errors="coerce")
    demand = pd.to_numeric(df["demand_kwh"], errors="coerce")
    dwell = pd.to_numeric(df["dwell_hours"], errors="coerce")

    valid = (
        hour.notna()
        & hour.between(0.0, 24.0)
        & weekday.notna()
        & (weekday >= 0.0)
        & (weekday <= 6.0)
        & (weekday == weekday.round())
        & charger.notna()
        & (charger > 0.0)
        & demand.notna()
        & (demand >= 0.0)
        & dwell.notna()
        & (dwell > 0.0)
    )
    df = df.loc[valid].reset_index(drop=True)

    if "last_requested_energy_kwh" in df.columns:
        requested = pd.to_numeric(df["last_requested_energy_kwh"], errors="coerce")
        keep = np.isfinite(requested) & (requested > 0.0)
        df["last_requested_energy_kwh"] = requested.where(keep)
    return df


def make_features(hour, weekday, pmax):
    """构造需求/停留回归特征。

    固定输出 7 列，训练与预测两端一致：24h/12h 周期编码、星期、是否周末、
    桩功率上限。不使用 last_requested_energy_kwh，因为训练截止前该字段
    基本全为 null，且评测时未来车辆通常也缺失。
    """
    hour = np.asarray(hour, dtype=float).reshape(-1)
    weekday = np.asarray(weekday, dtype=float).reshape(-1)
    pmax = np.asarray(pmax, dtype=float).reshape(-1)

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
        ]
    )


def cross_site_feature_names():
    """返回 make_cross_site_features 的列名，训练与预测两端保持一致。"""
    return [
        "hour_sin",
        "hour_cos",
        "weekday",
        "charger_limit_kw",
        "site_caltech",
        "site_jpl",
        "site_office1",
        "hour_x_weekday",
        "request_missing",
        "request_kwh",
    ]


def make_cross_site_features(hour, weekday, pmax, site, requested=None):
    """构造可跨站共享的预测特征底座。

    固定输出 10 列：sin/cos(hour)、weekday、charger_limit_kw、site 三站
    独热、hour×weekday 交互，以及 last_requested_energy_kwh 的“缺失指示 +
    有效值”双分支。缺失申报保持 NaN 并置 request_missing=1，既不填 0，
    也不当作目标；训练与预测调用同一函数以保证列对齐。
    """
    hour = np.asarray(hour, dtype=float).reshape(-1)
    weekday = np.asarray(weekday, dtype=float).reshape(-1)
    pmax = np.asarray(pmax, dtype=float).reshape(-1)
    site = np.asarray(site).reshape(-1)
    if requested is None:
        requested = np.full_like(hour, np.nan)
    else:
        requested = np.asarray(requested, dtype=float).reshape(-1)

    valid_request = np.isfinite(requested) & (requested > 0.0)
    request_kwh = np.where(valid_request, requested, np.nan)
    request_missing = (~valid_request).astype(float)

    hour = hour % 24.0
    site_matrix = np.column_stack(
        [(site == name).astype(float) for name in SITES]
    )
    return np.column_stack(
        [
            np.sin(hour * np.pi / 12.0),
            np.cos(hour * np.pi / 12.0),
            weekday,
            pmax,
            site_matrix,
            hour * weekday,
            request_missing,
            request_kwh,
        ]
    )


def chronological_split(training, ratio=0.8):
    """按每个站点的日期做时间序切分，返回 (train, val) 字典，禁止随机打散。"""
    import pandas as pd

    train, val = {}, {}
    for site, frame in training.items():
        frame = clean_training_frame(frame)
        if len(frame) == 0:
            train[site] = frame
            val[site] = frame.iloc[0:0].copy()
            continue
        day = pd.to_datetime(frame["day"], errors="coerce")
        days = sorted(day.dropna().unique())
        cut = days[max(1, int(len(days) * ratio)) - 1]
        train[site] = frame[day <= cut].copy()
        val[site] = frame[day > cut].copy()
    return train, val


def _laplace_hour_rates(counts, days, alpha):
    """对每组（星期或工作日/周末组）的 24 小时计数做拉普拉斯平滑。

    平滑只改变小时分布的细节，保持该组每天总到达数不变；对没有观测的组
    保持为 0，避免给 office1 这样的站点虚构周末到达。
    """
    counts = np.asarray(counts, dtype=float)
    days = np.asarray(days, dtype=float)
    alpha = max(float(alpha), 0.0)
    out = np.zeros_like(counts)
    for group in range(counts.shape[0]):
        total = float(counts[group].sum())
        if total <= 0:
            continue
        prob = (counts[group] + alpha) / (total + alpha * counts.shape[1])
        daily_rate = total / max(days[group], 1.0)
        out[group] = prob * daily_rate / 4.0
    return out


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
    """返回每个站点的需求/停留中位数标签。"""
    labels = {}
    for site, frame in training.items():
        frame = clean_training_frame(frame)
        if len(frame) == 0:
            continue
        labels[site] = (
            float(frame.demand_kwh.median()),
            float(frame.dwell_hours.median()),
        )
    return labels


def build_weekday_hour_rates(
    training, laplace_alpha=1.0, collapse_weekday_weekend=None
):
    """返回每个站点按星期x小时统计的到达率，单位：辆/15分钟步。

    先做星期×小时计数，再用拉普拉斯伪计数 ``laplace_alpha`` 平滑，避免
    稀疏小时被估计成 0。``collapse_weekday_weekend`` 为 True 时把 7 个
    星期折叠成“工作日/周末”两类；为 None 时只对没有周末观测的站点自动
    折叠（如 office1），数据充足的站点保留 7 类。
    """
    hour_rates = {}
    for site, frame in training.items():
        frame = clean_training_frame(frame)
        if len(frame) == 0:
            hour_rates[site] = np.zeros((7, 24), dtype=float)
            continue

        _, weekday_days = _day_and_weekday_counts(frame)
        counts = np.zeros((7, 24), dtype=float)
        hours = (np.asarray(frame.hour, dtype=float) % 24.0).astype(int)
        weekdays = np.asarray(frame.weekday, dtype=int) % 7
        np.add.at(counts, (weekdays, hours), 1.0)

        if collapse_weekday_weekend is None:
            collapse = counts[5:7].sum() == 0
        else:
            collapse = bool(collapse_weekday_weekend)

        if collapse:
            group_counts = np.stack(
                [counts[:5].sum(axis=0), counts[5:7].sum(axis=0)]
            )
            group_days = np.array(
                [weekday_days[:5].sum(), weekday_days[5:7].sum()]
            )
            smoothed = _laplace_hour_rates(group_counts, group_days, laplace_alpha)
            out = np.zeros((7, 24), dtype=float)
            out[:5] = smoothed[0]
            out[5:7] = smoothed[1]
        else:
            out = _laplace_hour_rates(counts, weekday_days, laplace_alpha)
        hour_rates[site] = out
    return hour_rates
