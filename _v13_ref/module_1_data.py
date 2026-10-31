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

SITE_PROFILE_FEATURE_NAMES = [
    "meta_n_curves",
    "meta_n_samples",
    "meta_mean_samples_per_curve",
    "meta_mean_interval_s",
    "missing_power_rate",
    "missing_voltage_rate",
    "missing_current_rate",
    "dwell_median_h",
    "dwell_q25_h",
    "dwell_q75_h",
    "dwell_mean_h",
    "charge_median_h",
    "energy_median_kwh",
    "energy_q25_kwh",
    "energy_q75_kwh",
    "energy_mean_kwh",
] + [f"shape_bin_{i}" for i in range(10)] + [
    "taper_ratio",
    "taper_slope",
] + [
    "elapsed_current_00_30",
    "elapsed_current_30_60",
    "elapsed_current_1_2h",
    "elapsed_current_2_3h",
    "elapsed_current_3_4h",
    "elapsed_current_4_6h",
    "elapsed_current_6_8h",
    "elapsed_current_8h_plus",
]


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


def site_profile_feature_names():
    """返回 build_site_profile_features 的特征名，长度与向量一致。"""
    return list(SITE_PROFILE_FEATURE_NAMES)


def _finite_array(values):
    return np.asarray(
        [v if v is not None and np.isfinite(float(v)) else np.nan for v in values],
        dtype=float,
    )


def _safe_nanmean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else np.nan


def _safe_nanpercentile(values, p):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, p)) if arr.size else np.nan


def _curve_profile(samples, start_ns, dwell_h, progress_bins, elapsed_edges):
    """从一条曲线的样本提取特征分量，返回与特征名同长的单曲线向量。"""
    import pandas as pd

    timestamps = (
        pd.to_datetime([s["timestamp_utc"] for s in samples])
        .tz_localize(None)
        .astype("datetime64[ns]")
        .astype(np.int64)
        .to_numpy()
    )
    current = _finite_array([s.get("current_a") for s in samples])
    power = _finite_array([s.get("power_kw_observed") for s in samples])
    voltage = _finite_array([s.get("voltage_v") for s in samples])

    n = len(samples)
    elapsed_h = (timestamps - start_ns) / 3.6e12
    elapsed_h = np.clip(elapsed_h, 0.0, None)
    if dwell_h <= 0:
        dwell_h = float(elapsed_h[-1]) if n else np.nan
    progress = elapsed_h / max(dwell_h, 1e-9)
    progress = np.clip(progress, 0.0, 1.0)

    # 采样间隔（秒）：逐曲线中位数。
    if n >= 2:
        intervals = np.diff(timestamps).astype(float) / 1e9
        mean_interval = float(np.nanmedian(intervals))
    else:
        mean_interval = np.nan

    curr_finite = np.isfinite(current)
    shape = np.full(progress_bins, np.nan)
    for b in range(progress_bins):
        lo, hi = b / progress_bins, (b + 1) / progress_bins
        mask = curr_finite & (progress >= lo) & (progress < hi)
        if mask.any():
            shape[b] = float(np.mean(current[mask]))
    peak = float(np.nanmax(shape))
    if np.isfinite(peak) and peak > 1e-9:
        shape = shape / peak

    early = current[curr_finite & (progress < 0.2)]
    late = current[curr_finite & (progress > 0.8)]
    early_mean = float(np.mean(early)) if early.size else np.nan
    late_mean = float(np.mean(late)) if late.size else np.nan
    taper_ratio = late_mean / early_mean if early_mean and early_mean > 0 else np.nan
    taper_slope = late_mean - early_mean if np.isfinite(late_mean) and np.isfinite(early_mean) else np.nan

    charge_frac = float(np.mean(curr_finite & (current > 1.0))) if n else np.nan
    charge_h = charge_frac * dwell_h

    # 能量：用功率对时间梯形积分，JPL 无功率时为 NaN。
    if n >= 2 and np.isfinite(power).sum() >= 2:
        p0, p1 = power[:-1], power[1:]
        dt = np.diff(elapsed_h)
        seg = np.isfinite(p0) & np.isfinite(p1) & np.isfinite(dt) & (dt >= 0)
        energy = float(np.sum((p0[seg] + p1[seg]) / 2.0 * dt[seg]))
    else:
        energy = np.nan

    elapsed = np.full(len(elapsed_edges) - 1, np.nan)
    for i in range(len(elapsed_edges) - 1):
        lo, hi = elapsed_edges[i], elapsed_edges[i + 1]
        mask = curr_finite & (elapsed_h >= lo) & (elapsed_h < hi)
        if mask.any():
            elapsed[i] = float(np.mean(current[mask]))

    return np.concatenate(
        [
            [n, mean_interval, charge_h, energy],
            shape,
            [taper_ratio, taper_slope],
            elapsed,
            [float(np.isnan(current).mean()), float(np.isnan(power).mean()), float(np.isnan(voltage).mean())],
        ]
    )


def build_site_profile_features(timeseries_dir=None, progress_bins=10):
    """从历史充电曲线提取站点级统计特征，返回 {site: np.ndarray}。

    特征只依赖 site，在线预测时可直接查表；只使用历史曲线的观测，不读取
    任何会话的未来。JPL 缺电压/功率，能量特征保持 NaN，曲线形状用电流按
    充电进度归一化，电流只作形状/活跃度代理，不当裁判功率。向量长度与
    site_profile_feature_names() 一致。
    """
    import gzip
    import json
    from pathlib import Path

    import pandas as pd

    timeseries_dir = (
        Path(timeseries_dir) if timeseries_dir else Path(__file__).resolve().parent / "data"
    )
    elapsed_edges = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, np.inf]
    profiles = {}
    for site in SITES:
        path = timeseries_dir / f"timeseries_{site}.jsonl.gz"
        if not path.exists():
            profiles[site] = np.full(len(SITE_PROFILE_FEATURE_NAMES), np.nan)
            continue

        dwells, charges, energies, shapes, tapers_ratio, tapers_slope = [], [], [], [], [], []
        interval_means, sample_counts = [], []
        miss_current, miss_power, miss_voltage = [], [], []
        elapsed_sums = np.zeros(len(elapsed_edges) - 1)
        elapsed_counts = np.zeros(len(elapsed_edges) - 1)

        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                curve = json.loads(line)
                samples = curve.get("samples") or []
                if not samples:
                    continue
                start = pd.to_datetime(curve["start_utc"], utc=True)
                end = pd.to_datetime(curve["end_utc"], utc=True)
                dwell_h = max((end - start).total_seconds() / 3600.0, 0.0)
                start_ns = pd.to_datetime(samples[0]["timestamp_utc"], utc=True).value
                per = _curve_profile(
                    samples, start_ns, dwell_h, progress_bins, elapsed_edges
                )
                n = int(per[0])
                sample_counts.append(n)
                dwells.append(dwell_h)
                interval_means.append(per[1])
                charges.append(per[2])
                energies.append(per[3])
                shapes.append(per[4 : 4 + progress_bins])
                tapers_ratio.append(per[4 + progress_bins])
                tapers_slope.append(per[4 + progress_bins + 1])
                elapsed_vals = per[
                    4 + progress_bins + 2 : 4 + progress_bins + 2 + len(elapsed_edges) - 1
                ]
                elapsed_sums += np.nan_to_num(elapsed_vals)
                elapsed_counts += np.isfinite(elapsed_vals)
                miss_current.append(per[-3])
                miss_power.append(per[-2])
                miss_voltage.append(per[-1])

        n_curves = len(sample_counts)
        n_samples = int(sum(sample_counts))
        mean_samples = n_samples / n_curves if n_curves else np.nan
        mean_interval = _safe_nanmean(interval_means)
        miss_cur = float(np.mean(_finite_array(miss_current))) if miss_current else np.nan
        miss_pow = float(np.mean(_finite_array(miss_power))) if miss_power else np.nan
        miss_vol = float(np.mean(_finite_array(miss_voltage))) if miss_voltage else np.nan

        shape_mean = np.full(progress_bins, np.nan)
        if shapes:
            shape_stack = np.vstack(shapes)
            for b in range(progress_bins):
                shape_mean[b] = _safe_nanmean(shape_stack[:, b])
        elapsed_mean = np.full(len(elapsed_edges) - 1, np.nan)
        mask = elapsed_counts > 0
        elapsed_mean[mask] = elapsed_sums[mask] / elapsed_counts[mask]

        profiles[site] = np.concatenate(
            [
                [n_curves, n_samples, mean_samples, mean_interval],
                [miss_pow, miss_vol, miss_cur],
                [
                    _safe_nanpercentile(dwells, 50),
                    _safe_nanpercentile(dwells, 25),
                    _safe_nanpercentile(dwells, 75),
                    _safe_nanmean(dwells),
                    _safe_nanpercentile(charges, 50),
                ],
                [
                    _safe_nanpercentile(energies, 50),
                    _safe_nanpercentile(energies, 25),
                    _safe_nanpercentile(energies, 75),
                    _safe_nanmean(energies),
                ],
                shape_mean,
                [
                    _safe_nanmean(tapers_ratio),
                    _safe_nanmean(tapers_slope),
                ],
                elapsed_mean,
            ]
        )
    return profiles


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
