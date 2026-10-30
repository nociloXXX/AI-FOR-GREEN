"""读取、清洗与重构原始数据。真实字段映射集中放在 config.py 与 _apply_aliases。"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def _apply_aliases(df, aliases):
    """把官方列名重命名为规范列名，只处理真实存在的列。"""
    if not aliases:
        return df
    df = df.copy()
    rename = {
        official: canonical
        for official, canonical in aliases.items()
        if official in df.columns and canonical not in df.columns
    }
    return df.rename(columns=rename) if rename else df


def load_raw_charging(path=None):
    """读取充电原始数据，映射字段并统一为规范列。"""
    path = Path(path) if path else C.RAW_CHARGING_FILE
    df = pd.read_csv(path)
    df = _apply_aliases(df, C.RAW_COLUMN_ALIASES)

    missing = [col for col in C.REQUIRED_RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"充电数据缺少字段: {missing}")

    df = df.copy()
    df[C.TIME_START] = pd.to_datetime(df[C.TIME_START], errors="coerce")
    df[C.TIME_END] = pd.to_datetime(df[C.TIME_END], errors="coerce")
    for col in (
        C.START_SOC_COL,
        C.END_SOC_COL,
        C.ENERGY_COL,
        C.MAX_POWER_COL,
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values([C.STATION_COL, C.TIME_START]).reset_index(drop=True)


def _anomaly_mask(df, method=None, columns=None, fit_ratio=None):
    """按 IQR、3σ 或 Isolation Forest 标记异常行。

    阈值只在前 ``fit_ratio`` 的时间段内估计，再应用到整段数据，保证清洗
    规则在线部署时不会借用未来观测。
    """
    method = method or C.ANOMALY_METHOD
    columns = list(columns or [C.UTILIZATION_COL])
    fit_ratio = fit_ratio if fit_ratio is not None else C.TRAIN_RATIO
    mask = pd.Series(False, index=df.index)
    if method in (None, "none"):
        return mask

    for station, group in df.groupby(C.STATION_COL, sort=False):
        if len(group) < 2:
            continue
        fit = group.iloc[: max(2, int(len(group) * fit_ratio))]
        local = pd.Series(False, index=group.index)
        for col in columns:
            if col not in group.columns:
                continue
            fit_values = fit[col].dropna()
            if fit_values.empty:
                continue
            x = group[col]
            if method == "iqr":
                q1, q3 = fit_values.quantile([0.25, 0.75])
                iqr = q3 - q1
                lo, hi = q1 - C.IQR_K * iqr, q3 + C.IQR_K * iqr
                local |= (x < lo) | (x > hi)
            elif method == "sigma":
                mu = float(fit_values.mean())
                sd = float(fit_values.std())
                if sd > 0:
                    local |= (x - mu).abs() > C.SIGMA * sd
            elif method == "isolation_forest":
                try:
                    from sklearn.ensemble import IsolationForest
                except ImportError:
                    mu = float(fit_values.mean())
                    sd = float(fit_values.std())
                    if sd > 0:
                        local |= (x - mu).abs() > C.SIGMA * sd
                    continue
                model = IsolationForest(
                    contamination=C.ISOLATION_CONTAMINATION, random_state=0
                )
                model.fit(fit[[col]].dropna())
                pred_frame = group[[col]].fillna(float(fit_values.median()))
                local |= model.predict(pred_frame) == -1
        mask.loc[group.index] = local
    return mask


def clean_charging(df):
    """规则清洗保底 + 缺失填补 + 异常检测，输出重构后的会话表。"""
    df = df.copy().sort_values([C.STATION_COL, C.TIME_START]).reset_index(drop=True)

    # 规则清洗：关键键、时间方向、物理约束先保底。
    df = df.dropna(subset=[C.STATION_COL, C.PILE_COL, C.TIME_START, C.TIME_END])
    df = df[df[C.TIME_END] > df[C.TIME_START]]
    df = df[
        df[C.START_SOC_COL].between(C.SOC_MIN, C.SOC_MAX, inclusive="both")
        | df[C.START_SOC_COL].isna()
    ]
    df = df[
        df[C.END_SOC_COL].between(C.SOC_MIN, C.SOC_MAX, inclusive="both")
        | df[C.END_SOC_COL].isna()
    ]

    # 缺失填补：SOC 与最大功率按桩、按时间只做前向填补，不借用未来值。
    for col in (C.START_SOC_COL, C.END_SOC_COL, C.MAX_POWER_COL):
        if col in df.columns:
            df[col] = df.groupby(C.PILE_COL, sort=False)[col].ffill()

    duration_hours = (
        (df[C.TIME_END] - df[C.TIME_START]).dt.total_seconds() / 3600
    )
    duration_hours = duration_hours.clip(lower=1e-6)

    # 缺电量用功率乘时长反推，再做物理约束过滤。
    if C.BACKFILL_ENERGY:
        missing_energy = df[C.ENERGY_COL].isna()
        if missing_energy.any():
            backfill = (
                df.loc[missing_energy, C.MAX_POWER_COL]
                * duration_hours[missing_energy]
                * C.ENERGY_BACKFILL_FACTOR
            )
            df.loc[missing_energy, C.ENERGY_COL] = backfill

    df = df[df[C.MAX_POWER_COL] > 0]
    df = df[df[C.ENERGY_COL] >= 0]
    df = df.dropna(subset=[C.ENERGY_COL, C.MAX_POWER_COL])

    # 重构会话：时长、平均功率与利用率，利用率超过物理上限的会话剔除。
    df = df.copy()
    df["duration_min"] = (
        (df[C.TIME_END] - df[C.TIME_START]).dt.total_seconds() / 60
    ).clip(lower=1e-6)
    df["duration_hours"] = df["duration_min"] / 60
    df["avg_power_kw"] = df[C.ENERGY_COL] / df["duration_hours"]
    df[C.UTILIZATION_COL] = (
        df[C.ENERGY_COL] / (df[C.MAX_POWER_COL] * df["duration_hours"])
    )
    df = df[df[C.UTILIZATION_COL] <= C.UTILIZATION_MAX]

    # 异常检测放在规则清洗之后，避免大电量合法会话被全局分位数误删。
    anomaly = _anomaly_mask(df)
    if anomaly.any():
        df = df[~anomaly]

    return (
        df.sort_values([C.STATION_COL, C.TIME_START]).reset_index(drop=True)
    )


def _regular_grid(df, value_cols, freq_minutes=None, time_col=None):
    """按固定步长重采样并做前向填补。

    只用过去值填补，避免外源特征把未来信息带进模型；前置缺失保持为空。
    """
    freq_minutes = freq_minutes or C.STEP_MINUTES
    time_col = time_col or C.TIME_COL
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    df = df.drop_duplicates(subset=[time_col], keep="last")
    if df.empty:
        return df

    freq = f"{freq_minutes}min"
    full_index = pd.date_range(df[time_col].min().floor(freq), df[time_col].max().ceil(freq), freq=freq)
    frame = df.set_index(time_col).reindex(full_index).rename_axis(time_col)
    for col in value_cols:
        if col not in frame.columns:
            continue
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame[col] = frame[col].ffill()
    return frame.reset_index()


def load_price(path=None):
    path = Path(path) if path else C.PRICE_FILE
    df = pd.read_csv(path)
    df = _apply_aliases(df, C.PRICE_COLUMN_ALIASES)
    return _regular_grid(df, [C.PRICE_VALUE_COL])


def load_weather(path=None):
    path = Path(path) if path else C.WEATHER_FILE
    df = pd.read_csv(path)
    df = _apply_aliases(df, C.WEATHER_COLUMN_ALIASES)
    return _regular_grid(df, [C.TEMPERATURE_COL])
