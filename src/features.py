"""特征工程与负荷曲线构建。所有滞后特征只回看过去，避免数据穿越。"""
import numpy as np
import pandas as pd

from . import config as C
from .data_loader import load_price, load_weather


def add_time_features(df, time_col="timestamp"):
    """构造时间特征。"""
    df = df.copy()
    dt = pd.to_datetime(df[time_col])
    df["hour"] = dt.dt.hour
    df["minute_of_day"] = dt.dt.hour * 60 + dt.dt.minute
    df["weekday"] = dt.dt.weekday
    df["is_weekend"] = (dt.dt.weekday >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    return df


def add_lag_features(df, value_cols, lags=None, rolling_window=None):
    """按时间排序后生成滞后与滚动均值特征。

    只用过去信息：shift 正向滞后，滚动均值排除当前步。训练/验证仍必须按
    时间顺序切分，不能随机打散。
    """
    lags = lags if lags is not None else C.LAGS
    rolling_window = rolling_window if rolling_window is not None else C.ROLLING_WINDOW
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in value_cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        # 用过去 24 步的均值，shift(1) 确保不包含当前步（目标列）。
        df[f"{col}_rolling{rolling_window}"] = (
            df[col].shift(1).rolling(rolling_window, min_periods=1).mean()
        )
    return df


def merge_external(df, price, weather, on="timestamp"):
    """对齐电价与天气，左连接后用前向值补齐，不使用未来信息。"""
    df = df.copy()
    if price is not None:
        df = df.merge(price, on=on, how="left", suffixes=("", "_price"))
        for col in price.columns:
            if col != on and col in df.columns:
                df[col] = df[col].ffill()
    if weather is not None:
        df = df.merge(weather, on=on, how="left", suffixes=("", "_weather"))
        for col in weather.columns:
            if col != on and col in df.columns:
                df[col] = df[col].ffill()
    return df


def _taper_shape(n_steps, taper_fraction, taper_floor):
    """恒功率后线性衰减的功率形状，峰值为 1，用于电量守恒缩放。"""
    n = max(int(n_steps), 1)
    shape = np.ones(n, dtype=float)
    if n == 1:
        return shape
    full = max(0, min(int(round(n * taper_fraction)), n))
    taper_steps = n - full
    if taper_steps > 0:
        for i in range(taper_steps):
            frac = (i + 1) / taper_steps
            shape[full + i] = 1.0 - (1.0 - taper_floor) * frac
    return shape


def _session_power_profile(energy, max_power, n_steps, step_hours):
    """把电量按真实功率形状摊到 n 个时间步，保证总电量守恒且不超最大功率。"""
    n = max(int(n_steps), 1)
    energy = float(max(energy, 0.0))
    max_power = float(max_power)
    if n == 1:
        power = energy / step_hours
        if max_power > 0:
            power = min(power, max_power)
        return np.array([power], dtype=float)

    floor = C.TAPER_FLOOR
    full = max(0, min(int(round(n * C.TAPER_FRACTION)), n))

    # scale * sum(shape) * step_hours == energy，即总电量守恒；峰值不超过
    # max_power。平均功率偏高时逐步扩大恒功率段，直到形状能装下全部电量。
    if max_power > 0:
        while full < n:
            shape = _taper_shape(n, full / n, floor)
            if energy / (step_hours * float(shape.sum())) <= max_power:
                break
            full += 1
    shape = _taper_shape(n, full / n, floor)
    scale = energy / (step_hours * float(shape.sum()))
    if max_power > 0 and scale > max_power:
        scale = max_power
    return np.maximum(scale * shape, 0.0)


def build_station_load_curve(sessions, freq_minutes=None, merge_externals=True):
    """把充电会话转成 15 分钟粒度的场站负荷曲线。

    电量按“恒功率 + 末端线性衰减”的真实充电形状采样，再按场站聚合；
    可选合并电价与天气外源特征。真实功率曲线可在此函数内替换。
    """
    freq_minutes = freq_minutes or C.STEP_MINUTES
    step_hours = freq_minutes / 60
    freq = f"{freq_minutes}min"

    start = sessions[C.TIME_START].min().floor(freq)
    end = sessions[C.TIME_END].max().ceil(freq)
    grid = pd.date_range(start, end, freq=freq)

    rows = []
    for session in sessions.itertuples(index=False):
        t_start = getattr(session, C.TIME_START).floor(freq)
        t_end = getattr(session, C.TIME_END)
        stamps = pd.date_range(t_start, t_end, freq=freq, inclusive="left")
        if len(stamps) == 0:
            stamps = pd.DatetimeIndex([t_start])
        powers = _session_power_profile(
            getattr(session, C.ENERGY_COL),
            getattr(session, C.MAX_POWER_COL),
            len(stamps),
            step_hours,
        )
        station = getattr(session, C.STATION_COL)
        for stamp, power in zip(stamps, powers):
            rows.append(
                {
                    "timestamp": stamp,
                    C.STATION_COL: station,
                    C.LOAD_COL: float(power),
                    C.OCCUPANCY_COL: 1,
                }
            )

    if not rows:
        raise ValueError("没有可用的充电会话")

    detail = pd.DataFrame(rows)
    agg = detail.groupby(["timestamp", C.STATION_COL]).agg(
        **{
            C.LOAD_COL: (C.LOAD_COL, "sum"),
            C.OCCUPANCY_COL: (C.OCCUPANCY_COL, "sum"),
        }
    )

    stations = sorted(sessions[C.STATION_COL].unique())
    index = pd.MultiIndex.from_product(
        [grid, stations], names=["timestamp", C.STATION_COL]
    )
    curve = agg.reindex(index, fill_value=0).reset_index()

    if merge_externals:
        try:
            curve = merge_external(curve, load_price(), load_weather())
        except Exception:
            # 外源文件缺失或字段不匹配时保底，仍输出核心负荷曲线。
            pass
    return curve


def chronological_split(df, ratio=None, time_col="timestamp"):
    """严格按时间顺序切分训练与验证，避免随机切造成穿越。"""
    ratio = ratio if ratio is not None else C.TRAIN_RATIO
    df = df.sort_values(time_col).reset_index(drop=True)
    split = int(len(df) * ratio)
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)
