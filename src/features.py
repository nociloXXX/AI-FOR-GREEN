"""特征工程与负荷曲线构建。"""
import numpy as np
import pandas as pd

from . import config as C


def add_time_features(df, time_col="timestamp"):
    df = df.copy()
    dt = pd.to_datetime(df[time_col])
    df["hour"] = dt.dt.hour
    df["minute_of_day"] = dt.dt.hour * 60 + dt.dt.minute
    df["weekday"] = dt.dt.weekday
    df["is_weekend"] = (dt.dt.weekday >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    return df


def add_lag_features(df, value_cols, lags=(1, 4, 24, 96)):
    """按时间排序后生成滞后与滚动均值特征。

    注意：这里必须按时间切分训练/验证，不能随机切。真实赛题里先确认
    在线评测是否隐藏未来，任何未来信息的特征都会在评测阶段失效。
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in value_cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        df[f"{col}_rolling24"] = df[col].rolling(24, min_periods=1).mean()
    return df


def merge_external(df, price, weather, on="timestamp"):
    """对齐电价与天气。真实字段可能不同，在这里改合并键。"""
    df = df.merge(price, on=on, how="left")
    if weather is not None:
        df = df.merge(weather, on=on, how="left")
    return df


def build_station_load_curve(sessions, freq_minutes=None):
    """把充电会话转成 15 分钟粒度的场站负荷曲线。

    简化假设：每段会话的电量在该会话持续时间内均匀释放。真实数据里
    功率曲线通常不是平的，这里可以替换成按真实功率采样。
    """
    freq_minutes = freq_minutes or C.STEP_MINUTES
    step_hours = freq_minutes / 60
    freq = f"{freq_minutes}min"

    start = sessions[C.TIME_START].min().floor(freq)
    end = sessions[C.TIME_END].max().ceil(freq)
    grid = pd.date_range(start, end, freq=freq)

    rows = []
    for session in sessions.itertuples():
        t = getattr(session, C.TIME_START).floor(freq)
        t_end = getattr(session, C.TIME_END)
        stamps = []
        while t < t_end:
            stamps.append(t)
            t += pd.Timedelta(minutes=freq_minutes)
        if not stamps:
            continue
        power = (getattr(session, C.ENERGY_COL) / len(stamps)) / step_hours
        for stamp in stamps:
            rows.append(
                {
                    "timestamp": stamp,
                    C.STATION_COL: getattr(session, C.STATION_COL),
                    C.LOAD_COL: power,
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
    return curve
