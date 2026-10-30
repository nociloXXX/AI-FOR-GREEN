"""读取与清洗原始数据。真实字段在这里做映射。"""
from pathlib import Path

import pandas as pd

from . import config as C


def load_raw_charging(path=None):
    path = Path(path) if path else C.RAW_CHARGING_FILE
    df = pd.read_csv(path)
    missing = [col for col in C.REQUIRED_RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"充电数据缺少字段: {missing}")

    df = df.copy()
    df[C.TIME_START] = pd.to_datetime(df[C.TIME_START])
    df[C.TIME_END] = pd.to_datetime(df[C.TIME_END])
    return df.sort_values([C.STATION_COL, C.TIME_START]).reset_index(drop=True)


def clean_charging(df):
    """基础清洗。这里最容易稳定拿分，拿到真实数据后继续补规则。"""
    df = df.copy()
    df = df.dropna(
        subset=[C.STATION_COL, C.PILE_COL, C.TIME_START, C.TIME_END, C.ENERGY_COL]
    )
    df = df[df[C.TIME_END] > df[C.TIME_START]]
    df = df[df[C.ENERGY_COL] >= 0]
    df = df[df["start_soc"].between(0, 100) & df["end_soc"].between(0, 100)]
    df = df[df[C.MAX_POWER_COL] > 0]
    return df.reset_index(drop=True)


def load_price(path=None):
    path = Path(path) if path else C.PRICE_FILE
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_weather(path=None):
    path = Path(path) if path else C.WEATHER_FILE
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
