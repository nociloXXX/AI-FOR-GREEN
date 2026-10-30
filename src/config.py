"""比赛配置。开题拿到真实数据后，优先改这一份文件。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SUBMISSION_DIR = ROOT / "submissions"
FIGURE_DIR = ROOT / "figures"

RAW_CHARGING_FILE = DATA_DIR / "raw_charging.csv"
PRICE_FILE = DATA_DIR / "price.csv"
WEATHER_FILE = DATA_DIR / "weather.csv"
ARTIFACTS_FILE = SUBMISSION_DIR / "run_artifacts.csv"

STEP_MINUTES = 15
HORIZON_STEPS = 4
TRAIN_RATIO = 0.8

# 合成数据里每个场站的变压器容量上限（kW）。真实比赛替换为官方给出的容量。
STATION_CAPACITY_KW = {
    "A": 300,
    "B": 400,
    "C": 200,
}

# TODO: 开题后替换为官方字段名。
TIME_START = "start_time"
TIME_END = "end_time"
STATION_COL = "station_id"
PILE_COL = "pile_id"
ENERGY_COL = "energy_kwh"
MAX_POWER_COL = "max_power_kw"

REQUIRED_RAW_COLUMNS = [
    STATION_COL,
    PILE_COL,
    TIME_START,
    TIME_END,
    "start_soc",
    "end_soc",
    ENERGY_COL,
    MAX_POWER_COL,
]

# 构建负荷曲线后生成的列名。
LOAD_COL = "power_kw"
OCCUPANCY_COL = "active_piles"
