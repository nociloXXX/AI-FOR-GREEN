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

# 决策模块：默认走零前瞻的因果削峰填谷，保证在线评测不使用未来信息。
# 可选 "mpc"（只用预测值做小步长前瞻）；"lp_load_shift" 仅限离线分析。
DECISION_STRATEGY = "causal"
MPC_HORIZON_STEPS = HORIZON_STEPS

# 干预推演：候选策略列表与评分惩罚。默认只含因果与预测前瞻，在线安全；
# 离线分析可临时加入 "lp_load_shift" 做全量能量守恒平移。
INTERVENTION_CANDIDATES = ["causal", "mpc"]
INTERVENTION_CURTAIL_PENALTY = 2.0
INTERVENTION_VIOLATION_PENALTY = 1000.0

# 储能调度参数。
BATTERY_EFFICIENCY = 0.95
BATTERY_SOC_MIN = 0.1
BATTERY_SOC_MAX = 1.0

# 合成数据里每个场站的变压器容量上限（kW）。真实比赛替换为官方给出的容量。
STATION_CAPACITY_KW = {
    "A": 300,
    "B": 400,
    "C": 200,
}

# 每个场站的充电桩数量。真实比赛替换为官方给的桩数口径。
STATION_PILES = {
    "A": 6,
    "B": 8,
    "C": 4,
}

# 世界模型的车辆口径：电池容量、默认起始/目标 SOC。SOC 统一用 0-1 小数，
# 外部 0-100 的百分数在 state_model.VehicleSession 里自动换算。
DEFAULT_BATTERY_CAPACITY_KWH = 60.0
DEFAULT_START_SOC = 0.2
DEFAULT_TARGET_SOC = 0.9

# 电价档位阈值（元/kWh）：<= LOW 为谷档，>= HIGH 为峰档，中间为平档。
PRICE_TIER_LOW = 0.5
PRICE_TIER_HIGH = 1.1

# TODO: 开题后替换为官方字段名。
TIME_START = "start_time"
TIME_END = "end_time"
START_SOC_COL = "start_soc"
END_SOC_COL = "end_soc"
STATION_COL = "station_id"
PILE_COL = "pile_id"
ENERGY_COL = "energy_kwh"
MAX_POWER_COL = "max_power_kw"

REQUIRED_RAW_COLUMNS = [
    STATION_COL,
    PILE_COL,
    TIME_START,
    TIME_END,
    START_SOC_COL,
    END_SOC_COL,
    ENERGY_COL,
    MAX_POWER_COL,
]

# 构建负荷曲线后生成的列名。
LOAD_COL = "power_kw"
OCCUPANCY_COL = "active_piles"

# 提交文件的列名与顺序。官方评测器未公开前，这是本仓库的格式契约；
# 开题后对照官方要求只改这里，pipeline 会按此校验并写出。
SUBMISSION_COLUMNS = [
    "timestamp",
    STATION_COL,
    "predicted_power_kw",
    "dispatch_power_kw",
]

# 电价与天气表的时间列和数值列。真实字段如果不同，在这里改。
TIME_COL = "timestamp"
PRICE_VALUE_COL = "price_yuan_per_kwh"
TEMPERATURE_COL = "temperature_c"

# 原始字段映射：开题后把官方列名填到右侧规范名，data_loader 会自动重命名。
# 示例: RAW_COLUMN_ALIASES = {"charge_start": TIME_START}
RAW_COLUMN_ALIASES = {}
PRICE_COLUMN_ALIASES = {}
WEATHER_COLUMN_ALIASES = {}

# ---------- 清洗与异常检测 ----------
# "iqr" | "sigma" | "isolation_forest" | "none"
ANOMALY_METHOD = "iqr"
IQR_K = 1.5
SIGMA = 3.0
ISOLATION_CONTAMINATION = "auto"

# 在“单次充电的功率利用率”上做异常检测，避免把大功率桩的正常大电量会话误删。
# 利用率 = 电量 / (最大功率 * 时长)，理论上落在 [0, 1]。
UTILIZATION_COL = "utilization"
UTILIZATION_MAX = 1.01

# 缺失电量反推：energy = max_power * 时长(小时) * 因子。
BACKFILL_ENERGY = True
ENERGY_BACKFILL_FACTOR = 1.0

# SOC 合法范围，超出范围的会话视为异常。
SOC_MIN = 0.0
SOC_MAX = 100.0

# ---------- 负荷曲线 ----------
# 充电功率先恒功率后线性衰减，衰减到底部保留的比例。真实功率曲线可在此基础上替换。
TAPER_FRACTION = 0.65
TAPER_FLOOR = 0.2

# ---------- 特征 ----------
LAGS = (1, 4, 24, 96)
ROLLING_WINDOW = 24
