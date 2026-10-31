"""选手接口的公开数据结构（JSON 可序列化）。

本模块是公开模块，随参赛包分发；它只定义"数据长什么样"，
不包含裁判生成器、隐藏场景或真值标签（裁判模块与标签单独存放，不在包内）。

选手的 submission.py、网站的评测沙箱、本地 public_self_eval.py
都使用同一份定义，保证三方字段一致。请不要修改本文件：
改动会让你的提交与评测器对不上。
"""
from dataclasses import dataclass,asdict
from typing import Tuple, Optional

@dataclass(frozen=True)
class ArrivalFeatures:
    """一辆车到达时模型可见的信息，是 predict() 的输入。"""
    token:str                     # 车辆标识（本次到达的唯一记号）
    site:str                      # 站点：caltech / jpl / office1
    hour:float                    # 到达时刻（小时，0-24，含小数）
    weekday:int                   # 星期几（0=周一 … 6=周日）
    charger_limit_kw:float        # 所接充电桩的功率上限（kW）
    last_requested_energy_kwh:Optional[float]=None
    # 该车最后一次申报的充电需求（kWh）。这是最终提交的参考快照，
    # 填写时间未经核实，不能当作到达时刻就已知；缺失时为 None，不能当 0。

@dataclass(frozen=True)
class VehicleObservation:
    """在线评测某一步中，一辆在场车辆的可见状态。"""
    features:ArrivalFeatures      # 该车到达时的特征
    connected_step:int            # 接入时的步序号
    elapsed_steps:int             # 已连接的步数
    delivered_kwh:float           # 已交付电量（kWh）
    charging_complete:bool        # 是否已充完

@dataclass(frozen=True)
class Observation:
    """act() / forecast() 每一步收到的站点观测。"""
    step:int                             # 当前步序号
    dt_hours:float                       # 每步时长（小时）
    horizon_steps:int                    # 场景总步数
    site:str                             # 站点
    local_hour:float                     # 当前当地时刻（小时）
    weekday:int                          # 当前星期几（0=周一 … 6=周日）
    vehicles:Tuple[VehicleObservation,...]  # 当前在场车辆
    capacity_kw:float                    # 当前站点总容量（kW，行动限额之一）
    background_kw:float                  # 背景（非充电）负荷（kW）
    prices:Tuple[float,...]              # 电价序列
    forecast_capacity:Tuple[float,...]   # 预测站点容量（行动限额之一）
    forecast_background:Tuple[float,...] # 预测背景负荷
    historical_arrivals:Tuple[int,...]   # 历史同时段到达数
    intervention_capacity:Tuple[float,...]=()
    # 公开的反事实站点总容量序列，只用于 forecast 的干预情景；
    # 不能当作 act() 的功率限额。

@dataclass(frozen=True)
class Prediction:
    """predict() 的输出：对一辆车的需求与停留预测。"""
    demand_kwh:float    # 预测需求电量（kWh）
    dwell_hours:float   # 预测停留时长（小时）

# ---------------------------------------------------------------------------
# 选手接口约定（在 submission.py 中实现；方法名、参数与返回结构不能改）
#   predict(ArrivalFeatures) -> Prediction
#   arrival_rate(site, hour, weekday) -> 每 15 分钟的期望到达数
#   act(Observation) -> {token: 请求功率 kW}
#   forecast(Observation, horizon_steps, intervention) -> 列表字典
#       {ev_load_kw, active_count}；intervention 取 'natural' 或 'capacity_edf'
#   explain() -> {'assumptions': [str,...], 'limitations': [str,...], 'method': str}
# ---------------------------------------------------------------------------
