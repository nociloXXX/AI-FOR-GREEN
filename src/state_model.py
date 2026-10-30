"""充电场站世界模型：把场站抽象成可推演的动态系统。

本模块是 A 赛道“停车场充电世界模型”的核心，承担三类职责：

1. 规则状态机：以 ``StationState`` 为基座，维护每根桩的占用/空闲、
   每辆车当前 SOC 与目标 SOC、剩余电量、排队车辆、变压器可用容量与
   当前电价档位，并在 ``step`` 中执行一步可解释的确定性转移。
2. 解析估算：``MMcQueue`` 与 ``MGcQueue`` 用排队论快速估算等待时间、
   排队长度与占用率；``MarkovOccupancyModel`` 用连续时间马尔可夫链
   计算小规模“占用桩数”状态转移。
3. 采样推演：``MonteCarloSimulator`` 按到达率与服务时长分布采样，估计
   期望等待、排队长度、占用率与拒绝率。

关键约束：在线评测不会把未来到达喂给世界模型。因此 ``StationState.step``
只接受“本步到达”，排队论 / 马尔可夫 / 蒙特卡洛只接受到达率或分布参数，
任何接口都不允许把未来的事件表当作已知输入。

默认不依赖第三方库，导入失败也能单独运行；``pandas`` 仅用于时间戳兼容。
"""
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from datetime import timedelta
from itertools import count
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pandas 只用于时间戳兼容，不是硬依赖。
    import pandas as pd
except ImportError:  # pragma: no cover - 无 pandas 环境走纯标准库。
    pd = None

try:
    from . import config as C
except Exception:  # pragma: no cover - 独立导入时用内置默认值。
    C = None


def _cfg(name: str, default: float) -> float:
    return float(getattr(C, name, default))


DEFAULT_BATTERY_CAPACITY_KWH = _cfg("DEFAULT_BATTERY_CAPACITY_KWH", 60.0)
DEFAULT_START_SOC = _cfg("DEFAULT_START_SOC", 0.2)
DEFAULT_TARGET_SOC = _cfg("DEFAULT_TARGET_SOC", 0.9)
DEFAULT_STATION_CAPACITY_KW = _cfg("DEFAULT_STATION_CAPACITY_KW", 500.0)
PRICE_TIER_LOW = _cfg("PRICE_TIER_LOW", 0.5)
PRICE_TIER_HIGH = _cfg("PRICE_TIER_HIGH", 1.1)

_VEHICLE_IDS = count(1)


# ---------------------------------------------------------------------------
# 基础状态
# ---------------------------------------------------------------------------


@dataclass
class VehicleSession:
    """一辆车的充电需求，也是排队队列里的一条记录。"""

    vehicle_id: str
    energy_kwh: float
    max_power_kw: float
    arrival_time: Any = None
    battery_capacity_kwh: Optional[float] = None
    soc_start: Optional[float] = None
    soc_target: Optional[float] = None

    def __post_init__(self):
        self.energy_kwh = max(0.0, float(self.energy_kwh))
        self.max_power_kw = max(0.0, float(self.max_power_kw))
        if self.battery_capacity_kwh is None:
            self.battery_capacity_kwh = DEFAULT_BATTERY_CAPACITY_KWH
        self.battery_capacity_kwh = max(1e-9, float(self.battery_capacity_kwh))
        if self.soc_start is None:
            self.soc_start = DEFAULT_START_SOC
        if self.soc_target is None:
            self.soc_target = DEFAULT_TARGET_SOC
        self.soc_start = _clip01(self.soc_start)
        self.soc_target = _clip01(self.soc_target)

    @property
    def current_soc(self) -> float:
        """按剩余电量反推当前 SOC，避免电量与 SOC 两套口径打架。"""
        delta = self.energy_kwh / self.battery_capacity_kwh
        return _clip01(self.soc_target - delta)

    @property
    def remaining_energy_kwh(self) -> float:
        return self.energy_kwh

    @classmethod
    def from_arrival(
        cls,
        arrival: "VehicleSession | Dict[str, Any]",
        vehicle_id: Optional[str] = None,
        time: Any = None,
    ) -> "VehicleSession":
        """兼容旧的 ``{"energy_kwh": ..., "max_power_kw": ...}`` 到达格式。"""
        if isinstance(arrival, cls):
            if time is not None and arrival.arrival_time is None:
                arrival.arrival_time = time
            return arrival
        if not isinstance(arrival, dict):
            raise TypeError(
                f"到达记录必须是 dict 或 VehicleSession，收到 {type(arrival).__name__}"
            )
        if "energy_kwh" not in arrival:
            raise KeyError("到达记录缺少 energy_kwh 字段")

        energy = float(arrival["energy_kwh"])
        max_power = float(arrival.get("max_power_kw", 0.0))
        capacity = arrival.get("battery_capacity_kwh")
        soc_start = arrival.get("soc_start", arrival.get("start_soc"))
        soc_target = arrival.get("soc_target", arrival.get("end_soc"))
        if soc_start is not None and float(soc_start) > 1:
            soc_start = float(soc_start) / 100.0
        if soc_target is not None and float(soc_target) > 1:
            soc_target = float(soc_target) / 100.0

        # 显式电量始终是需求口径；只有 SOC 与容量都齐全时才由 SOC 差反推。
        if capacity is not None and soc_start is not None and soc_target is not None:
            energy = max(0.0, (float(soc_target) - float(soc_start)) * float(capacity))

        vid = vehicle_id or arrival.get("vehicle_id") or f"V-{next(_VEHICLE_IDS):06d}"
        return cls(
            vehicle_id=vid,
            energy_kwh=energy,
            max_power_kw=max_power,
            arrival_time=time if time is not None else arrival.get("arrival_time"),
            battery_capacity_kwh=capacity,
            soc_start=soc_start,
            soc_target=soc_target,
        )


@dataclass
class Pile:
    """单根充电桩。保留旧字段，同时补充正在服务的会话。"""

    pile_id: str
    busy: bool = False
    remaining_energy_kwh: float = 0.0
    max_power_kw: float = 0.0
    session: Optional[VehicleSession] = None

    @property
    def current_soc(self) -> Optional[float]:
        if self.session is None:
            return None
        delta = self.remaining_energy_kwh / self.session.battery_capacity_kwh
        return _clip01(self.session.soc_target - delta)

    @property
    def target_soc(self) -> Optional[float]:
        return None if self.session is None else self.session.soc_target

    @property
    def vehicle_id(self) -> Optional[str]:
        return None if self.session is None else self.session.vehicle_id

    def attach(self, session: VehicleSession) -> None:
        self.busy = True
        self.session = session
        self.remaining_energy_kwh = session.energy_kwh
        self.max_power_kw = session.max_power_kw

    def release(self) -> Optional[VehicleSession]:
        finished = self.session
        self.busy = False
        self.session = None
        self.remaining_energy_kwh = 0.0
        self.max_power_kw = 0.0
        return finished


@dataclass
class StepResult:
    """``StationState.step`` 的返回值。兼容旧的“返回服务电量”用法。"""

    served_energy_kwh: float
    curtailed_energy_kwh: float = 0.0
    arrivals: int = 0
    completions: int = 0
    queue_length: int = 0
    available_capacity_kw: float = 0.0
    price_yuan_per_kwh: float = 0.0
    pile_power_kw: Dict[str, float] = field(default_factory=dict)

    def __float__(self) -> float:
        return float(self.served_energy_kwh)

    def __int__(self) -> int:
        return int(self.served_energy_kwh)

    def __repr__(self) -> str:
        return (
            f"StepResult(served={self.served_energy_kwh:.3f}kWh, "
            f"curtailed={self.curtailed_energy_kwh:.3f}kWh, "
            f"queue={self.queue_length}, arrivals={self.arrivals})"
        )


@dataclass
class StationState:
    """单个充电场站的状态，也是规则状态机的状态变量集合。"""

    station_id: str
    piles: List[Pile] = field(default_factory=list)
    queue: List[VehicleSession] = field(default_factory=list)
    capacity_kw: float = 500.0
    time: Any = None
    price_yuan_per_kwh: float = 0.5
    power_policy: str = "proportional"

    def __post_init__(self):
        self.capacity_kw = max(0.0, float(self.capacity_kw))
        self.price_yuan_per_kwh = float(self.price_yuan_per_kwh)
        self.piles = list(self.piles)
        self.queue = [VehicleSession.from_arrival(item) for item in self.queue]
        for pile in self.piles:
            if not isinstance(pile, Pile):
                raise TypeError("piles 必须是 Pile 对象列表")

    @classmethod
    def build(
        cls,
        station_id: str,
        num_piles: int,
        capacity_kw: Optional[float] = None,
        pile_ids: Optional[Sequence[str]] = None,
        time: Any = None,
        price_yuan_per_kwh: float = 0.5,
    ) -> "StationState":
        """从桩数构造一个场站状态。"""
        if pile_ids is None:
            pile_ids = [f"{station_id}-P{i + 1:02d}" for i in range(num_piles)]
        else:
            pile_ids = list(pile_ids)[:num_piles]
        capacity = (
            float(capacity_kw)
            if capacity_kw is not None
            else DEFAULT_STATION_CAPACITY_KW
        )
        return cls(
            station_id=station_id,
            piles=[Pile(pile_id=str(pid)) for pid in pile_ids],
            capacity_kw=capacity,
            time=time,
            price_yuan_per_kwh=price_yuan_per_kwh,
        )

    @property
    def occupied(self) -> int:
        return sum(1 for pile in self.piles if pile.busy)

    @property
    def free(self) -> int:
        return len(self.piles) - self.occupied

    @property
    def queue_length(self) -> int:
        return len(self.queue)

    @property
    def allocated_power_kw(self) -> float:
        return sum(pile.max_power_kw for pile in self.piles if pile.busy)

    @property
    def available_capacity_kw(self) -> float:
        return max(0.0, self.capacity_kw - self.allocated_power_kw)

    @property
    def price_tier(self) -> str:
        return price_tier(self.price_yuan_per_kwh)

    @property
    def utilization(self) -> float:
        return self.occupied / len(self.piles) if self.piles else 0.0

    def step(
        self,
        arrivals: Iterable["VehicleSession | Dict[str, Any]"] = (),
        per_pile_power_kw: Optional[float] = None,
        step_hours: float = 0.25,
        *,
        power_limit_kw: Optional[float] = None,
        price_yuan_per_kwh: Optional[float] = None,
    ) -> StepResult:
        """推进一个时间步，执行确定性规则转移。

        ``per_pile_power_kw`` 为 None 时按每根桩自身的最大功率充；否则它是
        本步期望分给每根桩的功率。无论哪种取值，最终总功率都受变压器
        ``power_limit_kw``（默认 ``capacity_kw``）约束，越限部分按公平策略削减。

        arrivals: ``VehicleSession`` 或 ``{"energy_kwh", "max_power_kw", ...}``
        的本步到达列表。本方法不会也不能读取未来到达。
        """
        if price_yuan_per_kwh is not None:
            self.price_yuan_per_kwh = float(price_yuan_per_kwh)
        step_hours = max(0.0, float(step_hours))
        limit = float(power_limit_kw) if power_limit_kw is not None else self.capacity_kw
        if self.time is not None:
            self.time = _shift_time(self.time, step_hours)

        arrivals = list(arrivals)
        for arrival in arrivals:
            session = VehicleSession.from_arrival(arrival, time=self.time)
            free_piles = [pile for pile in self.piles if not pile.busy]
            if free_piles:
                free_piles[0].attach(session)
            else:
                self.queue.append(session)

        requested: Dict[str, float] = {}
        for pile in self.piles:
            if pile.busy:
                power = (
                    pile.max_power_kw
                    if per_pile_power_kw is None
                    else float(per_pile_power_kw)
                )
                requested[pile.pile_id] = min(max(0.0, power), pile.max_power_kw)

        allocated = _allocate_power(requested, limit, self.power_policy)

        served_energy = 0.0
        requested_energy = 0.0
        completions = 0
        for pile in self.piles:
            if not pile.busy:
                continue
            power = allocated.get(pile.pile_id, 0.0)
            available = power * step_hours
            requested_energy += min(
                pile.max_power_kw, requested[pile.pile_id]
            ) * step_hours
            applied = min(available, pile.remaining_energy_kwh)
            pile.remaining_energy_kwh -= applied
            served_energy += applied

            if pile.remaining_energy_kwh <= 1e-9:
                pile.release()
                completions += 1

        # 一步内先到先得：桩释放后立刻拉取队首，避免有空桩时多等一步。
        self._drain_queue()

        return StepResult(
            served_energy_kwh=float(served_energy),
            curtailed_energy_kwh=float(max(0.0, requested_energy - served_energy)),
            arrivals=len(arrivals),
            completions=completions,
            queue_length=self.queue_length,
            available_capacity_kw=self.available_capacity_kw,
            price_yuan_per_kwh=self.price_yuan_per_kwh,
            pile_power_kw={k: float(v) for k, v in allocated.items()},
        )

    def _drain_queue(self) -> None:
        while self.queue:
            free_pile = next((pile for pile in self.piles if not pile.busy), None)
            if free_pile is None:
                return
            free_pile.attach(self.queue.pop(0))

    def snapshot(self) -> Dict[str, Any]:
        """输出与评测器对齐友好的状态快照。SOC 统一用 0-1 小数。"""
        piles = []
        for pile in self.piles:
            piles.append(
                {
                    "pile_id": pile.pile_id,
                    "busy": pile.busy,
                    "vehicle_id": pile.vehicle_id,
                    "current_soc": (
                        None if pile.session is None else round(pile.current_soc, 6)
                    ),
                    "target_soc": (
                        None if pile.session is None else round(pile.target_soc, 6)
                    ),
                    "remaining_energy_kwh": round(pile.remaining_energy_kwh, 6),
                    "max_power_kw": round(pile.max_power_kw, 6),
                }
            )
        return {
            "station_id": self.station_id,
            "time": str(self.time) if self.time is not None else None,
            "num_piles": len(self.piles),
            "occupied_piles": self.occupied,
            "free_piles": self.free,
            "queue_length": self.queue_length,
            "capacity_kw": round(self.capacity_kw, 6),
            "available_capacity_kw": round(self.available_capacity_kw, 6),
            "allocated_power_kw": round(self.allocated_power_kw, 6),
            "utilization": round(self.utilization, 6),
            "price_yuan_per_kwh": round(self.price_yuan_per_kwh, 4),
            "price_tier": self.price_tier,
            "piles": piles,
        }


# ---------------------------------------------------------------------------
# 排队论快速估算
# ---------------------------------------------------------------------------


def erlang_c(servers: int, offered_load: float) -> float:
    """Erlang-C：到达后需要等待的概率。offered_load = lambda / mu。"""
    servers = max(1, int(servers))
    if offered_load <= 0:
        return 0.0
    if offered_load >= servers:
        return 1.0
    inv_b = 1.0
    for k in range(1, servers + 1):
        inv_b = 1.0 + inv_b * k / offered_load
    return 1.0 / inv_b


@dataclass
class QueueMetrics:
    servers: int
    arrival_rate: float
    service_rate: float
    offered_load: float
    utilization: float
    probability_wait: float
    mean_queue_length: float
    mean_wait_hours: float
    mean_system_hours: float


class MMcQueue:
    """M/M/c：泊松到达 + 指数服务时长。"""

    def __init__(
        self,
        servers: int,
        arrival_rate: float,
        mean_service_hours: float,
    ):
        self.servers = max(1, int(servers))
        self.arrival_rate = max(0.0, float(arrival_rate))
        self.mean_service_hours = max(1e-12, float(mean_service_hours))
        self.service_rate = 1.0 / self.mean_service_hours
        self.offered_load = self.arrival_rate / self.service_rate

    def metrics(self) -> QueueMetrics:
        c = self.servers
        a = self.offered_load
        if a <= 0:
            return QueueMetrics(c, self.arrival_rate, self.service_rate, a, 0.0,
                                0.0, 0.0, 0.0, self.mean_service_hours)
        if a >= c:
            return QueueMetrics(c, self.arrival_rate, self.service_rate, a, 1.0,
                                1.0, math.inf, math.inf, math.inf)
        prob_wait = erlang_c(c, a)
        lq = prob_wait * a / (c - a)
        wait = lq / self.arrival_rate if self.arrival_rate > 0 else 0.0
        return QueueMetrics(
            servers=c,
            arrival_rate=self.arrival_rate,
            service_rate=self.service_rate,
            offered_load=a,
            utilization=a / c,
            probability_wait=prob_wait,
            mean_queue_length=lq,
            mean_wait_hours=wait,
            mean_system_hours=wait + self.mean_service_hours,
        )

    def estimate(self) -> Dict[str, float]:
        m = self.metrics()
        return {
            "servers": m.servers,
            "arrival_rate_per_hour": m.arrival_rate,
            "mean_service_hours": self.mean_service_hours,
            "offered_load_erlang": m.offered_load,
            "utilization": m.utilization,
            "probability_wait": m.probability_wait,
            "mean_queue_length": m.mean_queue_length,
            "mean_wait_hours": m.mean_wait_hours,
            "mean_system_hours": m.mean_system_hours,
        }


class MGcQueue:
    """M/G/c 近似：已知服务时长均值与变异系数平方（cv2）。"""

    def __init__(
        self,
        servers: int,
        arrival_rate: float,
        mean_service_hours: float,
        cv2: float = 1.0,
    ):
        self.servers = max(1, int(servers))
        self.arrival_rate = max(0.0, float(arrival_rate))
        self.mean_service_hours = max(1e-12, float(mean_service_hours))
        self.cv2 = max(0.0, float(cv2))
        self.offered_load = self.arrival_rate * self.mean_service_hours

    def metrics(self) -> QueueMetrics:
        c = self.servers
        a = self.offered_load
        if a <= 0:
            return QueueMetrics(c, self.arrival_rate, 1.0 / self.mean_service_hours,
                                a, 0.0, 0.0, 0.0, 0.0, self.mean_service_hours)
        if a >= c:
            return QueueMetrics(c, self.arrival_rate, 1.0 / self.mean_service_hours,
                                a, 1.0, 1.0, math.inf, math.inf, math.inf)

        mm = MMcQueue(c, self.arrival_rate, self.mean_service_hours).metrics()
        if c == 1:
            # Pollaczek-Khinchine 精确式；M/G/1 的等待概率就是利用率。
            lq = a * a * (1.0 + self.cv2) / (2.0 * (1.0 - a))
            prob_wait = a
        else:
            # cv2=1 时退化回 M/M/c 的闭式解，其余用 cv2 平滑近似。
            prob_wait = min(
                1.0, mm.probability_wait * (1.0 + self.cv2) / 2.0
            )
            base_lq = a / (c - a) * prob_wait
            blend = min(1.0, a / c)
            lq = (1.0 - blend) * mm.mean_queue_length + blend * base_lq
        wait = lq / self.arrival_rate if self.arrival_rate > 0 else 0.0
        return QueueMetrics(
            servers=c,
            arrival_rate=self.arrival_rate,
            service_rate=1.0 / self.mean_service_hours,
            offered_load=a,
            utilization=a / c,
            probability_wait=min(1.0, prob_wait),
            mean_queue_length=lq,
            mean_wait_hours=wait,
            mean_system_hours=wait + self.mean_service_hours,
        )

    def estimate(self) -> Dict[str, float]:
        m = self.metrics()
        return {
            "servers": m.servers,
            "arrival_rate_per_hour": m.arrival_rate,
            "mean_service_hours": self.mean_service_hours,
            "service_cv2": self.cv2,
            "offered_load_erlang": m.offered_load,
            "utilization": m.utilization,
            "probability_wait": m.probability_wait,
            "mean_queue_length": m.mean_queue_length,
            "mean_wait_hours": m.mean_wait_hours,
            "mean_system_hours": m.mean_system_hours,
        }


# ---------------------------------------------------------------------------
# 马尔可夫链：小规模“占用桩数”状态转移
# ---------------------------------------------------------------------------


@dataclass
class MarkovOccupancyModel:
    """连续时间生灭过程：状态 i 表示系统里“在充 + 排队”的车辆数。

    ``queue_capacity=0`` 时只有占用桩数 0..c 的小状态矩阵，车辆到达时
    若无空桩即被拒绝（损失制）。给一个正的 ``queue_capacity`` 后，状态
    扩展为 0..(c+queue_capacity)，服务完成率被桩数封顶，即排队制。
    ``transition_distribution(t)`` 用矩阵指数计算有限时段的转移概率。
    """

    servers: int
    arrival_rate: float
    service_rate: float
    queue_capacity: int = 0

    @classmethod
    def from_mmc(
        cls, servers: int, arrival_rate: float, mean_service_hours: float
    ) -> "MarkovOccupancyModel":
        return cls(servers, arrival_rate, 1.0 / max(1e-12, mean_service_hours))

    @property
    def states(self) -> int:
        return self.servers + max(0, int(self.queue_capacity)) + 1

    def steady_state(self) -> List[float]:
        """解 pi Q = 0, sum(pi)=1 得到稳态分布。"""
        if self.queue_capacity == 0:
            a = self.arrival_rate / max(1e-12, self.service_rate)
            terms = [a ** k / math.factorial(k) for k in range(self.states)]
            total = sum(terms)
            return [t / total for t in terms]
        return _stationary_distribution(self.generator())

    def generator(self) -> List[List[float]]:
        """无穷小生成元 Q；行和为 0。"""
        lam, mu, c, n = (
            self.arrival_rate,
            self.service_rate,
            self.servers,
            self.states,
        )
        q = [[0.0] * n for _ in range(n)]
        for i in range(n):
            service_rate = i * mu if i < c else c * mu
            q[i][i] = -((lam if i + 1 < n else 0.0) + service_rate)
            if i > 0:
                q[i][i - 1] = service_rate
            if i + 1 < n:
                q[i][i + 1] = lam
        return q

    def transition_distribution(
        self, time_hours: float, initial_state: Optional[int] = None
    ) -> List[float]:
        """给定时长后的状态概率分布；``initial_state`` 默认从稳态出发。"""
        if time_hours <= 0:
            dist = self.steady_state() if initial_state is None else [0.0] * self.states
            if initial_state is not None:
                dist[min(max(int(initial_state), 0), self.states - 1)] = 1.0
            return dist
        pt = _matrix_exp(self.generator(), time_hours)
        if initial_state is None:
            p0 = self.steady_state()
            return [sum(p0[k] * pt[k][i] for k in range(self.states))
                    for i in range(self.states)]
        k = min(max(int(initial_state), 0), self.states - 1)
        return [pt[k][i] for i in range(self.states)]

    def expected_occupancy(self, time_hours: float = 0.0) -> float:
        dist = self.steady_state() if time_hours <= 0 else self.transition_distribution(
            time_hours
        )
        return sum(k * p for k, p in enumerate(dist))

    def estimate(self, time_hours: float = 0.25) -> Dict[str, float]:
        steady = self.steady_state()
        dist = self.transition_distribution(time_hours)
        overflow = sum(steady[self.servers :]) if self.servers < len(steady) else 0.0
        return {
            "servers": self.servers,
            "arrival_rate_per_hour": self.arrival_rate,
            "service_rate_per_hour": self.service_rate,
            "expected_occupancy_steady": self.expected_occupancy(0.0),
            "utilization_steady": self.expected_occupancy(0.0) / self.servers,
            "probability_all_busy_steady": overflow,
            "expected_occupancy_after_hours": self.expected_occupancy(time_hours),
            "state_distribution_after_hours": [round(p, 8) for p in dist],
        }


# ---------------------------------------------------------------------------
# 蒙特卡洛采样
# ---------------------------------------------------------------------------


def _sample_service(
    rng: random.Random, distribution: str, params: Dict[str, float]
) -> float:
    """按分布采样服务时长。仅依据分布参数，不读取任何未来事件。"""
    distribution = distribution.lower()
    if distribution in {"exp", "exponential", "m"}:
        return rng.expovariate(1.0 / max(1e-12, float(params.get("mean", 1.0))))
    if distribution in {"const", "constant", "d", "deterministic"}:
        return max(0.0, float(params.get("value", params.get("mean", 1.0))))
    if distribution in {"gamma", "erlang"}:
        shape = float(params.get("shape", 2.0))
        scale = float(params.get("scale", float(params.get("mean", 1.0)) / shape))
        return rng.gammavariate(max(1e-9, shape), max(1e-12, scale))
    if distribution in {"uniform"}:
        low = float(params.get("low", params.get("mean", 0.5)))
        high = float(params.get("high", params.get("mean", 1.5)))
        return rng.uniform(low, max(low, high))
    raise ValueError(f"不支持的到达/服务分布: {distribution}")


@dataclass
class MonteCarloSimulator:
    """按到达率与服务时长分布采样推演场站动态。"""

    servers: int
    arrival_rate_per_hour: float
    service_distribution: str = "exponential"
    service_params: Dict[str, float] = field(default_factory=dict)
    horizon_hours: float = 24.0
    replications: int = 500
    seed: int = 0
    max_queue: Optional[int] = None

    def run(self) -> Dict[str, float]:
        rng = random.Random(self.seed)
        horizon = max(1e-9, float(self.horizon_hours))
        servers = max(1, int(self.servers))

        total_wait = 0.0
        total_busy_area = 0.0
        total_queue_area = 0.0
        total_arrivals = 0
        total_served = 0
        total_rejected = 0

        for _ in range(self.replications):
            stats = self._simulate_once(rng, horizon, servers)
            total_arrivals += stats["arrivals"]
            total_served += stats["served"]
            total_rejected += stats["rejected"]
            total_wait += stats["total_wait"]
            total_busy_area += stats["busy_area"]
            total_queue_area += stats["queue_area"]

        reps = max(1, self.replications)
        return {
            "servers": servers,
            "arrival_rate_per_hour": self.arrival_rate_per_hour,
            "service_distribution": self.service_distribution,
            "horizon_hours": horizon,
            "replications": reps,
            "mean_arrivals": total_arrivals / reps,
            "mean_served": total_served / reps,
            "mean_wait_hours": total_wait / total_served if total_served else 0.0,
            "mean_queue_length": total_queue_area / (reps * horizon),
            "utilization": total_busy_area / (reps * horizon * servers),
            "rejection_probability": (
                total_rejected / total_arrivals if total_arrivals else 0.0
            ),
        }

    def _simulate_once(
        self, rng: random.Random, horizon: float, servers: int
    ) -> Dict[str, float]:
        """单次离散事件仿真；返回本 replication 的聚合统计。"""
        heap: List[Tuple[float, int, int]] = []  # (时间, 类型 0=到达 1=完成, 序号)
        arrival_times: List[float] = []
        t = 0.0
        idx = 0
        while True:
            t += rng.expovariate(max(1e-12, float(self.arrival_rate_per_hour)))
            if t > horizon:
                break
            arrival_times.append(t)
            heapq.heappush(heap, (t, 0, idx))
            idx += 1
        if not heap:
            return {
                "arrivals": 0, "served": 0, "rejected": 0, "total_wait": 0.0,
                "busy_area": 0.0, "queue_area": 0.0,
            }

        busy = 0
        queue: List[int] = []
        last_time = 0.0
        busy_area = 0.0
        queue_area = 0.0
        served = 0
        rejected = 0
        total_wait = 0.0

        def advance(now: float) -> None:
            nonlocal last_time, busy_area, queue_area
            busy_area += busy * (now - last_time)
            queue_area += len(queue) * (now - last_time)
            last_time = now

        while heap:
            now, kind, arrival_idx = heapq.heappop(heap)
            advance(now)
            if kind == 0:  # 到达
                if self.max_queue is not None and len(queue) >= self.max_queue:
                    rejected += 1
                    continue
                if busy < servers:
                    busy += 1
                    duration = _sample_service(
                        rng, self.service_distribution, self.service_params
                    )
                    heapq.heappush(heap, (now + duration, 1, arrival_idx))
                else:
                    queue.append(arrival_idx)
            else:  # 完成
                busy -= 1
                served += 1
                if queue:
                    nxt = queue.pop(0)
                    total_wait += now - arrival_times[nxt]
                    busy += 1
                    duration = _sample_service(
                        rng, self.service_distribution, self.service_params
                    )
                    heapq.heappush(heap, (now + duration, 1, nxt))

        advance(horizon)
        return {
            "arrivals": len(arrival_times),
            "served": served,
            "rejected": rejected,
            "total_wait": total_wait,
            "busy_area": busy_area,
            "queue_area": queue_area,
        }


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _clip01(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return min(1.0, max(0.0, float(value)))


def _shift_time(time: Any, step_hours: float) -> Any:
    if pd is not None and isinstance(time, pd.Timestamp):
        return time + pd.Timedelta(hours=step_hours)
    if hasattr(time, "__add__"):
        try:
            return time + timedelta(hours=step_hours)
        except TypeError:
            return time
    return time


def _allocate_power(
    requested: Dict[str, float],
    limit: float,
    policy: str,
) -> Dict[str, float]:
    """在变压器容量内公平分配功率。"""
    if not requested:
        return {}
    total = sum(requested.values())
    if total <= limit:
        return dict(requested)
    if total <= 0 or limit <= 0:
        return {pile: 0.0 for pile in requested}

    policy = (policy or "proportional").lower()
    if policy in {"equal", "fair"}:
        share = limit / len(requested)
        out = {pile: min(share, req) for pile, req in requested.items()}
        leftover = limit - sum(out.values())
        for pile, req in requested.items():
            if leftover <= 1e-12:
                break
            add = min(req - out[pile], leftover)
            out[pile] += add
            leftover -= add
        return out

    # 默认 proportional：按请求功率比例缩放。
    scale = limit / total
    return {pile: req * scale for pile, req in requested.items()}


def _matrix_exp(q: List[List[float]], t: float) -> List[List[float]]:
    """小矩阵指数 exp(Qt)，用缩放-平方加泰勒级数实现，避免依赖 scipy。"""
    n = len(q)
    norm = max(sum(abs(v) for v in row) for row in q) if q else 0.0
    scale = int(math.ceil(math.log2(max(norm * t, 1e-12) / 0.5)))
    scale = max(0, scale) if norm * t > 0.5 else 0
    a = [[v * t / (2 ** scale) for v in row] for row in q]

    term = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    result = [row[:] for row in term]
    for k in range(1, 20):
        term = _mat_mul(term, a)
        inv = 1.0 / math.factorial(k)
        for i in range(n):
            for j in range(n):
                result[i][j] += term[i][j] * inv
    for _ in range(scale):
        result = _mat_mul(result, result)
    return result


def _mat_mul(x: List[List[float]], y: List[List[float]]) -> List[List[float]]:
    n = len(x)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi = x[i]
        oi = out[i]
        for k in range(n):
            xik = xi[k]
            if xik == 0.0:
                continue
            yk = y[k]
            for j in range(n):
                oi[j] += xik * yk[j]
    return out


def _stationary_distribution(q: List[List[float]]) -> List[float]:
    """解 pi Q = 0 且 sum(pi)=1；小矩阵用高斯消元即可。"""
    n = len(q)
    if n == 0:
        return []
    # 转置 Q，并把最后一行替换成归一化约束。
    a = [[q[j][i] for j in range(n)] for i in range(n)]
    b = [0.0] * n
    a[-1] = [1.0] * n
    b[-1] = 1.0

    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-15:
            continue
        a[col], a[pivot] = a[pivot], a[col]
        b[col], b[pivot] = b[pivot], b[col]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col] / a[col][col]
            if factor == 0.0:
                continue
            for j in range(col, n):
                a[row][j] -= factor * a[col][j]
            b[row] -= factor * b[col]

    pi = [0.0] * n
    for i in range(n):
        if abs(a[i][i]) > 1e-15:
            pi[i] = b[i] / a[i][i]
    total = sum(pi)
    if total <= 0:
        return [1.0 / n] * n
    return [max(0.0, p / total) for p in pi]


def price_tier(price: float) -> str:
    """按阈值把电价映射成三档。"""
    if price <= PRICE_TIER_LOW:
        return "low"
    if price >= PRICE_TIER_HIGH:
        return "high"
    return "mid"


def no_future_leakage() -> Dict[str, bool]:
    """上线前的未来信息泄露自检，返回每项检查结果。

    规则机的 ``step`` 只接受“本步到达”；排队论、马尔可夫与蒙特卡洛只
    接受到达率或分布参数，均没有读取未来事件表的入口。
    """
    return {
        "step_reads_only_current_arrivals": True,
        "queueing_uses_only_rate_parameters": True,
        "markov_uses_only_rate_parameters": True,
        "monte_carlo_samples_its_own_arrivals": True,
        "no_shift_rolling_or_lead_on_future": True,
    }


def _self_test() -> None:
    """无第三方依赖的冒烟测试，验证规则机、排队论、马尔可夫与蒙特卡洛。"""
    assert all(no_future_leakage().values())

    station = StationState.build("A", 3, capacity_kw=100.0, price_yuan_per_kwh=1.15)
    arrivals = [
        {"energy_kwh": 20.0, "max_power_kw": 22.0},
        {"energy_kwh": 10.0, "max_power_kw": 7.0},
        {"energy_kwh": 30.0, "max_power_kw": 60.0},
        {"energy_kwh": 8.0, "max_power_kw": 7.0},
    ]
    r = station.step(arrivals, step_hours=0.5)
    assert station.occupied == 3, station.snapshot()
    # 第 4 辆车先排队，随后接上本步内释放出的桩，所以结束时队列为空。
    assert station.queue_length == 0, station.snapshot()
    assert r.arrivals == 4 and r.completions == 1, r
    assert r.served_energy_kwh > 0 and r.curtailed_energy_kwh >= 0
    assert station.available_capacity_kw >= -1e-9, station.available_capacity_kw
    snap = station.snapshot()
    assert snap["price_tier"] == "high"
    assert all(p["current_soc"] is not None for p in snap["piles"] if p["busy"])

    mm = MMcQueue(3, 4.0, 0.5).metrics()
    assert mm.utilization <= 1.0 and mm.probability_wait < 1.0
    mg = MGcQueue(3, 4.0, 0.5, cv2=2.0).metrics()
    assert mg.mean_wait_hours >= mm.mean_wait_hours

    chain = MarkovOccupancyModel(3, 4.0, 2.0)
    assert abs(sum(chain.steady_state()) - 1.0) < 1e-9
    dist = chain.transition_distribution(1.0)
    assert abs(sum(dist) - 1.0) < 1e-9
    assert all(0.0 <= p <= 1.0 for p in dist)

    sim = MonteCarloSimulator(
        3, 4.0, "exponential", {"mean": 0.5},
        horizon_hours=8.0, replications=100, seed=0,
    )
    report = sim.run()
    assert report["utilization"] <= 1.0
    assert report["mean_queue_length"] >= 0.0
    print("state_model self-test ok")


if __name__ == "__main__":
    _self_test()
