"""充电场站状态建模：把场站抽象成可推演的动态系统。

所谓“世界模型”，就是这个类加一组转移规则。真实比赛里先用规则状态机
跑通，有余力再换成学出来的转移模型。
"""
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Pile:
    pile_id: str
    busy: bool = False
    remaining_energy_kwh: float = 0.0
    max_power_kw: float = 0.0


@dataclass
class StationState:
    station_id: str
    piles: list = field(default_factory=list)
    queue: list = field(default_factory=list)
    capacity_kw: float = 500.0
    time: pd.Timestamp = None

    @property
    def occupied(self):
        return sum(1 for pile in self.piles if pile.busy)

    @property
    def free(self):
        return len(self.piles) - self.occupied

    def _start(self, pile, energy_kwh, max_power_kw):
        pile.busy = True
        pile.remaining_energy_kwh = float(energy_kwh)
        pile.max_power_kw = float(max_power_kw)

    def step(self, arrivals, per_pile_power_kw, step_hours):
        """推进一个时间步。

        arrivals: [{"energy_kwh": float, "max_power_kw": float}, ...]
        per_pile_power_kw: 本步每根充电桩分配功率
        step_hours: 时间步长（小时）
        """
        for arrival in arrivals:
            free_piles = [pile for pile in self.piles if not pile.busy]
            if free_piles:
                self._start(free_piles[0], arrival["energy_kwh"], arrival["max_power_kw"])
            else:
                self.queue.append(arrival)

        served_energy = 0.0
        for pile in self.piles:
            if not pile.busy:
                continue
            applied = min(per_pile_power_kw, pile.max_power_kw) * step_hours
            applied = min(applied, pile.remaining_energy_kwh)
            pile.remaining_energy_kwh -= applied
            served_energy += applied

            if pile.remaining_energy_kwh <= 1e-9:
                pile.busy = False
                pile.max_power_kw = 0.0
                if self.queue:
                    nxt = self.queue.pop(0)
                    self._start(pile, nxt["energy_kwh"], nxt["max_power_kw"])

        return served_energy
