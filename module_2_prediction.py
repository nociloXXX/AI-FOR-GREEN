"""模块2：预测。

用 LightGBM（L1/MAE 目标）学习每辆车的需求电量与停留时长，并用星期x时段
到达率表估计期望到达数。特征包含小时/星期的周期编码、充电桩功率上限，以及
``last_requested_energy_kwh``（缺失保留 NaN，不当作 0）。

沙箱内优先加载 ``weights/`` 下的 LightGBM 文本权重，不重新训练；权重缺失或
加载失败时才用公开训练数据现场拟合。整个模块只读取公开历史，不读取未来到达、
真实离开时间或隐藏需求标签。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from acn_causal.student_api import Prediction


_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
_SITES = ("caltech", "jpl", "office1")


def _make_features(hour, weekday, pmax, requested):
    """构造需求/停留回归特征；requested 的缺失值保留为 NaN。"""
    hour = np.asarray(hour, dtype=float)
    weekday = np.asarray(weekday, dtype=float)
    pmax = np.asarray(pmax, dtype=float)
    requested = np.asarray(requested, dtype=float)
    missing = np.isnan(requested).astype(float)
    return np.column_stack(
        [
            np.sin(hour * np.pi / 12.0),
            np.cos(hour * np.pi / 12.0),
            np.sin(weekday * 2.0 * np.pi / 7.0),
            np.cos(weekday * 2.0 * np.pi / 7.0),
            weekday,
            pmax,
            requested,
            missing,
        ]
    )


def _fit_booster(x, y, num_leaves):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=num_leaves,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=20261008,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(x, y)
    return model.booster_


def _fit_extra_trees(x, y):
    """LightGBM 不可用时保留的 ExtraTrees 基线（单目标）。"""
    from sklearn.ensemble import ExtraTreesRegressor

    model = ExtraTreesRegressor(
        n_estimators=96,
        min_samples_leaf=15,
        max_depth=12,
        n_jobs=1,
        random_state=20261008,
    )
    model.fit(np.nan_to_num(x, nan=0.0), y)
    return model


def _load_booster(path):
    import lightgbm as lgb

    if not Path(path).is_file():
        raise FileNotFoundError(path)
    return lgb.Booster(model_file=str(path))


def _build_arrival_rates(training):
    """返回每站点分星期x时段的到达率表，单位：辆/15分钟步。"""
    rates = {}
    for site in _SITES:
        frame = training[site]
        days = pd.date_range(frame.day.min(), frame.day.max(), freq="D")
        weekday_days = np.bincount(days.dayofweek, minlength=7).astype(float)
        total_days = max(float(len(days)), 1.0)

        # 数据充足用 15 分钟粒度，数据稀疏退到小时粒度。
        sessions_per_day = len(frame) / total_days
        if sessions_per_day >= 10.0:
            nbins = 96
            steps_per_bin = 1.0
            bins = np.clip(np.rint(frame.hour.to_numpy() * 4.0).astype(int) % 96, 0, 95)
        else:
            nbins = 24
            steps_per_bin = 4.0
            bins = np.clip(frame.hour.to_numpy().astype(int) % 24, 0, 23)

        counts = np.zeros((7, nbins))
        weekdays = np.asarray(frame.weekday, dtype=int) % 7
        np.add.at(counts, (weekdays, bins), 1.0)

        observed = weekday_days[:, None] * steps_per_bin
        mean_rate = len(frame) / (total_days * 96.0)

        # 在计数空间做先验收缩：每个 weekday-hour 单元加入 pseudo_days 天
        # 的站点平均到达数，避免小样本单元被噪声主导。
        pseudo_days = 3.0
        pseudo = mean_rate * steps_per_bin * pseudo_days
        rate = (counts + pseudo) / (observed + steps_per_bin * pseudo_days)

        # 沿时段做小窗圆形平滑，保留相邻时段的连续性。
        smoothed = rate.copy()
        prev = np.roll(rate, 1, axis=1)
        nxt = np.roll(rate, -1, axis=1)
        smoothed = 0.25 * prev + 0.5 * rate + 0.25 * nxt
        rates[site] = (smoothed, nbins)
    return rates


class PredictionModule:
    """需求/停留 LightGBM 回归器 + 星期x时段到达率表。"""

    def __init__(self, training):
        self.demand = {}
        self.dwell = {}
        for site in _SITES:
            frame = training[site]
            requested = (
                frame["last_requested_energy_kwh"]
                if "last_requested_energy_kwh" in frame.columns
                else np.full(len(frame), np.nan)
            )
            x = _make_features(
                frame.hour, frame.weekday, frame.charger_limit_kw, requested
            )
            self.demand[site] = self._model_or_fit(
                _WEIGHTS_DIR / f"demand_{site}.txt", x, frame.demand_kwh, 31
            )
            self.dwell[site] = self._model_or_fit(
                _WEIGHTS_DIR / f"dwell_{site}.txt", x, frame.dwell_hours, 15
            )
        self.arrival_rates = _build_arrival_rates(training)
        self._cache = {}

    @staticmethod
    def _model_or_fit(path, x, y, num_leaves):
        try:
            return _load_booster(path)
        except Exception:
            pass
        try:
            return _fit_booster(x, y, num_leaves)
        except Exception:
            return _fit_extra_trees(x, y)

    @staticmethod
    def _model_predict(model, x):
        # LightGBM Booster 原生处理 NaN；ExtraTrees 需要把 NaN 置 0。
        if hasattr(model, "num_feature"):
            return model.predict(x)
        return model.predict(np.nan_to_num(x, nan=0.0))

    def predict(self, features):
        requested = getattr(features, "last_requested_energy_kwh", None)
        if requested is not None and not np.isfinite(float(requested)):
            requested = None
        key = (
            features.site,
            round(float(features.hour), 4),
            int(features.weekday),
            round(float(features.charger_limit_kw), 3),
            None if requested is None else round(float(requested), 3),
        )
        if key not in self._cache:
            x = _make_features(
                [features.hour],
                [features.weekday],
                [features.charger_limit_kw],
                [np.nan if requested is None else float(requested)],
            )
            demand = float(self._model_predict(self.demand[features.site], x)[0])
            dwell = float(self._model_predict(self.dwell[features.site], x)[0])
            self._cache[key] = Prediction(
                float(np.clip(demand, 0.01, 150.0)),
                float(np.clip(dwell, 0.25, 24.0)),
            )
        return self._cache[key]

    def arrival_rate(self, site, hour, weekday):
        table, nbins = self.arrival_rates[site]
        hour = float(hour) % 24.0
        weekday = int(weekday) % 7
        if nbins == 96:
            bin_idx = int(np.rint(hour * 4.0)) % 96
        else:
            bin_idx = int(hour) % 24
        return float(table[weekday, bin_idx])
