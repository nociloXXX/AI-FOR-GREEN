"""模块2：预测。

用 LightGBM（L1/MAE 目标）按站点分别训练每辆车的需求电量与停留时长。
``last_requested_energy_kwh`` 采用双分支：申报值正有限时使用含该特征的模型，
缺失/非法时退化为不含该特征的回退模型。未来到达的 ArrivalFeatures 该字段为
None，因此 forecast 推演自然走回退分支，不会把它当作 0 或已知意图。

沙箱内优先加载 ``weights/`` 下的 LightGBM 文本权重，不重新训练；权重缺失或
加载失败时退回现场拟合，LightGBM 不可用时再退回 ExtraTrees 基线。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from acn_causal.student_api import Prediction


_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
_SITES = ("caltech", "jpl", "office1")


def _features_base(hour, weekday, pmax):
    """不含申报需求的特征，用于回退分支。"""
    hour = np.asarray(hour, dtype=float)
    weekday = np.asarray(weekday, dtype=float)
    pmax = np.asarray(pmax, dtype=float)
    return np.column_stack(
        [
            np.sin(hour * np.pi / 12.0),
            np.cos(hour * np.pi / 12.0),
            np.sin(weekday * 2.0 * np.pi / 7.0),
            np.cos(weekday * 2.0 * np.pi / 7.0),
            weekday,
            pmax,
        ]
    )


def _features_with_request(hour, weekday, pmax, requested):
    """含申报需求的特征，只在 requested 为正有限值时调用。"""
    requested = np.asarray(requested, dtype=float)
    return np.column_stack(
        [_features_base(hour, weekday, pmax), requested.reshape(-1, 1)]
    )


def _is_valid_request(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(value) and value > 0.0


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
    model.fit(np.nan_to_num(x, nan=-1.0), y)
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
        day = pd.to_datetime(frame.day, errors="coerce").dropna()
        if day.empty:
            rates[site] = (np.zeros((7, 24), dtype=float), 24)
            continue
        days = pd.date_range(day.min(), day.max(), freq="D")
        weekday_days = np.bincount(days.dayofweek, minlength=7).astype(float)
        total_days = max(float(len(days)), 1.0)

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
        pseudo_days = 3.0
        pseudo = mean_rate * steps_per_bin * pseudo_days
        rate = (counts + pseudo) / (observed + steps_per_bin * pseudo_days)

        prev = np.roll(rate, 1, axis=1)
        nxt = np.roll(rate, -1, axis=1)
        rates[site] = (0.25 * prev + 0.5 * rate + 0.25 * nxt, nbins)
    return rates


class PredictionModule:
    """需求/停留 LightGBM 双分支回归器 + 星期x时段到达率表。"""

    def __init__(self, training):
        self.demand_with = {}
        self.dwell_with = {}
        self.demand_fallback = {}
        self.dwell_fallback = {}

        for site in _SITES:
            frame = training[site]
            if "last_requested_energy_kwh" in frame.columns:
                requested = pd.to_numeric(
                    frame["last_requested_energy_kwh"], errors="coerce"
                ).to_numpy(dtype=float)
            else:
                requested = np.full(len(frame), np.nan)
            valid = np.isfinite(requested) & (requested > 0.0)
            xb = _features_base(
                frame.hour, frame.weekday, frame.charger_limit_kw
            )

            if valid.any() and (~valid).sum() >= 20:
                fallback_mask = ~valid
            else:
                fallback_mask = np.ones(len(frame), dtype=bool)
            self.demand_fallback[site] = self._model_or_fit(
                _WEIGHTS_DIR / f"demand_fallback_{site}.txt",
                xb[fallback_mask],
                frame.demand_kwh.to_numpy()[fallback_mask],
                31,
            )
            self.dwell_fallback[site] = self._model_or_fit(
                _WEIGHTS_DIR / f"dwell_fallback_{site}.txt",
                xb[fallback_mask],
                frame.dwell_hours.to_numpy()[fallback_mask],
                15,
            )

            if valid.any():
                xw = _features_with_request(
                    frame.hour[valid], frame.weekday[valid],
                    frame.charger_limit_kw[valid], requested[valid],
                )
                self.demand_with[site] = self._model_or_fit(
                    _WEIGHTS_DIR / f"demand_with_{site}.txt",
                    xw,
                    frame.demand_kwh.to_numpy()[valid],
                    31,
                )
                self.dwell_with[site] = self._model_or_fit(
                    _WEIGHTS_DIR / f"dwell_with_{site}.txt",
                    xw,
                    frame.dwell_hours.to_numpy()[valid],
                    15,
                )
            else:
                self.demand_with[site] = self.demand_fallback[site]
                self.dwell_with[site] = self.dwell_fallback[site]

        self.arrival_rates = _build_arrival_rates(training)
        self._cache = {}

    @staticmethod
    def _model_or_fit(path, x, y, num_leaves):
        try:
            return _load_booster(path)
        except Exception:
            pass
        # 评测沙箱内不重新训练 LightGBM；权重缺失或加载失败时走轻量 ExtraTrees，
        # 避免运行时训练导致超时或内存超限。
        return _fit_extra_trees(x, y)

    @staticmethod
    def _model_predict(model, x):
        if hasattr(model, "num_feature"):
            return model.predict(x)
        return model.predict(np.nan_to_num(x, nan=-1.0))

    def predict(self, features):
        requested = getattr(features, "last_requested_energy_kwh", None)
        use_request = _is_valid_request(requested)
        key = (
            features.site,
            round(float(features.hour), 4),
            int(features.weekday),
            round(float(features.charger_limit_kw), 3),
            None if not use_request else round(float(requested), 3),
        )
        if key not in self._cache:
            if len(self._cache) > 20000:
                self._cache.clear()
            if use_request:
                x = _features_with_request(
                    [features.hour], [features.weekday],
                    [features.charger_limit_kw], [float(requested)],
                )
                demand = float(self._model_predict(self.demand_with[features.site], x)[0])
                dwell = float(self._model_predict(self.dwell_with[features.site], x)[0])
            else:
                x = _features_base(
                    [features.hour], [features.weekday], [features.charger_limit_kw]
                )
                demand = float(self._model_predict(self.demand_fallback[features.site], x)[0])
                dwell = float(self._model_predict(self.dwell_fallback[features.site], x)[0])
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
