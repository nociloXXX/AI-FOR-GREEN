"""模块2：预测。

预测一辆车到达时的需求电量与停留时长。demand/dwell 各用单目标 LightGBM
(L1/MAE)，按站点分开训练；`last_requested_energy_kwh` 用双分支：正有限值时
走含该特征的模型，缺失/非法时走不含该特征的回退模型。未来到达车辆该字段为
None，因此 forecast 推演自然走回退分支。

小样本站点 office1 额外用全站 pooled 模型做标定收缩：
``pred = alpha*site_model + (1-alpha)*pooled_model``，alpha 在 office1 时间序
holdout 上离线选出，避免 919 行小样本过拟合。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from acn_causal.student_api import Prediction


_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
_SITES = ("caltech", "jpl", "office1")
_SITE_INDEX = {"caltech": 0, "jpl": 1, "office1": 2}


def _features_base(hour, weekday, pmax):
    """与 module_1_data.make_features 对齐的 7 列基础特征。"""
    hour = np.asarray(hour, dtype=float) % 24.0
    weekday = np.asarray(weekday, dtype=float)
    pmax = np.asarray(pmax, dtype=float)
    return np.column_stack(
        [
            np.sin(hour * np.pi / 12.0),
            np.cos(hour * np.pi / 12.0),
            np.sin(hour * np.pi / 6.0),
            np.cos(hour * np.pi / 6.0),
            weekday,
            (weekday >= 5.0).astype(float),
            pmax,
        ]
    )


def _features_with_request(hour, weekday, pmax, requested):
    requested = np.asarray(requested, dtype=float).reshape(-1, 1)
    return np.column_stack([_features_base(hour, weekday, pmax), requested])


def _pooled_features(hour, weekday, pmax, site):
    base = _features_base(hour, weekday, pmax)
    onehot = np.zeros((len(base), len(_SITES)), dtype=float)
    onehot[:, _SITE_INDEX[site]] = 1.0
    return np.column_stack([base, onehot])


def _is_valid_request(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(value) and value > 0.0


def _fit_booster(x, y, num_leaves, sample_weight=None):
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
    model.fit(x, y, sample_weight=sample_weight)
    return model.booster_


def _fit_extra_trees(x, y):
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


def _load_alpha():
    path = _WEIGHTS_DIR / "office1_alpha.json"
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        demand = data["demand"]
        dwell = data["dwell"]
        return (
            float(demand["alpha"]),
            float(demand["gamma"]),
            float(dwell["alpha"]),
            float(dwell["gamma"]),
        )
    except Exception:
        return 0.5, 1.0, 0.5, 1.0


def _quantile_ratios(values, levels=(0.5, 0.75, 0.9)):
    """用站点训练标签的经验分位比，给单点预测加保守区间。"""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 20:
        base = 1.0
    else:
        base = float(np.quantile(values, 0.5))
    ratios = {}
    for level in levels:
        q = float(np.quantile(values, level)) if values.size >= 20 else base
        ratio = q / max(base, 1e-9)
        ratios[level] = float(np.clip(ratio, 1.0, 2.0))
    return ratios


def _build_arrival_rates(training):
    rates = {}
    for site in _SITES:
        frame = training[site]
        days = pd.date_range(frame.day.min(), frame.day.max(), freq="D")
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
    """需求/停留 LightGBM 双分支回归器 + pooled 收缩 + 到达率表。"""

    def __init__(self, training):
        self.demand_with = {}
        self.dwell_with = {}
        self.demand_fallback = {}
        self.dwell_fallback = {}
        self.with_available = {}

        for site in _SITES:
            frame = training[site]
            requested = (
                np.asarray(frame["last_requested_energy_kwh"], dtype=float)
                if "last_requested_energy_kwh" in frame.columns
                else np.full(len(frame), np.nan)
            )
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
            self.with_available[site] = bool(valid.any())

        self.demand_pooled = self._load_or_fit_pooled(
            _WEIGHTS_DIR / "demand_pooled.txt", training, "demand_kwh", 31
        )
        self.dwell_pooled = self._load_or_fit_pooled(
            _WEIGHTS_DIR / "dwell_pooled.txt", training, "dwell_hours", 15
        )
        (
            self.alpha_demand,
            self.gamma_demand,
            self.alpha_dwell,
            self.gamma_dwell,
        ) = _load_alpha()

        self.arrival_rates = _build_arrival_rates(training)
        self.demand_ratios = {}
        self.dwell_ratios = {}
        for site in _SITES:
            frame = training[site]
            self.demand_ratios[site] = _quantile_ratios(frame.demand_kwh.to_numpy())
            self.dwell_ratios[site] = _quantile_ratios(frame.dwell_hours.to_numpy())
        self._cache = {}

    def _load_or_fit_pooled(self, path, training, target, leaves):
        try:
            return _load_booster(path)
        except Exception:
            pass
        xs, ys = [], []
        for site in _SITES:
            frame = training[site]
            xs.append(_pooled_features(frame.hour, frame.weekday, frame.charger_limit_kw, site))
            ys.append(frame[target].to_numpy())
        return _fit_booster(np.vstack(xs), np.concatenate(ys), leaves)

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
            xb = _features_base(
                [features.hour], [features.weekday], [features.charger_limit_kw]
            )
            if use_request and self.with_available.get(features.site, False):
                x = _features_with_request(
                    [features.hour], [features.weekday],
                    [features.charger_limit_kw], [float(requested)],
                )
                demand = float(self._model_predict(self.demand_with[features.site], x)[0])
                dwell = float(self._model_predict(self.dwell_with[features.site], x)[0])
            else:
                demand = float(self._model_predict(self.demand_fallback[features.site], xb)[0])
                dwell = float(self._model_predict(self.dwell_fallback[features.site], xb)[0])
                if features.site == "office1":
                    xp = _pooled_features(
                        [features.hour], [features.weekday],
                        [features.charger_limit_kw], features.site,
                    )
                    pooled_demand = float(self._model_predict(self.demand_pooled, xp)[0])
                    pooled_dwell = float(self._model_predict(self.dwell_pooled, xp)[0])
                    demand = self.gamma_demand * (
                        self.alpha_demand * demand
                        + (1.0 - self.alpha_demand) * pooled_demand
                    )
                    dwell = self.gamma_dwell * (
                        self.alpha_dwell * dwell
                        + (1.0 - self.alpha_dwell) * pooled_dwell
                    )
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

    def predict_quantiles(self, features, levels=(0.5, 0.75, 0.9)):
        """返回 {level: Prediction} 的分位需求/停留估计。

        q50 复用现有单点预测，q75/q90 用站点训练标签的经验分位比放大，
        给 LLF/EDF 与未来到达注入提供保守 deadline 与需求区间。
        """
        point = self.predict(features)
        site = features.site
        out = {}
        for level in levels:
            demand_ratio = self.demand_ratios[site].get(level, 1.0)
            dwell_ratio = self.dwell_ratios[site].get(level, 1.0)
            out[level] = Prediction(
                float(np.clip(point.demand_kwh * demand_ratio, 0.01, 150.0)),
                float(np.clip(point.dwell_hours * dwell_ratio, 0.25, 24.0)),
            )
        return out
