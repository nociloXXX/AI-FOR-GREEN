"""模块2：预测。

demand/dwell 各用单目标模型，按站点分开训练。特征为时间周期编码、桩上限，以及
``last_requested_energy_kwh`` 的中位数填充值 + 缺失指示（缺失不当 0，也不当
地面真值）。caltech/jpl 用 LightGBM L1，office1 小样本用 HistGradientBoosting
quantile q50 双种子平均，并叠加余弦近邻轻集成。模型只读取公开历史，不读取
未来到达、真实离开或隐藏需求标签。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from acn_causal.student_api import Prediction


_SITES = ("caltech", "jpl", "office1")
_QUANTILE_LEVELS = (0.5, 0.75, 0.9)
_DWELL_QUANTILE_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)
_HIST_SEEDS = (20261008, 20261009)
_HALF_LIFE = {"caltech": 90.0, "jpl": 15.0, "office1": 180.0}


def _features_base(hour, weekday, pmax):
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


def _features_full(hour, weekday, pmax, request_filled, request_missing):
    return np.column_stack(
        [
            _features_base(hour, weekday, pmax),
            np.asarray(request_filled, dtype=float).reshape(-1, 1),
            np.asarray(request_missing, dtype=float).reshape(-1, 1),
        ]
    )


def _is_valid_request(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(value) and value > 0.0


def _weighted_median(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    total = cdf[-1]
    if total <= 1e-12:
        return float(values[len(values) // 2])
    return float(values[np.searchsorted(cdf, 0.5 * total)])


def _sample_weights(days, half_life):
    days = pd.to_datetime(days)
    age_days = (days.max() - days).dt.total_seconds() / 86400.0
    return np.exp(-np.log(2.0) * age_days / max(float(half_life), 1e-9))


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


class _LogTargetModel:
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        return np.expm1(self.model.predict(data))


class _AveragedModel:
    def __init__(self, models):
        self.models = models

    def predict(self, data):
        preds = [m.predict(data) for m in self.models]
        return np.mean(np.stack(preds, axis=0), axis=0)


def _fit_hist_quantile(x, y, q=0.5, log_target=False, sample_weight=None, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=q,
        max_iter=200,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=seed,
    )
    target = np.log1p(np.asarray(y, dtype=float)) if log_target else np.asarray(y, dtype=float)
    model.fit(x, target, sample_weight=sample_weight)
    return _LogTargetModel(model) if log_target else model


def _fit_hist_quantile_avg(x, y, q=0.5, log_target=False, sample_weight=None):
    return _AveragedModel(
        [
            _fit_hist_quantile(x, y, q, log_target, sample_weight, seed)
            for seed in _HIST_SEEDS
        ]
    )


def _request_medians(training):
    pooled = []
    medians = {}
    for site in _SITES:
        frame = training[site]
        requested = (
            np.asarray(frame["last_requested_energy_kwh"], dtype=float)
            if "last_requested_energy_kwh" in frame.columns
            else np.full(len(frame), np.nan)
        )
        valid = np.isfinite(requested) & (requested > 0.0)
        if valid.any():
            pooled.append(float(np.median(requested[valid])))
            medians[site] = float(np.median(requested[valid]))
        else:
            medians[site] = np.nan
    global_median = float(np.median(pooled)) if pooled else 0.0
    for site in _SITES:
        if not np.isfinite(medians[site]):
            medians[site] = global_median
    return medians


def _build_arrival_rates(training):
    rates = {}
    for site in _SITES:
        frame = training[site]
        days = pd.date_range(frame.day.min(), frame.day.max(), freq="D")
        weekday_days = np.bincount(days.dayofweek, minlength=7).astype(float)
        total_days = max(float(len(days)), 1.0)

        slots = np.clip(np.floor(frame.hour.to_numpy() * 4.0).astype(int) % 96, 0, 95)
        counts = np.zeros((7, 96))
        weekdays = np.asarray(frame.weekday, dtype=int) % 7
        np.add.at(counts, (weekdays, slots), 1.0)

        observed = weekday_days[:, None]
        mean_rate = len(frame) / (total_days * 96.0)
        pseudo_days = 3.0
        rate = (counts + mean_rate * pseudo_days) / (observed + pseudo_days)

        prev = np.roll(rate, 1, axis=1)
        nxt = np.roll(rate, -1, axis=1)
        rates[site] = (0.25 * prev + 0.5 * rate + 0.25 * nxt, 96)
    return rates


class FutureSampler:
    """未来到达 demand/dwell 的条件密度采样器（经验分位插值或对角 GMM）。"""

    def __init__(self, emp_demand_q, emp_dwell_q, gmms=None):
        self.emp_demand_q = emp_demand_q
        self.emp_dwell_q = emp_dwell_q
        self.gmms = gmms or {}

    def __call__(self, features, rng):
        site = features.site
        gmm = self.gmms.get(site)
        if gmm is not None:
            try:
                state = np.random.RandomState(rng.getrandbits(32))
                sample, _ = gmm.sample(random_state=state)
                return Prediction(
                    float(np.clip(np.expm1(sample[0, 0]), 0.01, 150.0)),
                    float(np.clip(np.expm1(sample[0, 1]), 0.25, 24.0)),
                )
            except Exception:
                pass

        demand_q = self.emp_demand_q[site]
        dwell_q = self.emp_dwell_q[site]
        u = rng.random()
        if u < 0.5:
            x = u * 2.0
            d = demand_q[0.5] + (demand_q[0.75] - demand_q[0.5]) * x
            w = dwell_q[0.5] + (dwell_q[0.75] - dwell_q[0.5]) * x
        else:
            x = (u - 0.5) * 2.0
            d = demand_q[0.75] + (demand_q[0.9] - demand_q[0.75]) * x
            w = dwell_q[0.75] + (dwell_q[0.9] - dwell_q[0.75]) * x
        return Prediction(
            float(np.clip(d, 0.01, 150.0)),
            float(np.clip(w, 0.25, 24.0)),
        )


def _build_future_sampler(training):
    emp_demand_q = {}
    emp_dwell_q = {}
    gmms = {}
    for site in _SITES:
        frame = training[site]
        values = frame[["demand_kwh", "dwell_hours"]].to_numpy(dtype=float)
        values = values[np.isfinite(values).all(axis=1)]
        if values.size == 0:
            continue
        emp_demand_q[site] = {q: float(np.quantile(values[:, 0], q)) for q in _QUANTILE_LEVELS}
        emp_dwell_q[site] = {q: float(np.quantile(values[:, 1], q)) for q in _QUANTILE_LEVELS}
        if len(values) >= 400:
            try:
                from sklearn.mixture import GaussianMixture

                gmm = GaussianMixture(
                    n_components=3 if len(values) >= 900 else 2,
                    covariance_type="diag",
                    reg_covar=1e-3,
                    random_state=20261008,
                )
                gmm.fit(np.log1p(values))
                gmms[site] = gmm
            except Exception:
                gmms[site] = None
    return FutureSampler(emp_demand_q, emp_dwell_q, gmms)


class PredictionModule:
    """车辆级需求/停留预测。"""

    KNN_WEIGHT = 0.5

    def __init__(self, training):
        self.request_medians = _request_medians(training)
        self.demand_model = {}
        self.dwell_model = {}
        self._office_knn = None

        for site in _SITES:
            frame = training[site]
            requested = (
                np.asarray(frame["last_requested_energy_kwh"], dtype=float)
                if "last_requested_energy_kwh" in frame.columns
                else np.full(len(frame), np.nan)
            )
            valid = np.isfinite(requested) & (requested > 0.0)
            request_filled = np.where(valid, requested, self.request_medians[site])
            request_missing = (~valid).astype(float)
            x = _features_full(
                frame.hour, frame.weekday, frame.charger_limit_kw,
                request_filled, request_missing,
            )
            weights = _sample_weights(frame.day, _HALF_LIFE[site])

            if site == "office1":
                xb = _features_base(
                    frame.hour, frame.weekday, frame.charger_limit_kw
                )
                self.demand_model[site] = _fit_hist_quantile(
                    xb, frame.demand_kwh.to_numpy(), 0.5, False, None, 0
                )
                self.dwell_model[site] = _fit_hist_quantile(
                    xb, frame.dwell_hours.to_numpy(), 0.5, True, None, 0
                )
                norms = np.linalg.norm(xb, axis=1, keepdims=True) + 1e-9
                self._office_knn = {
                    "X": xb / norms,
                    "demand": frame.demand_kwh.to_numpy(),
                    "dwell": frame.dwell_hours.to_numpy(),
                }
            else:
                self.demand_model[site] = _fit_booster(
                    x, frame.demand_kwh.to_numpy(), 31, weights
                )
                self.dwell_model[site] = _fit_booster(
                    x, frame.dwell_hours.to_numpy(), 15, weights
                )

        self.arrival_rates = _build_arrival_rates(training)
        self.demand_quantiles = {}
        self.dwell_quantiles = {}
        for site in _SITES:
            frame = training[site]
            xb = _features_base(frame.hour, frame.weekday, frame.charger_limit_kw)
            weights = _sample_weights(frame.day, _HALF_LIFE[site])
            self.demand_quantiles[site] = {
                q: _fit_hist_quantile(xb, frame.demand_kwh.to_numpy(), q, False, weights, 20261008)
                for q in _QUANTILE_LEVELS
            }
            self.dwell_quantiles[site] = {
                q: _fit_hist_quantile(xb, frame.dwell_hours.to_numpy(), q, True, weights, 20261008)
                for q in _DWELL_QUANTILE_LEVELS
            }
        self.future_sampler = _build_future_sampler(training)
        self._cache = {}

    @staticmethod
    def _model_predict(model, x):
        return np.asarray(model.predict(x), dtype=float).reshape(-1)

    def _knn_estimate(self, features, k=40):
        x = _features_base(
            [features.hour], [features.weekday], [features.charger_limit_kw]
        )
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
        sim = self._office_knn["X"] @ x[0]
        idx = np.argsort(-sim)[:k]
        weights = np.maximum(sim[idx], 0.0)
        if weights.sum() <= 1e-12:
            weights = np.ones(k)
        return (
            _weighted_median(self._office_knn["demand"][idx], weights),
            _weighted_median(self._office_knn["dwell"][idx], weights),
        )

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
            if features.site == "office1":
                x = _features_base(
                    [features.hour], [features.weekday], [features.charger_limit_kw]
                )
            else:
                request_filled = (
                    float(requested) if use_request else self.request_medians[features.site]
                )
                request_missing = 0.0 if use_request else 1.0
                x = _features_full(
                    [features.hour], [features.weekday], [features.charger_limit_kw],
                    [request_filled], [request_missing],
                )
            demand = float(self._model_predict(self.demand_model[features.site], x)[0])
            dwell = float(self._model_predict(self.dwell_model[features.site], x)[0])
            if features.site == "office1" and self.KNN_WEIGHT > 0.0:
                kd, kw = self._knn_estimate(features)
                demand = (1.0 - self.KNN_WEIGHT) * demand + self.KNN_WEIGHT * kd
                dwell = (1.0 - self.KNN_WEIGHT) * dwell + self.KNN_WEIGHT * kw
            if features.site != "office1":
                demand = min(demand, float(features.charger_limit_kw) * dwell)
            self._cache[key] = Prediction(
                float(np.clip(demand, 0.01, 150.0)),
                float(np.clip(dwell, 0.25, 24.0)),
            )
        return self._cache[key]

    def predict_quantiles(self, features, levels=(0.5, 0.75, 0.9)):
        x = _features_base(
            [features.hour], [features.weekday], [features.charger_limit_kw]
        )
        demand_models = self.demand_quantiles[features.site]
        dwell_models = self.dwell_quantiles[features.site]
        out = {}
        for q in levels:
            demand = float(self._model_predict(
                demand_models.get(float(q)) or demand_models[0.5], x
            )[0])
            dwell = float(self._model_predict(
                dwell_models.get(float(q)) or dwell_models[0.5], x
            )[0])
            demand = min(demand, float(features.charger_limit_kw) * dwell)
            out[q] = Prediction(
                float(np.clip(demand, 0.01, 150.0)),
                float(np.clip(dwell, 0.25, 24.0)),
            )
        return out

    def arrival_rate(self, site, hour, weekday):
        table, nbins = self.arrival_rates[site]
        hour = float(hour) % 24.0
        weekday = int(weekday) % 7
        bin_idx = int(np.floor(hour * 4.0)) % nbins
        return float(table[weekday, bin_idx])
