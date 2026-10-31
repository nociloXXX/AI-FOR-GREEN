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

from acn_causal.student_api import ArrivalFeatures, Prediction


_SITES = ("caltech", "jpl", "office1")
_QUANTILE_LEVELS = (0.5, 0.75, 0.9)
_DWELL_QUANTILE_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)
_HIST_SEEDS = (20261008, 20261009)
_HALF_LIFE = {"caltech": 90.0, "jpl": 15.0, "office1": 180.0}
_CHARGER_KW = 6.6
_SURVIVAL_BINS = (0.0, 1.5, 3.0, 6.0, 9.0, 12.0, 18.0, 24.0)


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


def _build_joint_cohorts(frame):
    """按停留时长排序的经验联合分位 cohort：保留 demand 与 dwell 的相关性。"""
    demand = np.asarray(frame["demand_kwh"], dtype=float)
    dwell = np.asarray(frame["dwell_hours"], dtype=float)
    valid = np.isfinite(demand) & (demand > 0) & np.isfinite(dwell) & (dwell > 0)
    demand, dwell = demand[valid], dwell[valid]
    if len(demand) == 0:
        return ((10.0, 4.0, 1.0),)
    order = np.argsort(dwell)
    demand, dwell = demand[order], dwell[order]
    buckets = np.array_split(np.arange(len(demand)), 5)
    cohorts = []
    for idx in buckets:
        if len(idx) == 0:
            continue
        d = float(np.median(demand[idx]))
        w = float(np.median(dwell[idx]))
        d = min(d, _CHARGER_KW * w)
        cohorts.append((d, w, 1.0 / 5.0))
    if not cohorts:
        return ((10.0, 4.0, 1.0),)
    return tuple(cohorts)


def _build_survival_residuals(frame, predicted_dwell):
    """离线学习“已停 lower 小时后，剩余停留比预测多/少多少”。"""
    actual = np.asarray(frame["dwell_hours"], dtype=float)
    predicted = np.asarray(predicted_dwell, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    residuals = {}
    for lower in _SURVIVAL_BINS:
        survived = actual > lower + 1e-8
        if survived.sum() < 12:
            continue
        actual_remaining = np.maximum(actual[survived] - lower, 0.25)
        predicted_remaining = np.maximum(predicted[survived] - lower, 0.25)
        residuals[lower] = float(
            np.clip(
                np.median(actual_remaining - predicted_remaining),
                -8.0,
                8.0,
            )
        )
    return residuals


def _build_occupancy_regressions(frame):
    """离线学习“当前在场车数 -> 未来 12 步到达数”的粗回归。"""
    import math as _math

    timestamps = pd.to_datetime(frame["day"], errors="coerce")
    daily = []
    for day, rows in frame.groupby(timestamps.dt.date, sort=True):
        if pd.isna(day):
            continue
        active = np.zeros(96, dtype=float)
        arrivals = np.zeros(96, dtype=float)
        for record in rows.itertuples(index=False):
            hour = float(record.hour)
            dwell = float(record.dwell_hours)
            if not _math.isfinite(hour) or not _math.isfinite(dwell):
                continue
            arrival = min(max(int(hour * 4.0), 0), 95)
            capped = min(max(dwell, 0.25), (96 - arrival) * 0.25)
            departure = min(96, arrival + max(1, int(_math.ceil(capped / 0.25))))
            active[arrival:departure] += 1.0
            arrivals[arrival] += 1.0
        daily.append((active, arrivals))
    if not daily:
        return {}

    regressions = {}
    for origin in range(0, 96, 12):
        current = np.asarray([entry[0][origin] for entry in daily])
        future = np.asarray(
            [entry[1][origin + 1 : origin + 12].sum() for entry in daily]
        )
        variance = float(np.var(current))
        slope = (
            float(np.cov(current, future, bias=True)[0, 1] / variance)
            if variance > 1e-8
            else 0.0
        )
        regressions[origin] = (
            max(0.0, float(np.mean(future) - max(0.0, slope) * np.mean(current))),
            max(0.0, slope),
        )
    return regressions


class PredictionModule:
    """车辆级需求/停留预测。"""

    DEMAND_KNN_WEIGHT = 0.3
    DWELL_KNN_WEIGHT = 0.5

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
                    xb, frame.demand_kwh.to_numpy(), 0.5, True, None, 0
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
        self.future_cohorts_table = {}
        self.cohort_models = {}
        self.survival_residuals = {}
        self.occupancy_regressions = {}
        self.weekday_days = {}
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
            if site == "office1":
                x = _features_base(
                    frame.hour, frame.weekday, frame.charger_limit_kw
                )
            else:
                x = _features_full(
                    frame.hour, frame.weekday, frame.charger_limit_kw,
                    request_filled, request_missing,
                )
            predicted_dwell = self._model_predict(self.dwell_model[site], x)
            self.future_cohorts_table[site] = _build_joint_cohorts(frame)
            self.survival_residuals[site] = _build_survival_residuals(
                frame, predicted_dwell
            )
            self.occupancy_regressions[site] = _build_occupancy_regressions(frame)
            days = pd.date_range(frame.day.min(), frame.day.max(), freq="D")
            self.weekday_days[site] = np.bincount(
                days.dayofweek, minlength=7
            ).astype(float)
            from sklearn.ensemble import ExtraTreesRegressor

            if site == "office1":
                cohort_x = _features_base(
                    frame.hour, frame.weekday, frame.charger_limit_kw
                )
            else:
                cohort_x = _features_full(
                    frame.hour, frame.weekday, frame.charger_limit_kw,
                    request_filled, request_missing,
                )
            cohort_model = ExtraTreesRegressor(
                n_estimators=192,
                min_samples_leaf=8,
                max_depth=16,
                max_features=0.8,
                n_jobs=1,
                random_state=20261008,
            )
            cohort_model.fit(
                cohort_x,
                np.column_stack(
                    [frame.demand_kwh.to_numpy(), frame.dwell_hours.to_numpy()]
                ),
            )
            self.cohort_models[site] = cohort_model
        self._cache = {}
        self._cohort_cache = {}

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
            if features.site == "office1":
                kd, kw = self._knn_estimate(features)
                demand = (1.0 - self.DEMAND_KNN_WEIGHT) * demand + self.DEMAND_KNN_WEIGHT * kd
                dwell = (1.0 - self.DWELL_KNN_WEIGHT) * dwell + self.DWELL_KNN_WEIGHT * kw
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

    def future_cohorts(self, site, hour, weekday):
        key = (site, round(float(hour) % 24.0, 4), int(weekday) % 7)
        cached = self._cohort_cache.get(key)
        if cached is not None:
            return cached
        model = self.cohort_models.get(site)
        if model is None:
            value = self.future_cohorts_table.get(site, ((10.0, 4.0, 1.0),))
            self._cohort_cache[key] = value
            return value

        hour = float(hour) % 24.0
        weekday = int(weekday) % 7
        if site == "office1":
            x = _features_base([hour], [weekday], [6.6])
        else:
            x = _features_full(
                [hour], [weekday], [6.6],
                [self.request_medians[site]], [1.0],
            )
        point = self.predict(
            ArrivalFeatures("cohort", site, hour, weekday, 6.6, None)
        )
        tree_pred = np.asarray(
            [tree.predict(x)[0] for tree in model.estimators_], dtype=float
        )
        if len(tree_pred) == 0:
            value = ((point.demand_kwh, point.dwell_hours, 1.0),)
            self._cohort_cache[key] = value
            return value

        tree_pred = tree_pred[np.argsort(tree_pred[:, 1])]
        picks = [int(round(q * (len(tree_pred) - 1))) for q in (0.1, 0.3, 0.5, 0.7, 0.9)]
        selected = tree_pred[picks].copy()
        demand_scale = point.demand_kwh / max(float(np.mean(selected[:, 0])), 1e-8)
        dwell_scale = point.dwell_hours / max(float(np.mean(selected[:, 1])), 1e-8)
        selected[:, 0] *= demand_scale
        selected[:, 1] *= dwell_scale
        selected[:, 1] = np.clip(selected[:, 1], 0.25, 24.0)
        selected[:, 0] = np.minimum(
            np.clip(selected[:, 0], 0.01, 150.0), _CHARGER_KW * selected[:, 1]
        )
        value = tuple(
            (float(d), float(w), 0.2) for d, w in selected
        )
        self._cohort_cache[key] = value
        return value

    def survival_remaining_hours(
        self, site, predicted_dwell_hours, elapsed_hours, dt_hours
    ):
        predicted = max(float(predicted_dwell_hours), float(dt_hours))
        elapsed = max(0.0, float(elapsed_hours))
        base = max(float(dt_hours), predicted - elapsed)
        residuals = self.survival_residuals.get(site)
        if not residuals or elapsed + 1e-8 < predicted:
            return base
        eligible = [lower for lower in residuals if lower <= elapsed + 1e-8]
        if not eligible:
            return base
        return max(float(dt_hours), base + 0.5 * residuals[max(eligible)])

    def arrival_multipliers(self, observation, steps):
        multipliers = np.ones(max(0, int(steps)), dtype=float)
        history = np.asarray(observation.historical_arrivals, dtype=float)
        if multipliers.size == 0 or len(history) == 0:
            return multipliers
        if not np.isfinite(history).all() or (history < 0).any():
            return multipliers
        site_days = self.weekday_days.get(
            observation.site, np.ones(7, dtype=float)
        )
        window = min(multipliers.size, len(history))
        for offset in range(window):
            absolute_hour = (
                float(observation.local_hour)
                + offset * float(observation.dt_hours)
            )
            hour = absolute_hour % 24.0
            weekday = (
                int(observation.weekday) + int(absolute_hour // 24.0)
            ) % 7
            days = max(float(site_days[weekday]), 1.0)
            expected = (
                self.arrival_rate(observation.site, hour, weekday) * days
            )
            ratio = float(history[offset]) / max(expected, 0.25)
            multipliers[offset] = float(
                np.clip(1.0 + 0.3 * (np.clip(ratio, 0.5, 1.5) - 1.0), 0.85, 1.15)
            )
        return multipliers

    def occupancy_multiplier(self, observation, steps):
        if int(steps) < 2:
            return 1.0
        regressions = self.occupancy_regressions.get(observation.site, {})
        origin = int(observation.step // 12 * 12)
        regression = regressions.get(origin)
        if regression is None:
            return 1.0
        intercept, slope = regression
        predicted_future = max(0.0, intercept + slope * len(observation.vehicles))
        base_future = 0.0
        for offset in range(1, min(int(steps), 12)):
            absolute_hour = (
                float(observation.local_hour)
                + offset * float(observation.dt_hours)
            )
            hour = absolute_hour % 24.0
            weekday = (
                int(observation.weekday) + int(absolute_hour // 24.0)
            ) % 7
            base_future += self.arrival_rate(observation.site, hour, weekday)
        if base_future <= 1e-8:
            return 1.0
        return float(np.clip(predicted_future / base_future, 0.5, 2.0))
