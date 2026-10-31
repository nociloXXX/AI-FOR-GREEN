"""模块2：预测。

用 ExtraTrees 学习每辆车的需求电量与停留时长，并用星期x小时表估计到达率。
"""
from __future__ import annotations

import numpy as np

from acn_causal.student_api import Prediction
from module_1_data import build_weekday_hour_rates, make_features


class PredictionModule:
    """需求/停留回归器 + 星期x小时到达率表。"""

    def __init__(self, training):
        from sklearn.ensemble import ExtraTreesRegressor

        self.models = {}
        for site, frame in training.items():
            x = make_features(
                frame.hour.to_numpy(),
                frame.weekday.to_numpy(),
                frame.charger_limit_kw.to_numpy(),
            )
            y = np.column_stack([frame.demand_kwh, frame.dwell_hours])
            model = ExtraTreesRegressor(
                n_estimators=96,
                min_samples_leaf=15,
                max_depth=12,
                n_jobs=1,
                random_state=20261008,
            )
            model.fit(x, y)
            self.models[site] = model
        self.hour_rates = build_weekday_hour_rates(training)
        self._cache = {}

    def predict(self, features):
        key = (
            features.site,
            round(float(features.hour), 3),
            int(features.weekday),
            round(float(features.charger_limit_kw), 2),
        )
        if key not in self._cache:
            y = self.models[features.site].predict(
                make_features(
                    [features.hour], [features.weekday], [features.charger_limit_kw]
                )
            )[0]
            self._cache[key] = Prediction(
                float(np.clip(y[0], 0.01, 150.0)),
                float(np.clip(y[1], 0.25, 24.0)),
            )
        return self._cache[key]

    def arrival_rate(self, site, hour, weekday):
        return float(self.hour_rates[site][int(weekday) % 7, int(hour) % 24])
