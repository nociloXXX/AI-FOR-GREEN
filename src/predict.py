"""负荷/需求预测器。先用梯度提升树打底，上分再换时序模型。

sklearn 不可用时自动回退到“分小时历史均值”，保证没有完整依赖也能跑通链路。
"""
import numpy as np
import pandas as pd

from . import config as C

try:
    from sklearn.ensemble import GradientBoostingRegressor

    HAS_SKLEARN = True
except ImportError:  # pragma: no cover - 环境缺依赖时走朴素基线
    HAS_SKLEARN = False

BASE_FEATURES = [
    "hour",
    "minute_of_day",
    "weekday",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "power_kw_lag1",
    "power_kw_lag4",
    "power_kw_lag24",
    "power_kw_lag96",
    "power_kw_rolling24",
    "active_piles_lag1",
    "active_piles_lag4",
    "active_piles_lag24",
]


class NaiveForecaster:
    """按小时取历史均值，作为无 sklearn 时的可运行兜底。"""

    def __init__(self):
        self.by_hour = None

    def fit(self, X, y):
        frame = pd.DataFrame({"hour": X["hour"], "target": np.asarray(y)})
        self.by_hour = frame.groupby("hour")["target"].mean()
        return self

    def predict(self, X):
        fallback = float(self.by_hour.mean())
        values = X["hour"].map(lambda hour: self.by_hour.get(hour, fallback))
        return np.asarray(values, dtype=float)


def train_predictor(train, val=None, target=C.LOAD_COL, feature_cols=None, seed=0):
    feature_cols = feature_cols or BASE_FEATURES
    if HAS_SKLEARN:
        model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            random_state=seed,
        )
    else:
        model = NaiveForecaster()
    model.fit(train[feature_cols], train[target])
    if val is not None:
        return model, np.maximum(model.predict(val[feature_cols]), 0)
    return model


def predict(model, frame, feature_cols=None):
    feature_cols = feature_cols or BASE_FEATURES
    return np.maximum(model.predict(frame[feature_cols]), 0)
