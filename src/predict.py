"""负荷/需求预测器。

模型按可用性从强到弱选择：

1. LightGBM / XGBoost：上分主打，环境安装后自动启用；
2. scikit-learn HistGradientBoosting：无 LightGBM/XGBoost 时的梯度提升；
3. NumPy 岭回归：纯 numpy/pandas 环境下的可解释基线，正则系数用时间序
   验证集选择，在线预测时只用过去信息；
4. NaiveForecaster：分小时历史均值，作为最终兜底。

特征在 ``_build_features`` 中统一构造，训练与预测两侧一致；所有滞后与滚动
特征都由上游 ``features.add_lag_features`` 提前生成，这里只做安全派生。
"""
import numpy as np
import pandas as pd

from . import config as C

# 保留给老调用方的显式列名入口。默认走 _build_features 的工程特征。
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

_RIDGE_LAMBDAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)


def _get(df, col, default=np.nan):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _build_features(df):
    """从上游已生成的时间/滞后列派生特征，只使用过去信息。"""
    f = pd.DataFrame(index=df.index)
    f["hour"] = _get(df, "hour")
    f["minute_of_day"] = _get(df, "minute_of_day")
    f["weekday"] = _get(df, "weekday")
    f["is_weekend"] = _get(df, "is_weekend")
    f["hour_sin"] = _get(df, "hour_sin")
    f["hour_cos"] = _get(df, "hour_cos")

    lag1 = _get(df, "power_kw_lag1")
    lag4 = _get(df, "power_kw_lag4")
    lag24 = _get(df, "power_kw_lag24")
    lag96 = _get(df, "power_kw_lag96")
    f["lag1"] = lag1
    f["lag4"] = lag4
    f["lag24"] = lag24
    f["lag96"] = lag96
    f["roll24"] = _get(df, "power_kw_rolling24")

    occ1 = _get(df, "active_piles_lag1")
    occ4 = _get(df, "active_piles_lag4")
    occ24 = _get(df, "active_piles_lag24")
    f["occ1"] = occ1
    f["occ4"] = occ4
    f["occ24"] = occ24
    f["occ96"] = _get(df, "active_piles_lag96")
    f["occ_roll"] = _get(df, "active_piles_rolling24")

    f["delta14"] = lag1 - lag4
    f["delta124"] = lag1 - lag24
    f["delta496"] = lag4 - lag96
    f["ppp1"] = lag1 / (occ1 + 1.0)
    f["ppp4"] = lag4 / (occ4 + 1.0)
    f["ppp24"] = lag24 / (occ24 + 1.0)
    f["interact1"] = lag1 * occ1
    return f


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


class NumpyRidgeForecaster:
    """带时间序正则选择的标准化岭回归，纯 numpy 实现。"""

    def __init__(self, seed=0):
        self.seed = seed
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sd = None

    def _fit_ridge(self, x, y, lam):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        mu = x.mean(axis=0)
        sd = x.std(axis=0) + 1e-9
        z = (x - mu) / sd
        a = np.column_stack([z, np.ones(len(z))])
        ata = a.T @ a + lam * np.eye(a.shape[1])
        coef = np.linalg.solve(ata, a.T @ y)
        w = coef[:-1] / sd
        b = coef[-1] - float((w * mu).sum())
        return w, b

    def fit(self, X, y):
        x = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)
        if n < 20:
            self.w, self.b = self._fit_ridge(x, y, 1.0)
            self.mu = x.mean(axis=0)
            self.sd = x.std(axis=0) + 1e-9
            return self

        holdout = min(48, max(8, n // 6))
        best = None
        for lam in _RIDGE_LAMBDAS:
            w, b = self._fit_ridge(x[:-holdout], y[:-holdout], lam)
            err = float(
                np.abs(y[-holdout:] - (x[-holdout:] @ w + b)).mean()
            )
            if best is None or err < best[0]:
                best = (err, lam)
        self.w, self.b = self._fit_ridge(x, y, best[1])
        self.mu = x.mean(axis=0)
        self.sd = x.std(axis=0) + 1e-9
        return self

    def predict(self, X):
        x = np.asarray(X, dtype=float)
        return x @ self.w + self.b


def _try_lightgbm(seed):
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        return None
    return LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        verbose=-1,
    )


def _try_xgboost(seed):
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None
    return XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        verbosity=0,
    )


def _try_sklearn(seed):
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        try:
            from sklearn.ensemble import GradientBoostingRegressor
        except ImportError:
            return None
        return GradientBoostingRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=seed,
        )
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=seed,
    )


def _select_model(seed):
    model = _try_lightgbm(seed)
    if model is None:
        model = _try_xgboost(seed)
    if model is None:
        model = _try_sklearn(seed)
    if model is None:
        model = NumpyRidgeForecaster(seed=seed)
    return model


def _prepare_X(df, feature_cols):
    if feature_cols is not None:
        X = df[feature_cols]
    else:
        X = _build_features(df)
    return X.apply(pd.to_numeric, errors="coerce")


def _align_features(X, feature_names, fill_values):
    X = X.reindex(columns=feature_names)
    if fill_values is not None:
        X = X.fillna(fill_values)
    return np.asarray(X, dtype=float)


def train_predictor(train, val=None, target=C.LOAD_COL, feature_cols=None, seed=0):
    """训练预测器；传入 val 时同时返回验证集预测。"""
    X = _prepare_X(train, feature_cols)
    y = np.asarray(train[target], dtype=float).ravel()

    usable = [col for col in X.columns if not X[col].isna().all()]
    if not usable:
        model = NaiveForecaster()
        model.fit(train, y)
        if val is not None:
            return model, predict(model, val)
        return model

    X = X[usable]
    feature_names = list(X.columns)
    fill_values = X.median(numeric_only=True).fillna(0.0)
    X_array = _align_features(X, feature_names, fill_values)

    model = _select_model(seed)
    model.fit(X_array, y)

    model.feature_names_ = feature_names
    model.fill_values_ = fill_values
    model.feature_cols_ = feature_cols
    model.target_ = target

    if val is not None:
        pred = predict(model, val)
        return model, pred
    return model


def predict(model, frame, feature_cols=None):
    """用训练好的模型对 frame 做点预测，输出非负负荷（kW）。"""
    if isinstance(model, NaiveForecaster):
        return np.maximum(model.predict(frame), 0)

    names = getattr(model, "feature_names_", None)
    if names is None:
        feature_cols = feature_cols if feature_cols is not None else BASE_FEATURES
        return np.maximum(model.predict(frame[feature_cols]), 0)

    X = _prepare_X(frame, getattr(model, "feature_cols_", feature_cols))
    X_array = _align_features(X, names, getattr(model, "fill_values_", None))
    return np.maximum(model.predict(X_array), 0)
