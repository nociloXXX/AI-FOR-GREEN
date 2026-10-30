"""端到端流程：读取 -> 清洗 -> 负荷曲线 -> 特征 -> 预测 -> 决策 -> 提交。"""
import pandas as pd

from . import config as C
from .data_loader import clean_charging, load_raw_charging
from .decision import peak_shaving
from .evaluate import decision_metrics, forecast_metrics
from .features import add_lag_features, add_time_features, build_station_load_curve
from .predict import train_predictor
from .submit import write_submission


def run():
    sessions = clean_charging(load_raw_charging())
    curve = build_station_load_curve(sessions, C.STEP_MINUTES)
    curve = add_time_features(curve, "timestamp")

    rows = []
    artifact_rows = []
    for station, group in curve.groupby(C.STATION_COL):
        group = add_lag_features(group, [C.LOAD_COL, C.OCCUPANCY_COL])
        group = group.dropna(subset=["power_kw_lag96"]).reset_index(drop=True)

        split = int(len(group) * C.TRAIN_RATIO)
        if split < 20 or len(group) - split < 5:
            print(f"[{station}] 数据量不足，跳过")
            continue

        train, val = group.iloc[:split], group.iloc[split:]
        model, pred = train_predictor(train, val)
        metrics = forecast_metrics(val[C.LOAD_COL].values, pred)

        capacity = C.STATION_CAPACITY_KW.get(station, 200)
        dispatch, _unmet = peak_shaving(pred, capacity)
        dec = decision_metrics(
            val[C.LOAD_COL].values, dispatch, capacity, C.STEP_MINUTES / 60
        )

        print(
            f"[{station}] MAE={metrics['mae']:.2f} kW | "
            f"RMSE={metrics['rmse']:.2f} kW | "
            f"peak={dec['peak_kw']:.1f} kW"
        )

        for ts, actual, p, d in zip(
            val["timestamp"], val[C.LOAD_COL], pred, dispatch
        ):
            rows.append(
                {
                    "timestamp": ts,
                    C.STATION_COL: station,
                    "predicted_power_kw": round(float(p), 3),
                    "dispatch_power_kw": round(float(d), 3),
                }
            )
            artifact_rows.append(
                {
                    "timestamp": ts,
                    C.STATION_COL: station,
                    "actual_power_kw": round(float(actual), 3),
                    "predicted_power_kw": round(float(p), 3),
                    "dispatch_power_kw": round(float(d), 3),
                }
            )

    if not rows:
        raise SystemExit("没有生成提交行，检查数据量是否足够")

    submission = pd.DataFrame(rows).sort_values(["timestamp", C.STATION_COL])
    path = write_submission(submission)
    print("提交文件:", path)

    artifacts = pd.DataFrame(artifact_rows).sort_values(["timestamp", C.STATION_COL])
    C.ARTIFACTS_FILE.parent.mkdir(exist_ok=True)
    artifacts.to_csv(C.ARTIFACTS_FILE, index=False)
    print("运行产物:", C.ARTIFACTS_FILE)
    return submission


if __name__ == "__main__":
    run()
