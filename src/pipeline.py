"""端到端流程：数据 -> 预测 -> 世界模型 -> 干预推演 -> 充电决策。"""
import pandas as pd

from . import config as C
from .data_loader import clean_charging, load_raw_charging
from .decision import choose_dispatch
from .evaluate import decision_metrics, forecast_metrics
from .features import add_lag_features, add_time_features, build_station_load_curve
from .predict import train_predictor
from .state_model import StationState
from .submit import write_submission


def run():
    # 1. 数据处理：清洗会话，构建 15 分钟场站负荷曲线与时间特征。
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

        # 2. 预测：训练预测器并输出验证集负荷预测。
        model, pred = train_predictor(train, val)
        metrics = forecast_metrics(val[C.LOAD_COL].values, pred)

        # 3. 世界模型：实例化场站状态，供干预推演与决策读取。
        capacity = C.STATION_CAPACITY_KW.get(station, 200)
        state = StationState.build(
            station, C.STATION_PILES.get(station, 4), capacity_kw=capacity
        )

        # 4 + 5. 干预推演与充电决策：对候选干预推演评分后选择最优调度。
        dispatch, _unmet, plan_info = choose_dispatch(
            pred, capacity, C.STEP_MINUTES / 60, state=state
        )
        dec = decision_metrics(
            val[C.LOAD_COL].values, dispatch, capacity, C.STEP_MINUTES / 60
        )

        print(
            f"[{station}] MAE={metrics['mae']:.2f} kW | "
            f"RMSE={metrics['rmse']:.2f} kW | "
            f"peak={dec['peak_kw']:.1f} kW | "
            f"policy={plan_info.get('selected_strategy')}"
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
