"""端到端流程：数据 -> 预测 -> 世界模型 -> 干预推演 -> 充电决策。"""
import pandas as pd

from . import config as C
from .data_loader import clean_charging, load_raw_charging
from .decision import choose_dispatch
from .evaluate import decision_metrics, forecast_metrics
from .features import add_lag_features, add_time_features, build_station_load_curve
from .predict import train_predictor
from .state_model import StationState, VehicleSession
from .submit import write_submission


def _seed_station_state(station, station_sessions, val_start, capacity_kw):
    """用验证期开始时仍在场的真实会话灌入世界模型。

    按会话已过去的时间比例估算剩余电量，逐个占桩，超出的排入队列。这样
    后续 choose_dispatch 的前向推演才会计入“本已存在的未满足电量”。
    """
    num_piles = C.STATION_PILES.get(station, 4)
    state = StationState.build(
        station, num_piles, capacity_kw=capacity_kw, time=val_start
    )
    active = station_sessions[
        (station_sessions[C.TIME_START] <= val_start)
        & (station_sessions[C.TIME_END] > val_start)
    ]
    for row in active.itertuples():
        start = getattr(row, C.TIME_START)
        end = getattr(row, C.TIME_END)
        energy = float(getattr(row, C.ENERGY_COL))
        elapsed = (val_start - start).total_seconds() / 3600.0
        duration = max((end - start).total_seconds() / 3600.0, 1e-6)
        remaining = energy * max(1.0 - elapsed / duration, 0.0)
        if remaining <= 1e-9:
            continue
        session = VehicleSession(
            vehicle_id=str(getattr(row, C.PILE_COL)),
            energy_kwh=remaining,
            max_power_kw=float(getattr(row, C.MAX_POWER_COL)),
            arrival_time=start,
        )
        free_pile = next((pile for pile in state.piles if not pile.busy), None)
        if free_pile is not None:
            free_pile.attach(session)
        else:
            state.queue.append(session)
    return state


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

        # 3. 世界模型：用验证期开始时的在充会话灌入状态，再交干预推演。
        capacity = C.STATION_CAPACITY_KW.get(station, 200)
        station_sessions = sessions[sessions[C.STATION_COL] == station]
        state = _seed_station_state(
            station,
            station_sessions,
            val["timestamp"].iloc[0],
            capacity,
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
            f"MAPE={metrics['mape']:.1f}% | "
            f"peak={dec['peak_after_kw']:.1f} kW | "
            f"violations={dec['after_violations']} | "
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
