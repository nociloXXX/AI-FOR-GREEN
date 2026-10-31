"""官方口径运行效果查看器：学习指标 + 五接口冒烟。

只读不写算法文件。数据预处理（合并三站训练表）与公开自评逻辑
与 public_self_eval.py 保持一致，输出便于读数的中文摘要。
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent


def merge_training():
    target = ROOT / "data" / "training_public.json"
    if target.exists():
        return target
    out = {}
    for site in ("caltech", "jpl", "office1"):
        src = ROOT / "data" / f"training_{site}.json"
        out[site] = json.loads(src.read_text(encoding="utf-8"))
    target.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return target


def split_train_holdout(training_path):
    source = json.loads(training_path.read_text(encoding="utf-8"))
    train = {}
    held = []
    for site, rows in source.items():
        frame = pd.DataFrame(rows)
        frame["day"] = pd.to_datetime(frame.day)
        days = sorted(frame.day.unique())
        cut = days[max(1, int(len(days) * 0.8)) - 1]
        train[site] = frame[frame.day <= cut].copy()
        held.extend((site, r) for r in frame[frame.day > cut].to_dict("records"))
    return train, held


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    from acn_causal.student_api import ArrivalFeatures, Observation, VehicleObservation
    from submission import build_model

    training_path = merge_training()
    train, held = split_train_holdout(training_path)
    model = build_model(train)

    errs = {
        site: {"demand_kwh": [], "dwell_hours": []}
        for site in ("caltech", "jpl", "office1")
    }
    finite = True
    for i, (site, record) in enumerate(held):
        requested = record.get("last_requested_energy_kwh")
        requested = float(requested) if requested is not None and pd.notna(requested) else None
        features = ArrivalFeatures(
            "held-" + str(i),
            site,
            float(record["hour"]),
            int(record["weekday"]),
            float(record["charger_limit_kw"]),
            requested,
        )
        pred = model.predict(features)
        for field in ("demand_kwh", "dwell_hours"):
            value = float(getattr(pred, field))
            if not math.isfinite(value) or value < 0:
                finite = False
                continue
            errs[site][field].append(abs(value - float(record[field])))

    print("官方口径运行效果")
    print("-" * 46)
    print("训练行数:", sum(len(f) for f in train.values()))
    print("验证行数:", len(held))
    print("预测全部有限且非负:", "是" if finite else "否")
    for site in ("caltech", "jpl", "office1"):
        d = errs[site]["demand_kwh"]
        w = errs[site]["dwell_hours"]
        demand = sum(d) / len(d) if d else None
        dwell = sum(w) / len(w) if w else None
        fmt = lambda v: f"{v:.4f}" if v is not None else "无"
        print(f"{site:8s} 需求MAE={fmt(demand)} kWh  停留MAE={fmt(dwell)} h  n={len(d)}")

    features = ArrivalFeatures("smoke", "caltech", 9.4, 0, 6.6, None)
    vehicle = VehicleObservation(features, 0, 0, 0.0, False)
    observation = Observation(
        step=0,
        dt_hours=0.25,
        horizon_steps=4,
        site="caltech",
        local_hour=9.0,
        weekday=0,
        vehicles=(vehicle,),
        capacity_kw=100.0,
        background_kw=10.0,
        prices=(0.6,) * 4,
        forecast_capacity=(100.0,) * 4,
        forecast_background=(10.0,) * 4,
        historical_arrivals=(0,) * 4,
        intervention_capacity=(60.0,) * 4,
    )
    natural = model.forecast(observation, 4, "natural")
    edf = model.forecast(observation, 4, "capacity_edf")
    print("-" * 46)
    print("act:", model.act(observation))
    print("natural ev_load_kw:", [round(v, 2) for v in natural["ev_load_kw"]])
    print("capacity_edf ev_load_kw:", [round(v, 2) for v in edf["ev_load_kw"]])
    print("active_count:", natural["active_count"])


if __name__ == "__main__":
    main()
