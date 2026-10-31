"""本地冒烟验证：加载三站训练表，跑通提交模型的五个接口方法。"""
import json

import pandas as pd

from acn_causal.student_api import ArrivalFeatures, Observation, VehicleObservation
from submission import build_model


def load_training():
    training = {}
    for site in ("caltech", "jpl", "office1"):
        rows = json.load(open(f"data/training_{site}.json", encoding="utf-8"))
        frame = pd.DataFrame(rows)
        frame["day"] = pd.to_datetime(frame["day"])
        training[site] = frame
    return training


def main():
    model = build_model(load_training())

    features = ArrivalFeatures("t1", "caltech", 9.4, 0, 6.6, None)
    print("predict:", model.predict(features))
    print("arrival_rate:", model.arrival_rate("caltech", 9, 0))

    vehicle = VehicleObservation(features, 0, 0, 0.0, False)
    observation = Observation(
        step=0,
        dt_hours=0.25,
        horizon_steps=96,
        site="caltech",
        local_hour=9.0,
        weekday=0,
        vehicles=(vehicle,),
        capacity_kw=100.0,
        background_kw=10.0,
        prices=(0.6,) * 96,
        forecast_capacity=(100.0,) * 96,
        forecast_background=(10.0,) * 96,
        historical_arrivals=(0,) * 96,
        intervention_capacity=(60.0,) * 96,
    )
    print("act:", model.act(observation))
    print("natural:", model.forecast(observation, 4, "natural"))
    print("capacity_edf:", model.forecast(observation, 4, "capacity_edf"))
    print("explain:", model.explain())


if __name__ == "__main__":
    main()
