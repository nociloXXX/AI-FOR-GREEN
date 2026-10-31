import json
from collections import defaultdict

import numpy as np
import pandas as pd

from acn_causal.student_api import Prediction
from completion_proxy import (
    DT,
    STEPS,
    _apply_condition,
    _background_curve,
    _build_day_vehicles,
    _peak_concurrent_kw,
    _simulate,
    load_training,
)
from submission import build_model


def main():
    training = load_training()
    frames = {}
    for site in ("caltech", "jpl", "office1"):
        frame = pd.DataFrame(training[site])
        frame["day"] = pd.to_datetime(frame["day"])
        frames[site] = frame
    model = build_model(frames)

    office = pd.DataFrame(training["office1"])
    office["day"] = pd.to_datetime(office["day"])
    rows = office[office["day"] == "2019-10-18"].to_dict("records")
    day_vehicles = _build_day_vehicles(rows)
    vehicles = _apply_condition(day_vehicles, "arrival_burst")
    base_peak = _peak_concurrent_kw(day_vehicles)
    background = _background_curve(base_peak)

    for mult in (0.8, 1.0, 1.2, 1.5, 2.0):
        def predict_fn(features, m=mult):
            pred = model.predict(features)
            return Prediction(pred.demand_kwh * m, max(0.25, pred.dwell_hours))

        rate, done, total = _simulate(
            vehicles, "office1", base_peak, background, predict_fn, True
        )
        print(f"demand x{mult}: completion={rate:.4f} ({done}/{total})")


if __name__ == "__main__":
    main()
