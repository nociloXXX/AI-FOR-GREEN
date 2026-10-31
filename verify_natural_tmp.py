"""临时冒烟：绕过 sklearn，单独验证 module_3 的自然推演输出。"""
import sys
import types

from acn_causal.student_api import (
    ArrivalFeatures,
    Observation,
    Prediction,
    VehicleObservation,
)


def _stub_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _predict(features):
    return Prediction(10.0, 2.0)


def _arrival_rate(site, hour, weekday):
    return 0.5


_stub_module(
    "module_1_data",
    summarize_site=lambda training: ({}, {}),
    make_features=lambda *a: None,
    build_weekday_hour_rates=lambda training: {},
)
_stub_module(
    "module_2_prediction",
    PredictionModule=None,
)
_stub_module("module_4_intervention", forecast_capacity_edf=lambda *a: {})
_stub_module("module_5_decision", act_edf=lambda *a: {})

from module_3_world_model import forecast_natural, rollout  # noqa: E402


def main():
    features = ArrivalFeatures("t1", "caltech", 9.4, 0, 6.6, None)
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
    )
    for horizon in (1, 4, 12):
        out = forecast_natural(observation, horizon, _predict, _arrival_rate)
        assert len(out["ev_load_kw"]) == horizon
        assert len(out["active_count"]) == horizon
        assert all(v >= 0 for v in out["ev_load_kw"])
        assert all(isinstance(v, int) and v >= 0 for v in out["active_count"])
        print(horizon, out)
    print("natural rollout ok")


if __name__ == "__main__":
    main()
