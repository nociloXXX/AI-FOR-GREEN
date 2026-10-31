"""AI FOR GREEN Track A 提交入口：五模块在线模型。

模块划分：
  1. 数据处理    module_1_data.py
  2. 预测        module_2_prediction.py
  3. 世界模型    module_3_world_model.py
  4. 干预推演    module_4_intervention.py
  5. 充电决策    module_5_decision.py

评测沙箱只调用 build_model(training)，返回的对象必须提供 predict、
arrival_rate、act、forecast、explain 五个方法。
"""

from module_1_data import summarize_site
from module_2_prediction import PredictionModule
from module_3_world_model import forecast_natural
from module_4_intervention import forecast_capacity_edf
from module_5_decision import act_edf


class FiveModuleModel:
    def __init__(self, training):
        self.labels = summarize_site(training)
        self.prediction = PredictionModule(training)

    def predict(self, features):
        return self.prediction.predict(features)

    def predict_quantiles(self, features, levels=(0.5, 0.75, 0.9)):
        return self.prediction.predict_quantiles(features, levels)

    @property
    def future_sampler(self):
        return self.prediction.future_sampler

    def future_cohorts(self, site, hour, weekday):
        return self.prediction.future_cohorts(site, hour, weekday)

    def survival_remaining_hours(self, site, dwell, elapsed, dt):
        return self.prediction.survival_remaining_hours(site, dwell, elapsed, dt)

    def arrival_multipliers(self, observation, steps):
        return self.prediction.arrival_multipliers(observation, steps)

    def occupancy_multiplier(self, observation, steps):
        return self.prediction.occupancy_multiplier(observation, steps)

    def arrival_rate(self, site, hour, weekday):
        return self.prediction.arrival_rate(site, hour, weekday)

    def act(self, observation):
        return act_edf(observation, self.predict)

    def forecast(self, observation, horizon_steps, intervention):
        if intervention == "capacity_edf":
            return forecast_capacity_edf(
                observation,
                horizon_steps,
                self.predict,
                self.arrival_rate,
                future_cohorts_fn=self.future_cohorts,
                survival_fn=self.survival_remaining_hours,
                arrival_calibration_fn=self.arrival_multipliers,
                occupancy_multiplier_fn=self.occupancy_multiplier,
            )
        return forecast_natural(
            observation,
            horizon_steps,
            self.predict,
            self.arrival_rate,
            future_cohorts_fn=self.future_cohorts,
            survival_fn=self.survival_remaining_hours,
            arrival_calibration_fn=self.arrival_multipliers,
            occupancy_multiplier_fn=self.occupancy_multiplier,
        )

    def explain(self):
        return {
            "method": (
                "five-module online model: LightGBM dual-branch demand/dwell "
                "(with/without last requested) for caltech/jpl, HistGradientBoosting "
                "median plus KNN blending for office1, weekday-hour arrivals, physics "
                "rollout world model, capacity_edf counterfactual, least-laxity-first "
                "+ TOU price-aware charging decision"
            ),
            "assumptions": [
                "charger limit is known at arrival",
                "demand and dwell are unchanged by the capacity intervention",
                "future arrivals are modeled from public history, not known in advance",
                "price-aware deferral only applies when a vehicle still has spare dwell slack, so completion outranks cost",
                "least-laxity-first protects at-risk vehicles; price deferral only applies to vehicles with positive laxity",
            ],
            "limitations": [
                "arrival features cannot capture all driver intent",
                "rollout does not model battery taper",
                "counterfactual response is model-internal physics, not real behavioral causal evidence",
            ],
        }


def build_model(training):
    return FiveModuleModel(training)
