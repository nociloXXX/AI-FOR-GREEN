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
        self.labels, self.rates = summarize_site(training)
        self.prediction = PredictionModule(training)

    def predict(self, features):
        return self.prediction.predict(features)

    def arrival_rate(self, site, hour, weekday):
        return self.prediction.arrival_rate(site, hour, weekday)

    def act(self, observation):
        return act_edf(observation, self.predict)

    def forecast(self, observation, horizon_steps, intervention):
        if intervention == "capacity_edf":
            return forecast_capacity_edf(
                observation, horizon_steps, self.predict, self.arrival_rate
            )
        return forecast_natural(
            observation, horizon_steps, self.predict, self.arrival_rate
        )

    def explain(self):
        return {
            "method": (
                "five-module online model: data features -> ExtraTrees demand/dwell "
                "+ weekday-hour arrivals -> physics world model -> counterfactual "
                "capacity rollout -> EDF charging decision"
            ),
            "assumptions": [
                "charger limit is known at arrival",
                "demand and dwell are unchanged by the capacity intervention",
                "future arrivals are modeled from public history, not known in advance",
            ],
            "limitations": [
                "arrival features cannot capture all driver intent",
                "rollout does not model battery taper",
                "counterfactual response is model-internal physics, not real behavioral causal evidence",
            ],
        }


def build_model(training):
    return FiveModuleModel(training)
