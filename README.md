# AI FOR GREEN Track A 提交模型

面向《停车场充电世界模型》赛道的在线充电模型。提交入口是根目录
`submission.py` 的 `build_model(training)`，返回对象实现五个接口方法。

## 模型结构

五个模块按 `数据 -> 预测 -> 世界模型 -> 干预推演 -> 充电决策` 组织：

1. 数据处理 `module_1_data.py`：训练表整理与特征构造。
2. 预测 `module_2_prediction.py`：LightGBM 双分支预测需求电量与停留时长，
   星期×小时到达率表；`last_requested_energy_kwh` 缺失时自动退化。
3. 世界模型 `module_3_world_model.py`：natural 条件的物理前向推演。
4. 干预推演 `module_4_intervention.py`：capacity_edf 反事实容量推演。
5. 充电决策 `module_5_decision.py`：完成率优先的 EDF 分配 + 价格延后绿色调度。

## 接口输入输出

- `predict(ArrivalFeatures) -> Prediction(demand_kwh, dwell_hours)`
- `arrival_rate(site, hour, weekday) -> 每15分钟期望到达数`
- `act(Observation) -> {token: 请求功率kW}`
- `forecast(Observation, horizon_steps, intervention) -> {ev_load_kw, active_count}`
- `explain() -> {method, assumptions, limitations}`

## 假设

- 充电桩额定上限在车辆到达时已知。
- 需求电量与停留时长不受容量干预影响。
- 未来到达由公开历史到达率建模，不是已知信息。

## 局限

- 到达特征无法完全捕捉驾驶员意图。
- 前向推演不建模电池 taper。
- 反事实响应是模型内部物理效应，不是真实用户行为的因果证据。

## 运行环境

官方离线 CPU 沙箱 `cpu-ml-methods-v3`：`numpy 1.26.4`、`pandas 2.1.4`、
`scipy 1.11.4`、`scikit-learn 1.4.1.post1`、`torch 2.15.0+cpu`、
`xgboost-cpu 3.1.3`、`lightgbm 4.6.0`、`statsmodels 0.14.6`。沙箱不联网。

## 本地验证与打包

```bash
python verify_submission.py
python make_submission.py --source . --out my_submission.zip \
  --include module_1_data.py module_2_prediction.py module_3_world_model.py \
  module_4_intervention.py module_5_decision.py weights/*.txt
```

## 协作

仓库地址：https://github.com/nociloXXX/AI-FOR-GREEN
日常流程：`git pull` -> 改代码 -> `git add .` -> `git commit` -> `git push origin main`。
