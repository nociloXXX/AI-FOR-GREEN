# 模块4 干预推演改进说明

## 结论

- `module_4_intervention.py` 是 `capacity_edf` 反事实推演的唯一入口；`module_3_world_model.py`
  只保留 `forecast_natural`，不再重复实现 `forecast_capacity_edf`。
- 默认策略保持 `strategy="edf"`（最早离开优先）。官方把 `capacity_edf` 定义为
  “容量扣除背景后按预测离开最早优先”，因此偏离 EDF 只会增大干预效应误差。
- 在线“未满足电量”统一为世界模型推演终点仍未充上的电量（`unmet_kwh`）；
  离线有真实标签时用 `actual - dispatch`（`evaluate.decision_metrics`），两种口径
  不再混用。

## 策略对照实验

运行 `python offline_intervention_eval.py`，用公开训练标签做“完美预测”消融：

1. caltech 6 辆在场车辆、前 4 步容量 10kW 后放开：`edf` 与 `slack` 的轨迹一致，
   served 均为 49.6 kWh（容量限制期间总负荷被压到容量线，放开后都能补足）。
2. 截断场景同样显示，站级 `ev_load_kw` 主要受“容量序列 + 自然需求”决定；
  分配顺序只有在多步内“有的车没充完就离开”的截断情况下才可能拉开差距，且
  对站级负荷的影响通常很小。

结论：干预响应差的改善空间来自需求/停留/到达预测精度（模块2），而不是容量内
分配顺序。因此提交默认保留官方一致的 EDF；`strategy="slack"` 仅作为离线消融保留。
