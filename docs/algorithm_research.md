# A 赛道《停车场充电世界模型》算法与论文调研（真实接口版）

> 交付对象：`module_1_data` 数据处理、`module_2_prediction` 预测、`module_3_world_model` 世界模型、`module_4_intervention` 干预推演、`module_5_decision` 充电决策五个模块的执行会话。
> 阅读方式：先看「规则合规闸门」和「总排序」，再跳到对应方法小节。结论按“队友直接照做”的口径写，不展开公式推导。
> 调研日期：2026-10-31（GMT+8）。

## 0. 规则合规闸门（新论文接入前必读）

平台沙箱约束与硬门槛如下。**以后每拿到一篇论文，先过这八条，不合规就不接入代码。**

1. 离线、不安装依赖：只能用预装的 `numpy/pandas/scipy/scikit-learn/torch/xgboost/lightgbm/statsmodels/patsy/packaging`；纯 Python 依赖可随 ZIP 源码提交，不能带自定义 `.so`/可执行求解器，不能跑 `pip`。
2. `submission.py` 不得训练深度学习模型；允许提交预训练权重做 CPU 推理（`torch` 预装），但一天赛程内不建议走这条。
3. 不得把场景数据、题目或接口输入交给在线大模型 API 获取结果；`explain.json` 只描述本地算法，不产生结果。
4. ZIP ≤ 20 MiB、解压 ≤ 100 MiB、≤ 500 项；ML 白名单 `.py/.json/.txt/.md/.npz/.npy/.pkl/.joblib`；权重优先 `.npz/.npy` 并用 `np.load(..., allow_pickle=False)`。
5. 禁用数据集发布方配套软件库（如 ACN 数据源发布方库）及其源码，不得改名、内嵌或动态加载规避；`acn_causal` 接口、基线与公开训练数据不在禁用范围。
6. 物理合法 + 逐场景完成率门槛：Caltech ≥ 0.85、JPL ≥ 0.65、Office ≥ 0.85；任一场景不达标即 DQ。
7. 时间/资源：公开开发累计 CPU 120 秒、私有终测累计 CPU 180 秒；内部墙钟 180/280 秒；整树内存 2 GiB、单线程；首次响应 60 秒，之后每次接口调用 30 秒。
8. 每篇新论文的检查流程：是否训练深度模型？是否依赖需安装的库？是否联网调 API？是否超体积/超时/超内存？是否触碰禁用库？是否增加物理违规或完成率风险？六项任一不通过即拒绝接入。

## 1. 真实任务与现有模板的对应

真实任务不是旧的“15 分钟场站负荷合成预测”，而是车辆级在线接口。`build_model(training)` 返回的对象必须提供五个方法：

| 接口 | 做什么 | 现有实现 |
| --- | --- | --- |
| `predict(ArrivalFeatures)` | 预测一辆车 `demand_kwh` 与 `dwell_hours` | `module_2_prediction.py`：ExtraTrees 多输出回归 |
| `arrival_rate(site, hour, weekday)` | 每 15 分钟期望到达数 | `module_1_data.py` 的星期×小时直方图 |
| `act(Observation)` | 给当前在场车分配请求功率 | `module_5_decision.py`：EDF 最早离开优先 |
| `forecast(..., "natural")` | 自然充电下未来负荷与活跃车辆 | `module_3_world_model.py`：物理 rollout |
| `forecast(..., "capacity_edf")` | 容量干预下按 EDF 的负荷与活跃车辆 | `module_4_intervention.py`：容量扣除背景 + EDF |
| `explain()` | 结构化说明方法/假设/限制 | `submission.py` / `explain.json` |

当前 `submission.py` 的五模块基线已经合规且能闭环。**第一步先原样打包提交一次验证流程**，再在它的基础上做增量替换，不要推倒重写。

固定环境要点：预装 `lightgbm==4.6.0`、`xgboost-cpu==3.1.3`、`statsmodels==0.14.6`；整数规划可用 `scipy.optimize.milp`（HiGHS）；没有 PuLP/OR-Tools/Prophet，也不要尝试安装。

## 2. 总排序（一天赛程内的投入优先级）

| 优先级 | 动作 | 涉及接口 | 是否采用 | 一句话理由 |
| --- | --- | --- | --- | --- |
| P0 | 原样提交现有五模块基线 | 全部 | 是 | 已合规、能闭环，先锁定可提交版本 |
| P0 | 预测换 LightGBM 或保留 ExtraTrees，补 `site`、小时×星期交互 | `predict` | 是 | 预装可用、训练快、直接抬高下游所有模块质量 |
| P1 | 到达率做星期×小时直方图 + 平滑 | `arrival_rate` | 是 | O(1) 查询、可解释，是“下一步到达数”与未来到达推演的基础 |
| P1 | 保留物理 rollout 与 EDF，确保两种反事实同状态、同假设 | `forecast` | 是 | 当前实现已符合官方定义，重点是别改坏 |
| P1 | 保留 EDF 决策，只做容量与完成率安全 | `act` | 是 | EDF 在 6.6 kW 均质桩下是完成率最优的简单在线策略 |
| P2 | dwell 分位数/对数目标校准 | `predict` | 可选 | 停留时长右偏，适当校准能改善 EDF 优先级与反事实 |
| P2 | 价格感知延后充电（完成率安全前提下） | `act` | 可选 | 能降成本，但有完成率回退风险，须先用自评脚本验证 |
| P3 | LSTM/Transformer、DQN/PPO | 全部 | 否 | 训练 DL 在 `submission.py` 被禁；纯推理也吃 2 GiB/120 s，一天内不划算 |
| P3 | Prophet、OR-Tools/PuLP | 预测/决策 | 否 | 沙箱未预装，无法安装；整数规划改用 SciPy HiGHS |

## 3. 分方法结论

### 3.1 predict：车辆需求与停留时长预测

**模型是什么**
在一辆车到达时，用可见特征（站点、当地小时、星期、桩功率上限）回归两个目标：交付电量 `demand_kwh` 与连接停留时长 `dwell_hours`。这两个量是后续世界模型、反事实与 EDF 决策的共同输入。

**解决什么问题**
在线评测看不到未来车辆的真实需求与离开时间，只能从历史交付和停留标签学习条件分布，给出点估计供调度使用。

**与本赛题的对应关系**
训练表列：`hour, weekday, charger_limit_kw, demand_kwh, dwell_hours, day, last_requested_energy_kwh`。注意 `charger_limit_kw` 固定 6.6，`last_requested_energy_kwh` 在训练截止（2020-01-01 前）基本全为 `null`，**不要把它当主特征，更不能当 0 或当目标**；未来测试车辆同样多为 `None`。

**推荐顺序与理由**
1. ExtraTrees（保留，P0）：基线已用 96 棵树，CPU 秒级、不依赖安装，先跑通。
2. LightGBM（P0/P1）：预装且原生处理缺失、训练快；建议单独为 `demand` 和 `dwell` 各训一个单目标模型，方便分别调参与裁剪。
3. XGBoost（P1）：与 LightGBM 接近，作为交叉验证后的备选。
4. 特征增强（P1）：`sin/cos(hour)`、`weekday`、`site` 独热、`hour×weekday` 交互；`charger_limit_kw` 恒定时可保留但信息量小。
5. dwell 校准（P2）：`dwell_hours` 右偏且被停留窗口截断，可试 `log1p(dwell)` 回归再 `expm1`，或用分位数回归；只影响 EDF 优先级，不改变物理推演。
6. 不上 LSTM/Transformer、不训练任何深度模型（P3）。

**优缺点**
- 优点：表格特征 + 树模型在 ACN 数据集上表现稳定、可解释、训练快，直接决定下游所有模块误差。
- 缺点：点估计无不确定性；停留时长长尾；站点间分布差异大，必须按站点分别训练。

**怎么接进现有模板**
改 `module_2_prediction.py` 的 `PredictionModule`：保留 `arrival_rate`，把 ExtraTrees 换成 LightGBM（`lightgbm` 预装）或保留 ExtraTrees 再加特征；`predict` 继续做缓存与裁剪，`demand_kwh` 裁剪到正数、`dwell_hours` 裁剪到 `[0.25, 24]`。

### 3.2 arrival_rate：到达率估计

**模型是什么**
估计指定站点、小时、星期下每 15 分钟的期望到达车辆数，本质是非齐次泊松过程（NHPP）的强度估计。

**解决什么问题**
`forecast` 要模拟未来未知到达，评分里也有“下一步到达数”项目；到达率给未来车辆注入提供期望流。

**与本赛题的对应关系**
现有 `build_weekday_hour_rates` 已按站点×星期×小时统计到达数，除以天数再除以 4，得到每 15 分钟期望数，方向正确。

**推荐顺序与理由**
1. 星期×小时直方图 + 拉普拉斯平滑（P1）：对零单元加伪计数，避免稀疏小时估计为 0；O(1) 查询，可解释。
2. 工作日/周末分组（P1）：数据量少时可把 7 个星期折叠成两类，减少方差；Caltech/Office 工作日照明显，JPL 可保留 7 类。
3. 泊松 GLM（P2，statsmodels）：用循环小时基 + 星期因子拟合强度，理论上更平滑，但直方图通常已够，不建议先做。
4. 不上深度点过程模型（P3）。

**优缺点**
- 优点：简单、稳健、查询 O(1)，是文献中 EV 到达建模的标准基线。
- 缺点：burst（到达高峰）场景由复制车池放大，历史强度无法预知放大倍数；这是赛题公开的信息缺口，所有队伍同条件。

**怎么接进现有模板**
在 `module_1_data.py` 的 `build_weekday_hour_rates` 里加平滑与工作日/周末选项；保持返回 `(site, weekday, hour) -> rate` 的接口不变。

### 3.3 世界模型：natural 前向推演

**模型是什么**
从当前观测出发，把每辆在场车表示为 `[剩余需求, 剩余停留步数, 桩功率上限]`，逐 15 分钟步推进：按期望到达注入未来车辆、按 `min(桩上限, 剩余需求/步长, 可用容量)` 交付电量、按剩余停留步数离场，输出未来 `ev_load_kw` 与 `active_count`。

**解决什么问题**
回答“不干预容量时，未来一段时间的站点负荷与活跃车辆会怎样”，是世界模型的主体。

**与本赛题的对应关系**
`module_3_world_model.py` 的 `rollout` 已实现这一物理推演，`natural` 传无限容量。物理约束（6.6 kW、能量守恒、容量上限）与官方一致。

**推荐顺序与理由**
1. 保留现有确定性 rollout（P0/P1）：单条期望路径足够，且与 `capacity_edf` 共用同一推演器，天然满足“同状态、同假设”。
2. 活跃车辆口径（P1）：指已到、未离、仍有充电需求的车，现有 `active_count` 口径正确，不要改成“还插着枪”。
3. 未来到达注入（P1）：现有“累积期望到达数，达到 1 再注入一辆”的做法可用；可改为每步按泊松期望抽样 1 条路径，但别做多副本蒙特卡洛，省 CPU。
4. 不引入电池衰减曲线（P3）：评测固定 6.6 kW，官方基线也不建模 taper，按满功率 + 剩余需求截断即可。

**优缺点**
- 优点：物理一致、确定性、快，直接对齐评分定义。
- 缺点：单路径期望到达平滑了随机性；需求/停留预测误差会传导到负荷与活跃车辆误差。

**怎么接进现有模板**
不动 `rollout` 的物理核心；若要做蒙特卡洛，只做 1 条路径并保持 `natural` 与 `capacity_edf` 用同一批采样（或都不采样）。

### 3.4 干预推演：capacity_edf 反事实

**模型是什么**
同一当前状态出发，只把可用容量从“无限”换成 `intervention_capacity - forecast_background`，并按预测离开最早优先（EDF）分配功率，输出限流后的负荷轨迹与后续回补。

**解决什么问题**
评分里的干预项目看的是 `capacity_edf - natural` 的响应差，要同时优于零效应与固定中位数两个基线，复制 natural 不得分。

**与本赛题的对应关系**
`module_4_intervention.py` 已按官方定义实现：容量取 `intervention_capacity or forecast_capacity`，再减背景；EDF 排序；需求/到达/离开假设与 natural 完全相同。关键是**保持两个条件同状态、同注入、同离场**。

**推荐顺序与理由**
1. 保留现有实现（P0/P1）：结构正确，先验证不因干预而改变需求/到达假设。
2. 保证回补建模（P1）：限流时剩余需求保留到后续步，现有 `car[0] -= power*dt` 已实现回补，不要清掉。
3. 提高需求/停留预测精度（P1）：响应差的误差主要来自需求与停留预测，优先改善 `predict`。
4. 不复制 natural、不把效应压成 0（P0 合规）：否则干预项目不得分。
5. 不做全信息离线最优反推（P3）：赛题只要求按在线信息推演，离线最优只用于理解上界。

**优缺点**
- 优点：物理一致、与官方 EDF 定义严格对齐，响应差由同一推演器自然得出。
- 缺点：效应精度受需求/停留/到达率预测限制；容量序列长时，滚动的 O(H·N) 计算要控制在毫秒级。

**怎么接进现有模板**
`forecast_capacity_edf` 继续复用 `rollout`；只改容量函数，不要复制一份推演逻辑，避免两个条件出现不一致。

### 3.5 act：充电决策

**模型是什么**
在线调度：根据当前观测与每车预测，决定给当前连接车辆分配多少功率。现有实现是 EDF：按预测剩余停留步数排序，依次给 `min(桩上限, 剩余需求/步长)` 直到可用容量用完。

**解决什么问题**
在容量约束内最大化逐车完成率，其次降低峰值与成本；完成率门槛是硬性 DQ 条件。

**与本赛题的对应关系**
`module_5_decision.py` 的 `act_edf` 已实现：可用容量 = `capacity_kw - background_kw`；跳过已完成车辆；按最早离开优先给满功率。这与 ACN 文献结论一致：在均质 6.6 kW 桩下，EDF 是完成率极优的简单在线策略。

**推荐顺序与理由**
1. 保留 EDF（P0）：完成率优先，最稳，先确保 Caltech/JPL/Office 门槛。
2. 行动限额只用 `capacity_kw` 与 `forecast_capacity`（P0 合规）：绝不把 `intervention_capacity` 当行动容量。
3. 容量前瞻（P2）：若未来某步 `forecast_capacity` 更紧，可提前充；但先做简单 EDF，避免复杂化。
4. 价格感知延后（P2）：电价 0.6-1.5 元/kWh，可在“剩余停留长、需求小”的车上把充电延到低价段，但必须先用 `public_self_eval.py` 验证完成率不回退。
5. MILP（P3，可选对照）：需要精确开/关桩或成本最优时用 `scipy.optimize.milp`（HiGHS），设 `time_limit` 与 `mip_rel_gap`，无解时回退 EDF；不用 OR-Tools/PuLP。
6. 不上 DQN/PPO（P3）：训练 DL 在 `submission.py` 被禁，纯推理也不划算。

**优缺点**
- 优点：简单、可解释、完成率高、CPU 开销极小。
- 缺点：贪心不显式优化电费与峰值；对预测误差敏感（停留低估会过度提前充电）。

**怎么接进现有模板**
保留 `act_edf` 为默认；如做价格延后，作为 `act_edf` 之后的轻量后处理，并加完成率回归开关。

### 3.6 explain

保持结构化 `{method, assumptions, limitations}`，内容必须与真实代码一致。换模型后同步更新 `submission.py` 的 `explain()` 与 `explain.json`；不要写“未使用的架构”或“在线 API 获取结果”。

## 4. 方向论文映射与合规判定

| 方向 | 论文典型结论 | 本任务位置 | 合规性 | 一天内建议 |
| --- | --- | --- | --- | --- |
| EV 充电需求/停留预测（LightGBM/XGBoost/ExtraTrees） | 表格特征下树集成稳定、可解释、快 | `predict` | 合规，预装 | 采用：ExtraTrees 保底，LightGBM 首选 |
| 到达率估计（NHPP） | 星期×小时强度是标准做法 | `arrival_rate`、未来到达注入 | 合规，纯 numpy | 采用：直方图 + 平滑 |
| 智能充电调度（EDF/LLF/LP） | EDF 是均质桩下完成率最优的简单基线，LP 是离线最优上界 | `act`、`forecast capacity_edf` | 合规，纯 numpy/scipy | 采用 EDF；LP 只用 `scipy` 或闭式注水 |
| 模型预测控制（MPC） | 滚动窗口用预测值调度，ACN 系列论文的标准范式 | `act` 的前瞻版 | 合规，用预测值不是真值 | P2 可选，先 EDF |
| MILP / OR-Tools / PuLP | 整数调度精确但规模敏感 | `act` 对照 | OR-Tools/PuLP 未预装，禁用自制二进制 | 需要时用 `scipy.optimize.milp`/HiGHS，无解回退 EDF |
| ARIMA / Prophet | ARIMA 可解释，Prophet 适合强季节 | `predict` 替代 | ARIMA 可经 statsmodels；Prophet 未预装 | 不采用，GBM 已覆盖 |
| 强化学习 DQN/PPO | 文献可行但调参/样本/复现成本高 | `act` | `submission.py` 禁训 DL；纯推理可但不划算 | 不采用 |
| LSTM / Transformer | 序列建模需大样本与调参 | `predict` | 同上，禁训 DL | 不采用 |
| 在线大模型 API | 外包结果 | 全部 | 明确禁止 | 不采用 |
| 排队论 M/M/c | 等待/占用分析 | 仅解释 | 合规，纯 Python | 可选，写进 explain 不计主分 |

## 5. 120 秒 CPU / 2 GiB 预算建议

- 预算分配：`build_model` 内训练与建表控制在 30 秒内，权重加载 10 秒内；`predict`/`arrival_rate` 每次毫秒级，`act` 每次 < 10 ms，`forecast` 每次 < 50 ms。
- 权重用 `.npz/.npy` + `allow_pickle=False`；LightGBM/ExtraTrees 单站模型很小，三站合计远低于 20 MiB。
- 不要加载 torch、不要加载 `timeseries_*.jsonl.gz`：官方说明这些曲线是离线补充，不是固定 6.6 kW 评测模型所需，评测时不读、也省时间与内存。
- `predict` 按 `(site, rounded hour, weekday, charger_limit)` 缓存（基线已做），减少重复树推理。
- `forecast` 用单条期望路径，不做蒙特卡洛多副本，避免累计 CPU 超限。

## 6. 给执行模块的一句话指令

- 数据处理（module_1_data）：保留现有特征与星期×小时到达率，加平滑与工作日/周末折叠；不要依赖 `last_requested_energy_kwh`。
- 预测（module_2_prediction）：ExtraTrees 保底，LightGBM 首选，按站点分开训练；需求与停留分别裁剪到合法范围。
- 世界模型（module_3_world_model）：保留物理 rollout，`natural` 无限容量，活跃车辆口径不变。
- 干预（module_4_intervention）：复用同一 `rollout`，容量 = `intervention_capacity - forecast_background`，EDF，保证两条件同状态。
- 决策（module_5_decision）：保留 EDF，行动容量只用 `capacity_kw` 与 `forecast_capacity`；价格延后要加完成率回归开关。
- 提交（submission.py / explain.json）：先原样提交验证流程，换模型后同步 `explain()` 与 `explain.json`。

## 7. 参考来源

- Lee, Z. J., Li, T., & Low, S. H., "ACN-Data: Analysis and Applications of an Open EV Charging Dataset"（Caltech/JPL/Office1 数据源与需求/停留/调度分析）。
- Lee, Z. J. et al., "Adaptive Charging Networks: A Framework for Smart Electric Vehicle Charging"（在线调度、EDF、离线 LP 上界）。
- Energy Informatics (Springer), "Using EV charging control to provide building load flexibility" (2023)，MPC 负荷塑形，DOI 10.1186/s42162-023-00261-8。
- IEEE Access, "A Comprehensive Survey of Electric Vehicle Charging Demand Forecasting Techniques" (2024)。
- arXiv 2410.16425, "Advancements in Electric Vehicle Charging Optimization: A Survey of Reinforcement Learning Approaches"。
- Transportation Research Part E, "Reinforcement learning for electric vehicle charging scheduling: A systematic review" (2024)。
- 官方文档：`docs/official_README.md`、`docs/INTERFACE.md`、`docs/SCORING.md`、`docs/SUBMISSION_GUIDE.md`、`docs/TIMESERIES.md`。

## 8. CNKI 论文逐篇审查（2026-10-31）

> 审查口径：CNKI 正文不可直接访问，以下按题录、摘要库与研究方向判断；只确认合规且对五个接口有直接价值的才建议接入，否则不入代码。

| 论文 | 主题 | 与本任务关系 | 合规/一天内结论 |
| --- | --- | --- | --- |
| 采用功率限制的住宅区电动汽车有序充电控制策略（电网技术 2016） | 在线控制 + 离线优化，TOU 电价，功率限制，移峰填谷 | 对 `act` 与 `capacity_edf` 有借鉴：在线只按当前可用容量分配，离线只做报告上界 | 在线规则可借鉴；离线优化不接在线链路 |
| 基于双目标分层优化和 TOPSIS 排序的电动汽车有序充电策略（2021） | 成本 + 负荷波动双层优化，TOPSIS 定充电优先级，越限实时调整 | 对 `act` 最有价值：把 EDF 的单一离场优先级扩展成多指标评分（剩余需求、停留、电价） | TOPSIS 评分纯 numpy 可做，分层优化不照搬 |
| 支持时变需求响应激励价格的电动汽车优化充电策略（2025） | 强化学习（MDP）+ 时变激励价 | 方法 RL、假设时变激励价 | 不接入：`submission.py` 禁训 DL，且赛题是固定 TOU |
| 考虑车网互动的电动汽车有序充电策略 | V2G/放电 | 接口只有单向 6.6 kW 充电，无放电 | 不接入 |
| 面向渗透率最大化的电动汽车智能充电策略 | 选址/容量规划 | 非在线调度 | 不相关 |
| 基于 WRSN 的多节点按需充电策略研究 | 无线可充电传感网 | 非充电桩 | 不相关 |
| 基于电动汽车充电优先级的协调充电策略（2024） | 时间监测形成充电优先级序列 | 对 `act` 轻量可借鉴 | 可借鉴优先级思路 |
| 基于充电优先级确定机制的电动汽车充电调度策略 | 充电优先级调度 | 对 `act` 同类可借鉴 | 可借鉴优先级思路 |
| 多目标协同下充电物资配送优先级策略 | 物资配送物流 | 非充电调度 | 不相关 |
| 基于顾客优先级的移动充电车路径规划研究 | 移动充电车路径 | 非固定停车场桩充 | 不相关 |
| 车充换电设施集总功率实时分解策略 | 集总功率分解 | 接口已给 `delivered_kwh`，桩固定 6.6 kW | 不相关 |
| 绿色物联网中多充电器协作的高效远场无线充电调度策略研究 | 无线 IoT 充电 | 非充电桩 | 不相关 |
| 其余截断标题（电动汽车优化充电、大规模电动…、价格相关等） | 待补全标题 | 大概率属有序充电优化或价格类 | 待确认，先不接入 |

一句话结论：这批论文只影响 `act` 的优先级规则设计，不影响 `predict`、`arrival_rate`、`forecast` 的物理推演。可取的是“在线容量分配 + TOPSIS/多指标优先级 + 越限实时调整”，不可取的是它们普遍隐含的“已知全部车辆需求/到达/离开”的离线全局优化，以及 V2G、RL、无线/物流等与赛题无关的设定。

### 8.1 模块归属（直接照做）

| 论文 | 放进哪个模块 | 改动位置与方式 |
| --- | --- | --- |
| 采用功率限制的住宅区有序充电 | `module_5_decision.py` | `act_edf` 保持在线容量分配；其“离线优化”部分不进在线链路，只写进离线报告 |
| 双目标分层优化 + TOPSIS 排序 | `module_5_decision.py` | 把 `act_edf` 的单一 `available` 排序改成 TOPSIS 多指标评分（剩余需求、停留、电价），越限时实时调功率 |
| 基于电动汽车充电优先级的协调充电 | `module_5_decision.py` | 用时间监测形成的优先级序列替换/扩展 `act_edf` 的排序键 |
| 基于充电优先级确定机制的调度 | `module_5_decision.py` | 同上，只借鉴优先级评分，不照搬离线全局优化 |

不改动的模块：`module_1_data`、`module_2_prediction`、`module_3_world_model`；`module_4_intervention` 已按官方定义实现了“容量内 EDF”，功率限制那篇不需要再往这里加代码。

### 8.2 第二批 CNKI 论文审查（2026-10-31）

| 论文 | 主题 | 与本任务关系 | 合规/一天内结论 |
| --- | --- | --- | --- |
| 考虑充电策略的 U 形自动化码头 | 港口自动化码头设备充电 | 非停车场 EV 桩充 | 不相关 |
| 计及 AGV 充电调度的自动化码头光储充电站低碳规划 | 港口 AGV + 光储规划 | 规划与港口物流 | 不相关 |
| 基于蚁群算法的电动汽车充电调度系统设计 | 蚁群元启发式求解大规模调度 | 离线全局优化，不适合每步在线 `act` | 不接在线；可作离线报告对照 |
| 基于负荷预测的居民小区电动汽车多目标充电调度策略 | 高斯过程回归负荷预测 + TOU 多目标调度 | 对 `act` 借鉴多目标优先级；GPR 是站级负荷，不用于车辆级 `predict` | 多目标权重可借鉴到 `act`，算法不照搬 |
| 基于改进 PSO-SA 的配电网下 EV 充放电调度 | V2G 充放电 + PSO-SA 元启发式 | 接口无放电，元启发式不宜在线 | 不接入 |
| 基于改进 NSGA-Ⅱ 的电动汽车多目标充电调度优化 | 多目标进化算法 | 多目标思想可参考，NSGA-Ⅱ 不宜在线 | 仅离线报告/explain，不进 `submission.py` 在线路径 |
| 考虑车辆路径约束的港口微电网电动转运车充电调度 | 港口转运车路径 + 微电网 | 非停车场桩充 | 不相关 |
| 码头 AGV 弹性调度 | 港口 AGV | 非停车场桩充 | 不相关 |
| 考虑需求分类的电动汽车预约充电调度优化 | 预约机制 + 需求分类 | 接口无预约；需求分类可自研为 `predict` 特征或 `act` 优先级档 | 借鉴“分类”思路，预约部分不适用 |
| 其余截断标题（基、充电调度策、略、计与实现等） | 待补全 | 无法确认 | 先不接入 |

模块归属：唯一可直接落代码的是 `module_5_decision.py`（多目标优先级、需求分类档位）；“需求分类”也可作为 `module_2_prediction.py` 的轻量派生特征。蚁群/PSO-SA/NSGA-Ⅱ 这类元启发式允许离线分析，但不要写进在线评测路径，会吃满 120 秒 CPU 预算。

### 8.3 第三批 CNKI 论文审查（2026-10-31）

| 论文 | 主题 | 与本任务关系 | 合规/一天内结论 |
| --- | --- | --- | --- |
| 基于充电优先级确定机制的电动汽车充电调度策略 | 充电优先级调度 | 已在 8.1 审查，属 `module_5_decision` | 同 8.1，重复项 |
| 基于分时电价的钦州港自动化码头智能导引车充电策略优化 | 港口 IGV + TOU 充电 | 港口 AGV 领域不相关；TOU 降本思路已由价格延后覆盖 | 不入代码 |
| 多场景资源约束下电动汽车充电调度优化研究 | 资源约束下的充电调度 | 资源约束即容量约束，可参考目标/约束设计 | 参考即可，离线优化不入在线路径 |
| 考虑动态行程时间及能耗… | 路径规划 + 行程时间/能耗 | 接口无路径，非停车场桩充 | 不相关 |
| 基于 LSTM 混合网络预测的新能源汽车充电线路调度 | LSTM + 线路调度 | LSTM 属深度模型且是线路调度 | 不接入：`submission.py` 禁训 DL |
| 电动公交充电设施与充电调度联合优化 | 公交设施规划 + 调度 | 公交/设施规划，非本任务 | 不相关 |
| 基于智能优化的电动汽车有序充电调度技术研究 | 元启发式智能优化 | 离线全局优化为主 | 参考即可，不进在线评测路径 |
| 基于分布式近似动态规划的电动汽车实时充电调度策略 | ADP 实时调度 + 值函数近似 | 方法上与 `act` 最相关，但“分布式”不适用单站，ADP 离线训练复杂 | 一天内不实现，可写进 explain 作“未采用方向” |

模块归属：本批没有需要新派发的代码工作。唯一方法侧可参考的 ADP 属于 `act` 方向，作为 explain 里的对照说明即可，不写实现；其余港口 AGV、路径、公交、LSTM、元启发式均不接入。

### 8.4 第四批 CNKI 论文审查（2026-10-31）

| 论文 | 主题 | 与本任务关系 | 合规/一天内结论 |
| --- | --- | --- | --- |
| 非理想网络环境下基于分层架构的电动汽车分布式充电调度 | 分布式分层调度 + 非理想网络通信 | 接口是单站点、集中观测、离线本地沙箱，无多代理通信/时延/丢包 | 不相关，不接入任何模块 |

## 9. 当前痛点定向调研（第五轮，针对模块现状）

> 面向当前实现：ExtraTrees demand/dwell + last_requested 分桶双分支、物理 rollout、capacity_edf、slack 分桶 TOPSIS + TOU 感知 EDF。约束：单核 CPU、开发 120s/180s 墙钟/2GiB，禁训 DL。

### 9.1 小样本与跨站迁移（office1 919 行）

- Location-Based Probabilistic Load Forecasting of EV Charging Sites（MQ-TCN，2024）。模型：多分位数时序卷积网络，用 Caltech/JPL 向 Office-1/NREL 做归纳式迁移。解决：数据稀缺站点负荷预测。对应：office1 样本少。是否采用：模型本身不采用（DL/TCN 禁训、超预算）；采用其“跨站预训练 + 目标站微调”的思想，换成 pooled LightGBM/ExtraTrees + 站点特征 + 目标站标定。
- Probabilistic forecasts of time and energy flexibility in BEV charging（Applied Energy 2020）。模型：分位数回归、MLP tilted loss、条件核密度，预测停车时长与能量需求的各分位点。解决：单点预测给不出不确定性，调度需要分位信息。对应：dwell MAE 高、EDF 优先级需要可靠停留估计。是否采用：是，落地为 sklearn 的 `HistGradientBoostingRegressor(loss="quantile")` 或 `GradientBoostingRegressor(loss="quantile")`，纯 CPU、预装。
- Machine learning approaches to short-term EV charging demand prediction: applicability and geographical transferability（2026）。结论：站点间可迁移性存在但不稳定，迁移前要先做站点相似度/标定。是否采用：参考，不做重迁移管线。

**落地动作**：`module_2_prediction.py` 用全站 pooled 模型 + `site` 特征作为共享底座，再对 office1 做输出标定（把站点预测向 pooled 中位数收缩，shrinkage = α·site + (1-α)·pooled，α 在验证集上选）。这是最省时的“借力”手段。

### 9.2 到达已知但需求申报缺失/迟到

- ACN-Data 的用户行为建模（NSF 公开摘要，GMM 联合建模 arrival/duration/energy）。模型：高斯混合模型同时刻画到达时刻、停留时长、交付电量的联合分布。解决：到达后无法知道真实需求与离开时间，只能从历史联合分布条件采样。对应：`last_requested_energy_kwh` 常缺失，不能当 0。是否采用：是，sklearn `GaussianMixture` 预装且轻量，用 (site, hour, weekday) 条件化抽样。
- Time Matters: A Survival Analysis of Public EV Charging Infrastructure Utilization（2026）。模型：Kaplan-Meier/生存分析刻画充电会话时长分布。解决：停留时长右删失、长尾。对应：dwell 预测高误差。是否采用：部分采用，`lifelines` 未预装，可用 numpy 手写 KM 或直接用分位数回归替代；只作诊断，不接在线。

**落地动作**：保留现有 last_requested 分桶双分支；当缺失时用条件分位点/条件经验分布替代单点均值，给 EDF 一个保守（偏短）的停留估计，避免漏充。

### 9.3 轨迹推演中的停留与剩余需求不确定性

- Probabilistic forecasts of time and energy flexibility（2020）：用分位点（deciles）生成多情景，再据此做时移充电。解决：确定性轨迹忽略不确定性。对应：`module_3_world_model.py` 的 rollout 目前是单点未来到达。是否采用：是，但要省 CPU：未来到达用 1-3 个分位情景（p10/p50/p90）或从条件经验分布抽 1 条路径，不做大批量蒙特卡洛。
- ACN-Data GMM 联合建模：未来到达的 demand/dwell 从条件 GMM 抽样。是否采用：可作 9.2 的 GMM 复用。

**落地动作**：`module_3_world_model.py` 在注入未来车辆时，用 `predict` 的分位输出或条件经验分布抽 dwell/demand，而不是固定点值；保持 natural 与 capacity_edf 同一套采样。

### 9.4 容量受限下 EDF 与价格感知如何兼顾完成率

- Liu 1973 调度 + Constrained/Smoothed Least Laxity First（LAPSE 2023）。模型：最小松弛优先，laxity = 剩余时间 - 满功率完成所需时间。解决：EDF 只看截止，LLF 同时看剩余需求与剩余时间，容量受限下完成率更好。对应：`module_5_decision.py` 的 act。是否采用：是，直接把优先级从“剩余停留步数”改成“laxity = remaining_steps - remaining_energy/(charger_limit·dt)”，一行级改动，完成率提升最直接。
- Optimizing EV charging schedules based on probabilistic forecast of individual mobility（2022）。模型：分位预测 + 时移智能充电。解决：在保证完成下把充电移到低价段。对应：当前 TOU 感知 EDF。是否采用：是，保持“先 LLF 保完成，再在松弛为正的车里做价格延后”。
- Adaptive Charging Networks（Lee et al. 2021）：MPC + 凸优化在线调度，EDF/LLF 为基线。是否采用：参考，作为 explain 里的对照。

**落地动作**：`module_5_decision.py` 的排序键改为 LLF（laxity），价格延后只作用于 laxity 为正且需求小的车；改完用 `public_self_eval.py` 看三站完成率是否守住 Caltech 0.85/JPL 0.65/Office 0.85。

### 9.5 一天内可落地的轻量清单（按投入排序）

1. LLF 替换 EDF 排序键（P0，几分钟）：最直接提完成率，尤其 Office。
2. 全站 pooled + 站点标定收缩，治 office1 小样本（P0）。
3. demand/dwell 分位数回归（sklearn quantile GBM），替代/补充单点 ExtraTrees（P1）。
4. 未来到达注入改用条件分位情景（P1）。
5. GMM 联合建模做需求/停留条件抽样（P2，有富余时间再做）。
6. 生存分析/KM 仅作离线诊断（P2）。

## 10. 给模块会话的第五轮派发

- module_2_prediction：pooled 底座 + office1 标定收缩；demand/dwell 加分位输出。
- module_3_world_model：未来到达按条件分位/经验分布注入，natural 与 capacity_edf 同采样。
- module_5_decision：优先级改 LLF（laxity），价格延后只作用于正松弛车辆，并回归三站完成率。
- module_1_data / module_4_intervention：本轮无新增。

## 11. 当前最需要的论文方向清单（可执行版）

> 给各模块会话直接照做；每条 = 论文方向/关键词 → 解决的当前问题 → 对应模块/文件 → 紧急度 → 剩余赛程内可落地的具体做法。

1. Least Laxity First / deadline feasibility 调度 → 低容量下 just_in_time 压功率可能牺牲完成率，tight_ahead 只判“未来更紧”没保护“现在已紧” → `module_5_decision.py` → P0 → 排序键改为 `laxity = remaining_steps - remaining_energy/(charger_limit·dt)`；laxity 小者优先满充到 deadline 可完成；价格延后只作用于 laxity>0 且需求小的车；加“现在已紧”保护（剩余需求/剩余步数超阈值就立即满充）。改完跑 `public_self_eval.py` 逐站核对完成率。

2. Deadline 约束下的 TOU / 价格感知充电优化 → 价格延后与完成率硬约束冲突 → `module_5_decision.py` → P0 → 先按 LLF 分配保完成基线功率，剩余容量/正松弛车辆再按电价梯度延后；绝不因价格延后 laxity≤0 的车；做成可开关后处理，完成率回退即关闭。参考 2022 分位预测 + 时移充电。

3. 稀疏站点 / 迁移学习的 demand & dwell 预测 → office1 仅 919 行、last_requested 有效 0 条、demand MAE 6.24 → `module_2_prediction.py` → P0 → 全站 pooled LightGBM + `site` 特征共享底座，office1 输出向 pooled 中位数收缩（`α·site + (1-α)·pooled`，α 验证集选）；缺失分支不依赖 last_requested，改用条件经验分布/分位。

4. 分位数回归 + 生存分析做 dwell/demand 不确定性量化 → dwell MAE 1.56h 偏高、单点预测无区间 → `module_2_prediction.py`、`module_3_world_model.py` → P1 → sklearn `HistGradientBoostingRegressor(loss="quantile")` 出 p50/p75/p90；dwell 偏右可试 log1p；生存分析/KM 仅离线诊断（lifelines 未预装，numpy 手写）。

5. 容量干预反事实负荷预测 → 干预要求同时优于零效应与中位数双基线 → `module_4_intervention.py` → P1 → 与 natural 共用同一 rollout、同一需求/到达/离开采样；capacity_edf 用 EDF/LLF 一致策略；未来到达注入用条件分位情景；不复制 natural 压平效应。

6. 完成率本地代理验证 → 世界模型还缺完成率回归 → `module_3_world_model.py` / `public_self_eval.py` → P1 → 本地回放逐站统计“剩余未交付 ≤ 3% 目标电量”车辆占比，核对 Caltech≥0.85 / JPL≥0.65 / Office≥0.85；改 act 后必跑。

7. GMM 联合建模 arrival/duration/energy → 未来到达的 demand/dwell 联合分布 → `module_3_world_model.py` → P2 → sklearn `GaussianMixture` 条件抽样，有富余时间再做。

执行顺序：1、3 先做（P0），2 紧随；4、5、6 为 P1；7 有富余再做。所有改动后必须跑一遍完成率代理验证，守住三站门槛。

## 12. P0 论文库与合规映射（2.1 轮）

| 论文 | 类型 | 合规性 | 对应模块/文件 | 2.1 落地动作 |
| --- | --- | --- | --- | --- |
| IEEE Access 2024 需求预测综述 | 综述 | 合规，仅作方法论基线 | `module_1_data.py`、`module_2_prediction.py` | 指导 GBM/分位/迁移选型，不直接写代码 |
| MQ-TCN 2024（跨站迁移） | 深度迁移 + 多分位 TCN | 禁训 DL，仅借鉴迁移思想 | `module_2_prediction.py` | pooled LightGBM + `site` 特征 + office1 标定收缩 |
| 2026 地理可迁移性 | ML 六类方法比较 | 合规参考 | `module_2_prediction.py` | 迁移前做站点相似度/标定，不盲目迁移 |
| Zhang 2020 分位数组合密度预测 | 分位预测组合 | 合规 | `module_2_prediction.py`、`module_3_world_model.py` | 多分位输出组合成 demand/dwell 区间，注入 rollout |
| Lee 2021 ACN | 凸优化 + MPC 调度 | 合规，轻量实现 | `module_5_decision.py` | MPC 前瞻 + 过订阅分配，EDF/LLF 基线 |
| LAPSE 2023 GA 调度 | 遗传算法元启发式 | 仅离线参考 | 无 | 不接在线路径，可作离线对照 |
| Liu 1973 | EDF/LLF 经典调度 | 合规 | `module_5_decision.py` | LLF 排序键落地 |
| Energy Informatics 2023 MPC 负荷塑形 | MPC 约束优化 | 合规 | `module_4_intervention.py`、`module_5_decision.py` | capacity_edf 的 MPC/负荷塑形对照 |
| Cai 2022 分位预测 + 时移充电 | 分位回归 + 调度 | 合规 | `module_2_prediction.py`、`module_3_world_model.py`、`module_5_decision.py` | demand/dwell 分位 + 价格时移 |

2.1 落地顺序：`module_5_decision` 的 LLF（P0）→ `module_2_prediction` 的 pooled + office1 标定（P0）→ demand/dwell 分位数（P1）→ `module_3_world_model` 分位注入（P1）→ `module_4_intervention` 保持双基线合规并跑完成率代理验证。

## 13. 2.1 P0 可落地方案

### P0-1 小样本 + 跨站迁移 + 分位数回归（→ 数据处理、预测）

- 特征：`sin/cos(hour)`、`weekday`、`site` 独热、`charger_limit_kw`；`last_requested_energy_kwh` 保留“缺失指示 + 有效值”双分支，但缺失时不得当 0、不得当目标。
- 跨站迁移：训练一个全站 pooled LightGBM（含 `site` 特征），再为 office1 做输出标定收缩：`pred_office = α·site_model + (1-α)·pooled_model`，α 在 office1 时间序 holdout 上从 `{0,0.3,0.5,0.7,1}` 选；或做乘性偏差校正 `scale = mean(demand)/mean(pred)`。
- 分位数回归：sklearn `HistGradientBoostingRegressor(loss="quantile")` 分别出 demand/dwell 的 q50/q75/q90；dwell 试 `log1p` 目标再反变换；q50 做点预测，q75/q90 给调度保守值。
- 验收：office1 demand MAE 低于 6.2444、dwell MAE 低于 1.5617；仍按时间序切分，不随机打散。

### P0-2 最小松弛优先（LLF）在线调度（→ 充电决策）

- 每辆在场未完成车计算：`remaining_energy = max(demand_pred - delivered, 0)`、`steps_left = max(1, ceil(dwell_pred/dt - elapsed_steps))`、`time_to_finish = remaining_energy / (charger_limit_kw·dt)`、`laxity = steps_left - time_to_finish`。
- 按 `laxity` 升序分配功率（laxity 小者最紧急），每辆最多给 `min(charger_limit, remaining_energy/dt)`，直到可用容量用完。
- “现在已紧”保护：任一辆 `laxity ≤ 0` 或 `laxity ≤ safety_margin`（如 1 步）时立即满充，不允许价格延后；价格延后只作用于 `laxity > threshold` 的车辆。
- 参考 Lee 2021 ACN 的 MPC 为可选 P2，EDF/LLF 作为基线；验收用 `public_self_eval.py` 核对三站完成率不下降。

### P0-3 先保完成基线、再削峰（→ 充电决策）

- 两阶段：先按 LLF/EDF 算出“保证在预测停留内完成”的基线功率；再把正松弛车辆的富余电量按电价从高到低延后到低价步，绝不把 `laxity≤0` 的车辆延后。
- 约束优先：完成率硬约束 > 峰值/成本目标；Energy Informatics 2023 的目标 `min ||目标负荷 - 实际负荷||²` 只在完成基线满足后作用于剩余灵活性。
- 用 2022 概率预测思路：dwell/energy 的分位预测决定“多少松弛可以安全被移动”。
- 验收：先看完成率不回退，再看峰值/成本改善；二者冲突时以完成率为准。
