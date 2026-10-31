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
