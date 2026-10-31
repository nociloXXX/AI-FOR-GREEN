# A 赛道《停车场充电世界模型》算法与论文调研

> 交付对象：数据处理、状态建模、预测、决策四个模块的执行会话。
> 阅读方式：先看「总排序」，再跳到对应模块的小节。结论已按“队友直接照做”的口径写，不展开公式推导。
> 调研日期：2026-10-31（GMT+8）。

## 0. 一句话结论

现有模板已经搭对了骨架。一天赛程内不要换架构，而是按下面的顺序把每个模块的
“主模型 + 兜底 + 可解释报告”三层补齐：

1. 预测：LightGBM 为主，XGBoost/HistGradientBoosting 兜底，Ridge/Naive 保底；Prophet 只作可选加权融合。
2. 决策：因果削峰填谷为默认在线策略；评测若允许用预测值做前瞻，再上 MPC 滚动优化；LP 注水法只用于离线报告；MILP 在 PuLP 可装且问题规模小时作为离线精确解；贪心永远兜底。
3. 状态建模：保留 `StationState` 确定性状态机作主线，M/M/c 与 M/G/c 做等待时间/占用率诊断，马尔可夫与蒙特卡洛只做离线情景分析。
4. 数据处理：保留现有 IQR/3σ/IsolationForest，补日历、滞后、滚动与电价/天气对齐；不要引入 GAN 或生成式数据增强。
5. 强化学习（DQN/PPO）：只在四条主线全部跑通且时间有余时，作为展示项接入，不作为得分主路径。

## 1. 总排序（一天赛程内的投入优先级）

| 优先级 | 方向 | 对应模块 | 当天是否采用 | 一句话理由 |
| --- | --- | --- | --- | --- |
| P0 必做 | LightGBM / XGBoost 负荷预测 | 预测 | 是 | 表格型特征、训练快、对日历与滞后特征天然友好，是 EV 充电负荷文献里的主流上分模型 |
| P0 必做 | 因果削峰填谷（零前瞻） | 决策 | 是 | 不碰未来信息、解释性强、能保证在线评测不翻车，模板已实现 |
| P0 必做 | 规则状态机 + M/M/c 诊断 | 状态建模 | 是 | `StationState` 是唯一能在在线评测中安全推演状态的确定性底座，排队论负责解释 |
| P1 强烈建议 | MPC 滚动优化（用预测值做小步长前瞻） | 决策 | 视评测规则 | 文献里的主流做法（ACN 等），能把预测模块的产出直接变现为削峰收益 |
| P1 强烈建议 | LP 注水法 / 电价加权平移 | 决策 | 仅离线 | 是连续功率分配的闭式解，可当 MPC 每一步的求解器，也是离线报告的上界 |
| P1 强烈建议 | HistGradientBoosting / Ridge / Naive 多级兜底 | 预测 | 是 | 保证断网、装不上 lightgbm 时链路仍闭环，模板已实现 |
| P2 有把握再做 | MILP（PuLP / OR-Tools）整数调度 | 决策 | 可选 | 精确处理“开/关桩”的整数约束，但规模一大就慢；小场站离线可以用 |
| P2 有把握再做 | Prophet 季节项融合 | 预测 | 可选 | 对强日/周周期与节假日有用，单独用不如 GBM，适合做加权融合 |
| P3 仅展示 | ARIMA / SARIMA | 预测 | 否（或只作基线） | 闭式可解释但精度通常不如表格特征 GBM，且多场站要逐站建模 |
| P3 仅展示 | DQN / PPO 强化学习 | 决策 | 否（有时间再上） | 文献证明可行，但调参、奖励设计、复现成本高，一天内风险大 |
| P3 仅展示 | 马尔可夫链 / 蒙特卡洛 | 状态建模 | 可选 | 稳态与分布推断有价值，但属于“解释性加分项”，不是得分关键路径 |

## 2. 与现有模板的对应关系

| 模块 | 文件 | 现状 | 推荐动作 |
| --- | --- | --- | --- |
| 数据处理 | `src/data_loader.py`、`src/features.py` | 已有 IQR/3σ/IsolationForest、时间/滞后/滚动特征、电价天气对齐、恒功率+尾部衰减的负荷曲线重构 | 基本保留；开题后只调 `config.py` 的字段映射、`LAGS` 与异常检测阈值 |
| 状态建模 | `src/state_model.py` | 已有 `StationState` 确定性状态机、M/M/c、M/G/c、CTMC、蒙特卡洛，且全部只接受到达率/分布参数，不读未来事件 | 主线不动；把 `MMcQueue.estimate()` 的结果接进报告，排队指标作为“世界模型可解释性”输出 |
| 预测 | `src/predict.py` | 已有 LightGBM → XGBoost → HistGB → Ridge → Naive 的按可用性降级链 | 首选装 `lightgbm`；Prophet 只作可选融合；不要上 LSTM/Transformer |
| 决策 | `src/decision.py` | 已有 `causal_load_shift`（默认）、`lp_load_shift`、`mpc_schedule`、`greedy_priority`、`milp_schedule`（PuLP）、`battery_arbitrage` | 默认保持 causal；评测允许时切 `DECISION_STRATEGY="mpc"`；离线报告用 LP；MILP 视 PuLP 安装情况 |

## 3. 分模块结论

### 3.1 数据处理

**模型是什么**
处理模块负责把原始充电会话表变成“场站级 15 分钟负荷曲线 + 可训练特征”，并剔除异常、补齐缺失。文献里这一层通常叫“数据预处理 + 特征工程”，不是独立的预测模型。

**解决什么问题**
- 原始会话表是“事件流”，不能直接进回归器，需要重采样成规则时间网格。
- 异常会话（SOC 越界、功率利用率 > 1、时间倒挂）会污染预测和调度。
- 在线评测隐藏未来数据，任何用 `shift(-1)`、未来滚动、整段归一化的特征都会造成数据穿越，直接毁掉成绩。

**与本赛题的对应关系**
你们已实现：规则清洗保底、缺电量反推、利用率过滤、IQR/3σ/IsolationForest 异常检测、严格时间序切分、`shift` 正向滞后 + `shift(1).rolling` 的因果滚动、外源电价/天气前向对齐。这已经覆盖文献里的标准做法，方向正确。

**推荐顺序与理由**
1. 保留 IQR/3σ 异常检测（P0）。IQR 对偏态负荷稳健，3σ 适合近似正态的利用率；IsolationForest 在有 sklearn 时可选，不是必需。
2. 时间 + 滞后 + 滚动特征必做（P0）。15 分钟粒度下 `96` 步 = 1 天、`24` 步 = 6 小时，`LAGS=(1,4,24,96)` 合理；开题后按官方粒度核对，如果官方是 5 分钟，就把 `96` 对应到 8 小时而不是 1 天。
3. 外源特征（电价档位、温度、周末）做轻量对齐（P0/P1）。已有 `merge_external`，开题后只要把官方列名填进 `config.py` 的 alias。
4. 不做 GAN / VAE 数据增强、不做复杂缺失值插补模型（P3）。一天内收益低、风险高。

**怎么接进现有模板**
- 开题后改 `config.py`：`RAW_COLUMN_ALIASES`、`PRICE_COLUMN_ALIASES`、`WEATHER_COLUMN_ALIASES`、`LAGS`、`ROLLING_WINDOW`、`IQR_K`、`UTILIZATION_MAX`。
- 不要动 `build_station_load_curve` 的“电量守恒”逻辑，真实功率曲线只能在此基础上替换 `_session_power_profile`，否则预测和调度会口径不一致。

### 3.2 状态建模

**模型是什么**
- 规则状态机：用桩占用/空闲、每车 SOC、剩余电量、排队长度、变压器可用容量、电价档位组成的确定性系统，每步按规则转移。
- M/M/c 队列：泊松到达 + 指数服务时长，Erlang-C 闭式给出等待概率、平均队长、平均等待时间。
- M/G/c 队列：服务时长服从一般分布，用变异系数平方（cv²）修正 M/M/c 的结果。
- 连续时间马尔可夫链（CTMC）：小状态空间的“占用桩数”生灭过程，可算稳态分布与有限时段转移概率。
- 蒙特卡洛：按到达率与服务时长分布采样，估计期望等待、队长、占用率、拒绝率。

**解决什么问题**
- 规则状态机解决“在线评测中我们只能看到当前步到达时，如何可复现地推演场站状态”的问题。
- 排队论解决“用很少的参数（到达率 λ、服务率 μ、桩数 c）快速回答等待多久、排多长队、桩会不会不够用”的规划问题。
- CTMC / 蒙特卡洛解决“小规模场站的稳态占用分布与长时情景分析”问题，适合路演解释。

**与本赛题的对应关系**
`state_model.py` 已完整实现上述四类模型，且刻意保持“在线不读未来”的约束。这正好对齐赛题“停车场充电世界模型”的定位：`StationState` 是执行层底座，排队论/马尔可夫/蒙特卡洛是观察层。

**优缺点**
- 优点：确定性强、可解释、无第三方依赖、天然满足在线因果性。
- 缺点：M/M/c 假设指数服务，真实充电时长不是指数分布，绝对数值会有偏差；CTMC 状态空间随排队容量增长，只适合小规模；规则状态机对复杂电价/分时策略的表达能力有限。

**一天内是否值得采用**
值得，但只做“保留 + 接报告”。不要重写状态机，也不要把主决策逻辑改成马尔可夫决策过程（MDP）。M/M/c 与 M/G/c 的等待指标是低成本、高解释性的加分项。

**怎么接进现有模板**
- 主线保留 `StationState.step` 做在线推演。
- 在报告里对每个场站调用 `MMcQueue(c, λ, 平均服务小时).estimate()`，输出 `utilization`、`probability_wait`、`mean_wait_hours`；服务时长明显非指数时改用 `MGcQueue` 并传入实测 `cv²`。
- 离线情景分析才用 `MarkovOccupancyModel` 与 `MonteCarloSimulator`，不要接在线链路。

### 3.3 预测（负荷预测）

**模型是什么**
- LightGBM / XGBoost：梯度提升决策树（GBDT）的高性能实现，用表格特征（时间、滞后、滚动、日历、电价、天气）直接回归下一时段场站功率。
- HistGradientBoosting：scikit-learn 的直方图 GBDT，作为无 lightgbm/xgboost 时的等价兜底。
- Ridge / Naive：线性正则与分小时均值，作为可解释保底基线。
- ARIMA / SARIMA：统计时间序列模型，对单序列做自回归/差分/季节项。
- Prophet：分解为趋势 + 季节 + 节假日的可加模型，对强日/周周期和缺失值友好。

**解决什么问题**
预测未来一段时间的场站充电负荷，给调度模块提供“未来会涨还是会降”的信号。调度质量的上限受预测误差直接影响：预测越准，MPC/LP 削峰越不误伤。

**与本赛题的对应关系**
`predict.py` 的降级链已经是对的。文献里 EV 充电负荷预测的主流结论是：表格型特征场景下 GBDT（LightGBM/XGBoost）通常优于 ARIMA/SARIMA；Prophet 对强季节性和节假日有优势；LSTM 等深度学习需要更多数据和调参，短赛程收益不稳定。

**推荐顺序与理由**
1. LightGBM（P0）：训练快、内存小、对滞后/日历特征友好，文献与工业界默认首选。
2. XGBoost（P0/P1）：精度与 LightGBM 接近，作为第二主模型或交叉验证后的选择。
3. HistGradientBoosting（P1）：装不上 lightgbm/xgboost 时的兜底，已实现。
4. Ridge / Naive（P1）：可解释基线与最终保底，保证链路闭环。
5. Prophet（P2）：单独用通常不如 GBM；如果时间允许，做 `0.8 * LightGBM + 0.2 * Prophet` 的验证集加权融合，再决定是否保留。
6. ARIMA / SARIMA（P3）：仅作“无表格特征”时的诊断基线，多场站要逐站建模，不推荐当主模型。
7. LSTM / Transformer / GAN（P3）：跳过。一天内调参和复现成本高，不如把时间花在特征和调度上。

**优缺点（GBM 主线）**
- 优点：表格特征表达力强、训练快、自带特征重要性、不需要长序列、对缺失值稳健（LightGBM 原生支持）。
- 缺点：点预测不带不确定性；对极端峰值可能欠拟合；参数（叶子数、学习率、正则）需要小规模网格搜索。

**怎么接进现有模板**
- 安装依赖：`pip install lightgbm xgboost`，并把它们加进 `requirements.txt`（保留 fallback，装不上也不崩）。
- 保持 `train_predictor` / `predict` 接口不动；开题后核对目标列 `C.LOAD_COL` 与官方口径。
- 训练/验证继续用 `chronological_split` 时间序切分，严禁随机切分。
- 可选：在 `_build_features` 增加 `price_lag`、`temperature_lag`、`price_tier` 特征，但先测有无提升再上。

### 3.4 决策（有序充电调度）

**模型是什么**
- 因果削峰填谷：第 t 步只依赖 0..t 已见数据，超容电量记入积压，有裕量时消化积压。零前瞻、在线安全。
- MPC（模型预测控制）：滚动窗口内求解一个优化问题，只取第一步执行，下一时刻重新求解；前瞻来自预测模块的预测值，不是未来真实数据。
- LP 注水法 / 电价加权平移：连续功率分配问题 `min Σ(d_i - p_i)²` 在容量约束下的闭式解，能量守恒；加电价后可把峰段电量优先搬到谷段。
- MILP：引入 0/1 开关变量精确表示“桩是否开启”与整数约束，用 PuLP（CBC）或 OR-Tools（CP-SAT/SCIP）求解。
- 贪心规则：FCFS / 低 SOC 优先 / 最早离场优先，兜底。
- 强化学习（DQN/PPO）：把调度建模成 MDP，学习状态到动作的策略。

**解决什么问题**
在变压器容量约束内，尽量满足每辆车的充电需求，同时削峰填谷、降低电费与碳排；在线评测下还必须保证不使用未来信息。

**与本赛题的对应关系**
`decision.py` 已经按复杂度分层实现，且明确区分“在线安全 / 离线专用”。文献里的主流做法与之一致：MPC + 线性规划是 EV 充电调度的标准范式（如 Adaptive Charging Network，Lee et al. 2021，把在线调度与离线完美信息上界对比；Energy Informatics 2023 的 MPC 用二次误差目标 + 每车功率/能量/SOC 约束做楼宇负荷塑形）。MILP 是“开/关桩”这类整数约束的精确解，但属 NP-hard，规模大时要用启发式。

**推荐顺序与理由**
1. `causal_load_shift`（P0，默认）：在线安全、可解释、能量守恒，保证评测不翻车。
2. `mpc_schedule`（P1，评测允许时）：把预测值变成前瞻收益，是文献证明有效的主路径；窗口取小（`MPC_HORIZON_STEPS=4`，即 15 分钟粒度下 1 小时）。
3. `lp_load_shift`（P1，仅离线）：作 MPC 每一步的求解器（模板已是这么接的），也是离线报告里的“完美信息上界”。
4. `milp_schedule`（P2，视 PuLP）：小场站、离线分析时给出精确开/关桩解；在线评测建议不用，求解时间不可控。
5. `greedy_priority`（P2 兜底）：无第三方库时的最后保障，模板已实现。
6. `battery_arbitrage`（P2，若赛题含储能）：模板已做因果滚动阈值版本，储能不在赛题范围内就保留但不用。
7. DQN / PPO（P3，仅展示）：把策略接口包成 `state → per-pile action` 后可以接入，但一天内不要作为得分主路径。

**优缺点（MPC/LP 主线）**
- 优点：优化目标明确、约束可写可解释、能量守恒、与预测模块解耦清晰、离线 LP 给上界。
- 缺点：连续 LP 不能精确表示“开/关桩”；MILP 规模受限；MPC 收益依赖预测质量；在线零前瞻约束下只能用 causal 版本。

**怎么接进现有模板**
- 在线默认保持 `DECISION_STRATEGY="causal"`。
- 评测明确允许“使用自己预测的未来值做决策”时，把 `config.py` 改为 `DECISION_STRATEGY="mpc"`，并保持 `MPC_HORIZON_STEPS` 小。
- 离线报告统一用 `lp_load_shift` 计算削峰上界，和在线 causal 结果并列展示“可达上界 vs 在线安全结果”。
- PuLP 可装时，对每个场站做一次离线 `milp_schedule` 作为对照实验；装不上就自动回退贪心，不影响提交。

## 4. 论文要点速查

- EV 充电负荷预测综述：GBDT（XGBoost/LightGBM/CatBoost）与 LSTM 是两大主流，表格特征场景 GBDT 常更稳；ARIMA/Prophet 常作统计基线或混合组件。
- 调度综述：LP/MILP 与启发式是主流，MILP 表达“开/关桩”精确但规模敏感。
- Adaptive Charging Network（ACN）：EV 充电调度的经典在线框架，用预测 + 在线调度对比离线完美信息上界，文献里被广泛引用为基线。
- MPC 楼宇负荷塑形：每步求解 `min ||目标负荷 - (楼宇负荷 + 各车功率 + 储能功率)||²`，约束为每车功率上限、能量上限、储能 SOC；5 分钟步长，滚动只取第一步。这与模板 `mpc_schedule + lp_load_shift` 的结构一致。
- M/M/c：EV 场站等待时间与选址定容常用模型，Erlang-C 闭式解；服务时长非指数时用 M/G/c 或仿真校正。
- DRL（DQN/PPO）：EV 充电调度系统综述表明可行，但调参、奖励设计、样本效率与复现成本高，适合研究性展示而非一天赛程主路径。

## 5. 给执行模块的一句话指令

- 数据处理：开题后只改 `config.py` 字段映射与粒度参数，不动“电量守恒 + 因果滞后”的既有逻辑。
- 状态建模：保留 `StationState`，把 M/M/c（必要时 M/G/c）指标接进报告；马尔可夫/蒙特卡洛仅离线。
- 预测：优先装 `lightgbm` 走 GBM 主线，保留现有降级链；Prophet 只做可选融合，不上 LSTM/Transformer。
- 决策：在线默认 `causal`，评测允许前瞻再切 `mpc`；离线报告用 `lp_load_shift` 给上界；MILP 视 PuLP 可用性做对照，RL 有时间再上。

## 6. 参考来源

- IEEE Access, "A Comprehensive Survey of Electric Vehicle Charging Demand Forecasting Techniques" (2024).
- IEEE Access, "Exploring AI-Driven Approaches for Predicting Electric Vehicle Charging Behaviour: A Literature Review" (2026).
- Energy Informatics (Springer), "Using EV charging control to provide building load flexibility" (2023), DOI 10.1186/s42162-023-00261-8.
- Lee et al., "Adaptive Charging Networks: A Framework for Smart Electric Vehicle Charging"（ACN 框架）。
- IEEE, "Location Planning of Fast Charging Infrastructure for Electric Vehicles and System Resilience Analysis via a Demand-Based Queueing Model" (2025)，M/M/c 等待时间应用。
- arXiv 2410.16425, "Advancements in Electric Vehicle Charging Optimization: A Survey of Reinforcement Learning Approaches".
- Transportation Research Part E, "Reinforcement learning for electric vehicle charging scheduling: A systematic review" (2024).
- 坦佩雷大学硕士论文, "Comparison of predictive models for time series forecasting: Power demand prediction of DC charging network" (2024)，SARIMA/XGBoost/Prophet/LSTM 对比。
