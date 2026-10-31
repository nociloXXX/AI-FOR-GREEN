# AGENTS.md

本文件供进入本仓库的 AI 协作代理（Codex、Claude Code 等）阅读，帮助它们快速理解：项目是什么、已经做了什么、关键约束是什么。它不会被打进比赛提交包。

## 项目

AI FOR GREEN（Track A 技术评分路线）：充电站在线模型竞赛。评测沙箱只调用 `submission.py` 的 `build_model(training)`，返回对象必须实现五个方法：`predict`、`arrival_rate`、`act`、`forecast`、`explain`。

`training` 是字典 `{site: pandas.DataFrame}`，站点为 `caltech`、`jpl`、`office1`；训练列包括 `hour`、`weekday`、`charger_limit_kw`、`demand_kwh`、`dwell_hours`、`day`。

## 最近做了什么（截至 2026-10-31）

- 第一轮优化已完成，标签 `v0.1` 指向 commit `e4e8a55`。
- 当前 `main` 最新 commit 为 `6e16a31`（2026-10-31 14:55）。历史里大量 `auto sync ...` 提交来自本机每 5 分钟的自动同步，不代表每次都有语义改动。
- 已用官方公开训练数据离线训练 LightGBM 双分支权重到 `weights/`：申报需求有值时用 `demand_with_*` / `dwell_with_*`，缺失或非法时用 `fallback_*`。
- 公开学习指标（仅预测留出集，非官方成绩）：`public-metrics.json` 中 `demand_kwh` MAE 约 3.877、`dwell_hours` MAE 约 1.561。

## 架构

- `submission.py`：唯一入口，组装五模块在线模型。
- `module_1_data.py`：数据处理；`module_2_prediction.py`：预测；`module_3_world_model.py`：世界模型；`module_4_intervention.py`：干预推演；`module_5_decision.py`：充电决策。
- `acn_causal/student_api.py`：官方接口与数据结构。
- `weights/`：LightGBM 文本权重，沙箱内只加载不训练；缺失时现场回退 LightGBM，再退 ExtraTrees。
- `docs/`：接口、评分、时序数据、提交指南等详细说明。
- `backup/` 与 `.autosync/`：本机自动备份模块和本地日志，不属于提交内容。

## 常用命令

```bash
# 接口冒烟
python verify_submission.py

# 公开学习指标
python public_self_eval.py --submission . --training data/training_public.json --out public-metrics.json

# 训练预测权重
python train_module2_weights.py

# 分站口径
python view_official.py

# 打包提交
python make_submission.py --source . --out my_submission.zip --include weights/*.txt
```

## 关键约束（改代码前必读）

- 沙箱 `cpu-ml-methods-v3` 离线、不安装依赖；固定版本见 `requirements.txt`。
- 不要向 stdout 输出调试信息（评分协议占用 stdout），调试信息写 stderr。
- 不允许使用未来观测、隐藏标签、离开时间或记忆测试场景。
- 逐场景完成率门槛：Caltech ≥ 0.85、JPL ≥ 0.65、Office ≥ 0.85；功率必须物理合法。
- `make_submission.py` 只打包白名单文件，不要把 `data/` 或官方禁用库打进提交包。
- 保持 `submission.py` 的 `build_model` 契约和 `explain.json` 结构。

## 给新 AI 的起点

先读 `docs/SCORING.md` 和 `docs/INTERFACE.md`，再读 `submission.py` 与五个模块；最后跑 `python verify_submission.py` 确认环境可用。
