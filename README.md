# AI FOR GREEN

Competition repository for the AI FOR GREEN challenge.

## 本地开发环境

评测使用官方隔离 CPU 沙箱 `cpu-ml-methods-v3`，不联网、不装依赖。本地开发请按
同一组固定版本安装，避免 pickle/joblib 或 LightGBM 模型跨版本不兼容：

```bash
python -m pip install -r requirements.txt
```

关键版本：`lightgbm==4.6.0`、`xgboost==3.1.3`（官方包名为 `xgboost-cpu`，import
仍为 `xgboost`）、`scikit-learn==1.4.1.post1`、`numpy==1.26.4`、`pandas==2.1.4`。
离线预测权重由 `python train_module2_weights.py` 生成到 `weights/`；沙箱内只加载，
不重新训练。

## 加入协作（第一次）

1. 接受邀请：打开 https://github.com/nociloXXX/AI-FOR-GREEN/invitations ，点 Accept。
2. 装 Git（Windows）：https://git-scm.com/download/win ，一路默认安装。
3. 打开终端，克隆仓库：

   ```bash
   git clone https://github.com/nociloXXX/AI-FOR-GREEN.git
   cd AI-FOR-GREEN
   ```

4. 第一次配置身份：

   ```bash
   git config --global user.name "你的名字"
   git config --global user.email "你的邮箱"
   ```

## 日常提交

```bash
git pull
# 改代码
git add .
git commit -m "说明这次改了什么"
git push origin main
```

第一次 push 会弹出浏览器让你登录 GitHub，登录一次后以后就不会再问。

## 小提示

- 开始工作前先 `git pull`，避免冲突。
- 如果 push 被拒，先 `git pull` 再 push。
- 不习惯命令行的话，可以装 GitHub Desktop（https://desktop.github.com/ ）用图形界面克隆和提交。

## 基线模块结构

1. 数据处理：`module_1_data.py`
2. 预测：`module_2_prediction.py`
3. 世界模型：`module_3_world_model.py`
4. 干预推演：`module_4_intervention.py`
5. 充电决策：`module_5_decision.py`

提交入口是 `submission.py` 的 `build_model(training)`。本地先跑接口冒烟
`python verify_submission.py`，再跑公开学习指标
`python public_self_eval.py --submission . --training data/training_public.json --out public-metrics.json`；
分站口径可跑 `python view_official.py`。打包用
`python make_submission.py --source . --out my_submission.zip --include weights/*.txt`。
