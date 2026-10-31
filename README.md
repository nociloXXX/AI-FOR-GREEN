# AI FOR GREEN

Competition repository for the AI FOR GREEN challenge.

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

1. 数据处理：`src/data_loader.py` + `src/features.py`
2. 预测：`src/predict.py`
3. 世界模型：`src/state_model.py`
4. 干预推演：`src/intervention.py`
5. 充电决策：`src/decision.py`

端到端入口是 `src/pipeline.py`，运行 `python -m src.pipeline`；运行效果报告用
`python -m src.report` 生成到 `submissions/report.html`。
