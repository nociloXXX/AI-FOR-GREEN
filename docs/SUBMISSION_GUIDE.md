# 提交指南

## 最快试一次

1. 解压本参赛包，找到 `baseline_submission.zip`。
2. 登录你的队伍账号，在网站点击「上传并评分」，选择这个小ZIP。
3. 等待排队、评测，查看提交记录和聚合分数。

本参赛包不是提交包。不要直接把完整开发包 ZIP 或本参赛包传上去。组委会需要先开放提交；关闭时网站会拒绝上传。

参考提交是CPU学习模型：ExtraTrees96 + 星期/小时到达率 + 基于当前观测的物理预测与EDF策略。参考提交用于验证接口和开发流程；在当前输入信息范围（last-request 快照）下，其参考成绩以网站评测为准。这不保证最终测试通过或分数相同；Office 完成门槛仍可能 DQ（不进入合格技术榜）。固定中位数是评分对照，不等同这个学习参考。

## 技术路线

可使用自己实现的 CPU 机器学习或神经网络模型（含 MLP）。报名 Track A / Track B 不变；本指南的模型评分只用于 Track A。

- ML：保留原五接口及 ExtraTrees96 样例，支持自有机器学习或神经网络模型提前训练并加载权重；评测使用统一离线 CPU 隔离沙箱，不提供 GPU 加速。

## 提交 ZIP 文件结构

根目录必须有 `manifest.json`、`README.md`、`explain.json`。manifest 声明 route、contract_version、environment_id、seed，并记录其余所有文件 SHA-256。用脚本生成哈希，不手填示例占位值。网站重新校验并保留不可覆盖原件。

ML 增加 `submission.py` 和必要的白名单源码/权重。不自动安装依赖、联网下载或运行安装脚本。

`explain.json`：method 是字符串，assumptions / limitations 是字符串数组。baseline_submission.zip 内附与 ExtraTrees baseline 一致的例子，请按实际方法修改。

## 必须的入口

提交ZIP根目录有 `submission.py`，导出：

```python
# submission.py includes the complete MedianModel and LearnedModel classes.
def build_model(training):
    return LearnedModel(training)
```

training是字典：站点名称 -> pandas.DataFrame。站点键为caltech、jpl、office1；列为hour、weekday、charger_limit_kw、demand_kwh、dwell_hours、day，day为已解析日期。模型对象必须提供predict、arrival_rate、act、forecast、explain。字段见acn_causal/student_api.py。

根目录submission.py已包含完整可运行baseline：训练、predict、arrival_rate、act、forecast、explain都能直接读和改。它是当前在线接口的CPU学习参考。

你可以改本文件的模型类/训练特征/策略/预测，或将LearnedModel换成你的模型类；模型可读取提交目录中的相对路径权重文件。不允许预测时读取未来观测、离开时间或评测真值。stdin/stdout由评分协议使用，不要print调试文字到stdout；调试信息用stderr。无外网，不能在评测中pip安装。

## 自己打包

先解压 baseline_submission.zip 到单独开发目录，在该目录修改 submission.py、README.md、explain.json，然后使用本赛事的 CPU 环境ID `cpu-ml-methods-v3`（对应下面“ML 运行环境与版本”里的固定依赖）：

```sh
python make_submission.py --source . --route ml --environment-id cpu-ml-methods-v3 --out my_submission_v1.zip
python make_submission.py --source . --route ml --environment-id cpu-ml-methods-v3 --out my_submission_v2.zip --include my_model.py weights/model.joblib
```

脚本默认 ML，自动生成 manifest 文件与哈希，不执行模型、不给正式分数；不会覆盖已有输出。重新生成后的 `baseline_submission.zip` 将使用环境ID `cpu-ml-methods-v3`。环境ID `cpu-ml-v1` 与 `legacy-cpu` 的提交仍可正常上传。

不要上传整个开发包、训练数据、评测器或 venv。ML 允许 .py/.json/.txt/.md/.npz/.npy/.pkl/.joblib；20 MiB ZIP / 100 MiB 展开 / 500 条目。pickle/joblib 只在原 CPU 沙箱加载。路径穿越、链接、重复文件、压缩炸弹拒收。

## 演示材料与截止

演示文件与评分 ZIP 分开上传：**PDF / PPTX，30 MiB 内，每队最多20个版本**。`.ppt` 请先导出为 PDF。转换为静态页面后可翻页/全屏；PPTX 动画与音视频不保留，另备原件。禁止宏、嵌入对象和外部内容。

2026-11-01 09:00（北京时间）模型与演示材料截止。之后 demo 与私有终评同步进行，15:00 颁奖前完成。每队计入成绩的是通过资格门槛的最高开发分版本，不以最后一次提交为准。最终测试不下载，私有评测结果不向队伍即时公开。

## ML 运行环境与版本

网站公开开发评测与私有终测使用同一套固定 ML 沙箱依赖：

```text
numpy==1.26.4
pandas==2.1.4
scipy==1.11.4
scikit-learn==1.4.1.post1
torch==2.15.0+cpu
xgboost-cpu==3.1.3
lightgbm==4.6.0
statsmodels==0.14.6
patsy==1.0.3
packaging==26.3
```

请按以上版本开发、训练和保存模型，尤其是 pickle / joblib 模型。其他第三方包不保证可用；沙箱不联网，不安装 requirements，也不执行安装脚本。需要的纯 Python 依赖可以随 ZIP 源码提交，仍须符合源码、文件大小与禁用库规则；不能携带自定义原生扩展或用打包方式绕过限制。PyTorch CPU 版本已纳入固定评测环境。

本地 `public_self_eval.py` 是学习指标检查工具，不是官方沙箱，也不产生官方分数。请先安装上述版本；官方公开开发评测与终测的运行环境一致，但使用不同数据，因此不承诺两次分数相同。

基准包以网站“可直接上传的 baseline ZIP”下载为准；开发包 examples 与 baseline ZIP 是两种入口，请勿混用。当前 `baseline_submission.zip` SHA-256：`a76d27c6d1561d4b44e07453597116e4c06d193948355c37242f966a7fea780c`。下载后可按此哈希核对文件。

## 本地开发与依赖

```sh
mkdir -p my_model
cd my_model
python -m zipfile -e ../baseline_submission.zip .
# 按指南的固定版本安装本地依赖，再修改 submission.py / README.md / explain.json
python ../make_submission.py --source . --route ml --environment-id cpu-ml-methods-v3 --out ../my_submission.zip
```

上述命令从可上传样例创建独立开发目录；打包脚本只检查格式，不代表评分通过。本参赛包（22 个文件）不含 student_worker.py、requirements.txt 或合并训练表 training_public.json，三站训练JSON需按脚本需要在本地合并，或使用网站开发包的合并训练表。自定义模型通过公开学习指标和网站隔离评测检查；可运行 `python public_self_eval.py --submission . --training training_public.json --out public-metrics.json`，只输出学习指标，不计算官方分数。服务器固定版本见上节；不要提交依赖另行安装的代码。

每队每小时最多4次，间隔60秒，一次只排一个任务。公开开发评测进程树累计CPU120秒/内部墙钟180秒（外层210秒），私有终测累计CPU180秒/内部墙钟280秒（外层300秒）；整树物理内存上限2GiB，无swap，单线程。首次响应60秒，之后每调用30秒。ML 使用上述 CPU 限额。

## 常见反馈

- Login failed：队伍名称和密码不匹配；首次设密码后邀请链接即作废。
- Invite invalid or expired：一次性码已用/过期/被新码替换；向组委会要重设码。
- Submissions closed：请组委会开放提交。
- ZIP root must contain submission.py：压缩包套了额外一层目录，或错误上传开发包。
- 运行/资源错误：检查方法签名、返回形状、预装依赖、stdout输出和CPU/内存限制。
- DQ：模型违反物理限制或某episode未达到完成门槛，不等于网站故障。

只返回开发集聚合反馈，最终测试标签不下载、不即时展示。训练历史曲线另在网站按站点下载，不含在这个小提交样例里。

## 干预分的双基线条件

干预响应须同时优于零效应和固定中位数参考；干预轨迹技能受效应技能限制。复制自然轨迹不能获得干预改善分。参考误差接近零时不产生改善分；仅效应无改善不单独导致 DQ。

## 公开容量冲击与行动环境分离

行动测试仍使用原有容量环境。act(observation) 的限额是 capacity_kw 与 forecast_capacity，不能把干预容量当作行动限额。字段 intervention_capacity 是从当前步起的公开反事实总站点容量序列，长度与 forecast_capacity 相同，单位kW。它不含未来车辆或真实需求标签。

forecast(observation, horizon_steps, "natural") 保持自然充电定义；"capacity_edf" 使用 intervention_capacity 减 forecast_background 后的EV容量，按最早离开优先，评估限流和后续回补。两种预测须从同一状态开始。仅改变反事实容量，不改变需求、到达、离开或已经交付的能量。字段 forecast_capacity 只用于实际行动容量。

干预环境按全信息离线可达性设计，并在全部场景验证全部目标可以完成；选手仍只能使用在线信息。该反事实是模型内物理效应，不是对真实用户价格反应的证据。可以从截日前公开需求/停留数据学习潜在状态，再结合容量和当前交付状态推演响应。不允许使用隐藏标签、未来到达或记忆场景。

## 提交内容禁用库规则

参赛提交不得使用、调用、打包、复制或内嵌该数据集发布方配套的软件库及其代码；不得通过改名、源码内嵌或动态加载规避本条。本条只约束提交包及其运行内容，不禁止线下训练或预处理时使用该库。组委会提供的 acn_causal 接口、基线和公开训练数据不在禁用范围内。允许在既定接口和资源限制内使用通用数值计算、机器学习库自行实现预测与决策。

新提交在上传时检查，发现明确禁用库依赖或源码将拒收，并提示修改后重新提交。运行环境也禁止按上述库名导入。已提交代码若发现相关内容，将标记并通知对应队伍，由组委会处理，不自动取消资格或删除历史成绩。静态检查无法保证识别所有深度改写或隐藏源码，提交包须可供组委会审查。

### 完成率判定容差

评分时逐车判断：在该场景可服务时段内，剩余未交付电量不超过该车场景目标电量的3%，即计为完成。完成率是满足此条件的车辆数除以场景车辆总数，不是全站送达电量占比。3%按每辆车自身的目标电量计算，不能用其他车辆的超额交付抵消。Caltech85%、JPL65%、Office85%的逐场景门槛以及物理功率约束保持不变。仅改变评分的完成判定，不改变模拟器充电过程或charging_complete观测。

## 预训练模型与 CPU 资源

允许参赛者在评测前使用合法的公开训练数据训练自己的模型，评测时加载随 ZIP 提交的权重。不允许读取或利用隐藏评测数据、评测场景的未来观测/未来离开时间或评测真值训练模型或直接获取答案。公开历史训练表中的合法标签可以用于训练。公开开发反馈用于调试，不是训练真值来源。禁止把题目或接口输入提交给在线大模型 API 直接获取结果；AI 编程辅助须披露。

支持自有 PyTorch CPU 模型，包括 MLP。评测环境固定 Python 3.12.3、torch 2.15.0+cpu、numpy 1.26.4、pandas 2.1.4、scipy 1.11.4、scikit-learn 1.4.1.post1。评测离线，不运行 requirements 或安装脚本，不接受完整 venv。不依赖 Torch 的 ML 包保持兼容。

公开开发评测每项累计 CPU 预算为 120 秒，包含导入、模型初始化/加载、仍在评测现场进行的训练及全部场景评测；不是每场景 120 秒，也不是额外赠送初始化时间；内部墙钟 180 秒，外层 210 秒。私有终测每项累计 CPU 预算为 180 秒，结构相同；内部墙钟 280 秒，外层 300 秒。每项整个评测进程树最多 2 GiB，单核速率、单线程。模型启动到首次响应最多 60 秒，之后每次接口调用最多 30 秒；全部计入同一累计 CPU 预算。超限或运行失败不产生有效分数。开发与最终评测共用最多 3 个评测槽，可能排队，不承诺即时返回。

提交入口仍为根目录 `submission.py` 的 `build_model(training)`。返回对象仍提供 `predict`、`arrival_rate`、`act`、`forecast`、`explain`。模型结构、自有加载代码与权重一起打包。推荐 `.npz` / `.npy` 纯数值权重，用 `np.load(..., allow_pickle=False)` 读取。不要将 `.pt` / `.pth` 改扩展名绕过类型检查；本次不增加这些后缀。

```python
from pathlib import Path
import numpy as np

def build_model(training):
    path = Path(__file__).resolve().parent / "weights" / "model.npz"
    with np.load(path, allow_pickle=False) as weights:
        # 用自己的模型结构严格校验并加载数值权重。
        model = YourModel.from_arrays(weights)
    return model
```

上例中的 `YourModel` 由参赛者实现，不是预装类。请同时提交模型结构、特征转换、归一化参数，并使用与评测相同的固定版本验证加载。不要把训练数据或评测器打入提交包。

ZIP 的根目录须含 `manifest.json`、`README.md`、`explain.json`、`submission.py`；用打包脚本生成文件哈希，增加 `--include weights/model.npz your_model.py`。ZIP 20 MiB / 展开 100 MiB / 500 项及白名单格式保持不变。每队滚动 60 分钟最多 4 次接受提交、接受间隔至少 60 秒、同队最多一项排队或运行。

## 通用方法库与整数规划

环境预装 XGBoost CPU（安装包名 xgboost-cpu，import xgboost）、LightGBM、statsmodels。整数规划可用现有 SciPy 1.11.4 的 scipy.optimize.milp/HiGHS；可设 time_limit 和 mip_rel_gap，未获得可行解时须自行提供合法回退决策。它们仍计入同一个整树120CPU秒、2GiB与每次调用时限，不保证任意规模都能完成。

预训练可在评测外完成，提交自有源码、结构/特征转换和允许格式的权重；GPR等模型可只导出预测需要的数值状态，避免携带训练优化中间态。使用 np.load(..., allow_pickle=False) 读取纯数值NPZ/NPY。pickle/joblib仍限原隔离沙箱且需版本一致。不得携带自定义.so/可执行求解器或安装脚本，不新增.pt/.pth/safetensors后缀。模型效果由其实现和数据决定，不因方法名加分。
