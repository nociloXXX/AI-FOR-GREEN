# 原始观测历史曲线 · 训练补充

416条历史充电曲线，约252万观测点，所有曲线的最后观测均早于2020-01-01 UTC。来自原包GZ；保留真实时间戳与观测信号，替换文件名中的站点/会话标识。不包含邮箱、姓名、用户ID、原sessionID或冻结评测真值。

按站点下载JSONL.GZ，每行一条curve：curve_token、site、start_utc、end_utc、samples。samples字段：timestamp_utc、current_a、pilot_a、voltage_v、state、energy_kwh_observed、power_kw_observed。原约4秒观测间隔不规则，缺失值为null。JPL主要只有电流，不能假称真实kW负荷。能量表可能重置，并非用户需求意图标签。

这是稀疏选取的真实会话曲线，不是完整站点连续负荷。用作时序/充电动力学研究；不等同裁判的固定6.6kW模型，不要直接把观测电流当成裁判功率。预测只能用预测时点前的历史，后续观测只能用于离线训练目标；不能在当前预测里读取该会话未来。按时间分训练/验证，不随机混合相邻样本。

读取示例：

```python
import gzip,json
with gzip.open("timeseries_caltech.jsonl.gz","rt") as f:
    for line in f:
        curve=json.loads(line)
        samples=curve["samples"]
```

TIMESERIES_MANIFEST.json提供每站曲线数、样本数、最早/最后时间、字节数和SHA256。当前ExtraTrees参考仍只使用会话级标签，此补充未用于重训，分数不变。
