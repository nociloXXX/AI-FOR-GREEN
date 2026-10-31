"""本地公开自评脚本：只算学习指标，不含评分权重、隐藏数据或裁判代码。

用途
    在你自己的电脑上、用自己的提交目录，算一组"预测误差"指标，
    用来检查模型的需求电量 / 停留时长预测是否靠谱。

    它只是公开训练数据上的预测 holdout：
    - 不是官方总分，也不是完整的五接口评测；
    - 结果仅供本地参考，与网站开发评测、最终测试的分数没有对应关系。

用法
    python public_self_eval.py --submission 你的提交目录 \
        --training training_public.json --out metrics.json

    --submission  你的提交目录（内含 submission.py，须提供 build_model(train)）
    --training    合并后的训练标签 JSON：{站点名: [记录, ...]}，
                  参赛包的三站训练 JSON 需先按此格式合并（见 SUBMISSION_GUIDE.md）
    --out         指标输出文件（JSON）

    依赖版本以 SUBMISSION_GUIDE.md 的固定版本为准；请只在本机运行，
    不要把它当作网站评测器，也不要 import 到网站代码里。
"""
import argparse
import json
import math
import sys
from pathlib import Path


# =============================================================================
# 1. 解析命令行参数
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='本地公开自评：公开数据按日期 holdout 的学习指标（非官方分数）')
    parser.add_argument('--submission', required=True, help='提交目录（内含 submission.py）')
    parser.add_argument('--training', required=True, help='合并后的训练标签 JSON')
    parser.add_argument('--out', required=True, help='指标输出文件路径（JSON）')
    return parser.parse_args()


# =============================================================================
# 2. 按站点做时间先后 80:20 划分：前 80% 的日期用于训练，后 20% 用于验证
# =============================================================================
def split_train_holdout(training_path):
    import pandas as pd

    source = json.loads(Path(training_path).read_text())
    train = {}   # 站点 -> 训练 DataFrame
    held = []    # [(站点, 验证记录), ...]
    for site, rows in source.items():
        frame = pd.DataFrame(rows)
        frame['day'] = pd.to_datetime(frame.day)
        days = sorted(frame.day.unique())
        cut = days[max(1, int(len(days) * .8)) - 1]   # 80% 分界日
        train[site] = frame[frame.day <= cut].copy()
        held.extend((site, r) for r in frame[frame.day > cut].to_dict('records'))
    if not held:
        raise ValueError('不同日期太少，无法按时间划分验证集')
    return train, held


# =============================================================================
# 3. 逐条调用选手模型的 predict，累计需求电量 / 停留时长的绝对误差
# =============================================================================
def evaluate(model, held):
    import pandas as pd
    from acn_causal.student_api import ArrivalFeatures

    errors = {'demand_kwh': [], 'dwell_hours': []}
    all_finite_nonnegative = True
    for i, (site, record) in enumerate(held):
        requested = record.get('last_requested_energy_kwh')
        requested = float(requested) if requested is not None and pd.notna(requested) else None
        features = ArrivalFeatures(
            'held-' + str(i), site,
            float(record['hour']), int(record['weekday']),
            float(record['charger_limit_kw']), requested)
        pred = model.predict(features)
        for field in ('demand_kwh', 'dwell_hours'):
            value = float(getattr(pred, field))
            if not math.isfinite(value) or value < 0:
                all_finite_nonnegative = False   # 非法预测：标记，但不计入误差
                continue
            errors[field].append(abs(value - float(record[field])))
    return errors, all_finite_nonnegative


# =============================================================================
# 4. 汇总指标、写 JSON 文件、打印可读结果
# =============================================================================
def main():
    args = parse_args()

    # 选手的 submission.py 从提交目录导入
    sys.path.insert(0, str(Path(args.submission).resolve()))
    from submission import build_model

    train, held = split_train_holdout(args.training)
    model = build_model(train)
    errors, all_finite_nonnegative = evaluate(model, held)

    report = {
        'kind': 'public-learning-metrics',
        'split': 'time/day 80:20; fit only earlier days',
        'train_rows': sum(len(f) for f in train.values()),
        'validation_rows': len(held),
        'finite_nonnegative_predictions': all_finite_nonnegative,
        'mae': {k: sum(v) / len(v) if v else None for k, v in errors.items()},
        'note': 'Prediction holdout only. Not an official score or a full five-interface evaluation.',
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))

    # 控制台可读输出（指标文件保持 JSON 格式不变）
    def fmt(v):
        return ('%.4f' % v) if v is not None else '无（没有有效预测）'

    print()
    print('公开学习指标（本地自评，不是官方分数）')
    print('-' * 46)
    print('划分方式：按日期 80:20 时间划分，只用较早日期训练')
    print('训练行数：%d' % report['train_rows'])
    print('验证行数：%d' % report['validation_rows'])
    print('预测全部有限且非负：%s' % ('是' if all_finite_nonnegative else '否'))
    print('需求电量 MAE：%s kWh' % fmt(report['mae']['demand_kwh']))
    print('停留时长 MAE：%s 小时' % fmt(report['mae']['dwell_hours']))
    print('-' * 46)
    print('同一组指标已写入 %s。' % args.out)
    print('注意：这只是公开数据上的预测误差，不是官方总分，')
    print('也不是完整的五接口评测。')


if __name__ == '__main__':
    main()
