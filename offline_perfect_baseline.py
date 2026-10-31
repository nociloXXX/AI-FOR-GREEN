"""离线全信息基准：完成率上界与最小峰值负荷。

只用公开训练标签，在“未来到达、真实需求、真实离开全部已知”的假设下计算：
  1) 完成率上界：每辆车是否能在自身服务窗口内物理充到 97% 目标电量；
  2) 最小峰值负荷：ALAP（越晚充越好）全信息调度的并发功率峰值；
  3) 当前在线模型（真实 predict + LLF）在相同容量下的完成率。

结果写入 submissions/offline_perfect_baseline.json 与 docs/offline_perfect_baseline.md，
仅作路演对照，不进入 submission.py 在线路径（在线路径用未来真值即违规）。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from completion_proxy import (
    DT,
    SITES,
    _alap_lower_bound,
    _background_curve,
    _build_day_vehicles,
    _peak_concurrent_kw,
    _perfect_predict,
    _real_predict,
    _simulate,
    load_training,
)


def _is_feasible(veh):
    window_kwh = veh["limit"] * (veh["depart"] - veh["arrive"]) * DT
    return 0.97 * veh["target"] <= window_kwh + 1e-9


def _mean(values):
    return round(sum(values) / len(values), 4) if values else 0.0


def main():
    training = load_training()
    real_predict = _real_predict(training)

    report = {"kind": "offline-perfect-baseline", "sites": {}}
    lines = ["# 离线全信息基准对照", ""]
    lines.append("假设：未来到达、真实需求、真实离开全部已知；容量充足时逐车判断可完成性。")
    lines.append("")

    for site in SITES:
        by_day = defaultdict(list)
        for row in training[site]:
            by_day[row["day"]].append(row)
        chosen = sorted(by_day.values(), key=len, reverse=True)[:4]

        ceilings, real_lim, real_normal, perfect_lim = [], [], [], []
        natural_peaks, alap_peaks = [], []
        day_rows = []

        for day_rows_raw in chosen:
            vehicles = _build_day_vehicles(day_rows_raw)
            if not vehicles:
                continue
            natural_peak = _peak_concurrent_kw(vehicles)
            alap_peak = _alap_lower_bound(vehicles)
            feasible = sum(1 for veh in vehicles if _is_feasible(veh))
            ceiling = feasible / len(vehicles)

            background = _background_curve(natural_peak)
            perfect = _simulate(vehicles, site, alap_peak, background, _perfect_predict(vehicles), True)[0]
            real_cap = _simulate(vehicles, site, alap_peak, background, real_predict, True)[0]
            real_norm = _simulate(vehicles, site, natural_peak, background, real_predict, True)[0]

            ceilings.append(ceiling)
            real_lim.append(real_cap)
            real_normal.append(real_norm)
            perfect_lim.append(perfect)
            natural_peaks.append(natural_peak)
            alap_peaks.append(alap_peak)
            day_rows.append(
                {
                    "day": day_rows_raw[0]["day"],
                    "vehicles": len(vehicles),
                    "natural_peak_kw": round(natural_peak, 1),
                    "alap_peak_kw": round(alap_peak, 1),
                    "offline_ceiling": round(ceiling, 4),
                    "online_real_completion_capacity_limited": round(real_cap, 4),
                    "online_real_completion_normal": round(real_norm, 4),
                    "online_perfect_completion_capacity_limited": round(perfect, 4),
                }
            )

        site_report = {
            "days": day_rows,
            "mean_offline_ceiling": _mean(ceilings),
            "mean_online_real_completion_capacity_limited": _mean(real_lim),
            "mean_online_real_completion_normal": _mean(real_normal),
            "mean_online_perfect_completion_capacity_limited": _mean(perfect_lim),
            "mean_natural_peak_kw": _mean(natural_peaks),
            "mean_alap_peak_kw": _mean(alap_peaks),
        }
        report["sites"][site] = site_report

        gap = site_report["mean_offline_ceiling"] - site_report["mean_online_real_completion_capacity_limited"]
        peak_cut = (
            (site_report["mean_natural_peak_kw"] - site_report["mean_alap_peak_kw"])
            / site_report["mean_natural_peak_kw"] * 100
            if site_report["mean_natural_peak_kw"] > 1e-6
            else 0.0
        )
        lines.append(f"## {site}")
        lines.append(
            f"- 完成率上界 {site_report['mean_offline_ceiling']:.1%}，"
            f"在线(容量受限) {site_report['mean_online_real_completion_capacity_limited']:.1%}，"
            f"能力差 {gap:.1%}；在线(正常容量) {site_report['mean_online_real_completion_normal']:.1%}"
        )
        lines.append(
            f"- 全信息最小峰值 {site_report['mean_alap_peak_kw']:.1f} kW，"
            f"自然峰值 {site_report['mean_natural_peak_kw']:.1f} kW，"
            f"理论上可削峰 {peak_cut:.1f}%"
        )
        lines.append(
            f"- 完美预测在线完成率(容量受限) {site_report['mean_online_perfect_completion_capacity_limited']:.1%}："
            f"与离线理论上界的差来自决策，与真实预测的差来自预测误差"
        )
        lines.append("")

    out_json = Path("submissions/offline_perfect_baseline.json")
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = Path("docs/offline_perfect_baseline.md")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("JSON:", out_json)
    print("MD:", out_md)


if __name__ == "__main__":
    main()
