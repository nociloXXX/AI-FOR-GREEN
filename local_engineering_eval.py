"""本地工程分模拟检查。

参考 docs/SCORING.md 与 docs/SUBMISSION_GUIDE.md 的公开口径，模拟官网
打包/格式/接口/资源检查，给出 0-40 分的本地工程分估算。只做检查，不修改
任何算法文件，也不产生官方分数。

用法:
    python local_engineering_eval.py --source . --out engineering-metrics.json
    python local_engineering_eval.py --zip my_submission.zip
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

ALLOWED_EXT = {".py", ".json", ".txt", ".md", ".npz", ".npy", ".pkl", ".joblib"}
FORBIDDEN_PARTS = ("data/", "__pycache__", ".venv", ".git/", "node_modules")
FORBIDDEN_EXT = {".pt", ".pth", ".safetensors", ".so", ".dll", ".exe"}

SOURCE_FILES = [
    "README.md",
    "explain.json",
    "submission.py",
    "module_1_data.py",
    "module_2_prediction.py",
    "module_3_world_model.py",
    "module_4_intervention.py",
    "module_5_decision.py",
    "acn_causal/student_api.py",
]


def parse_args():
    parser = argparse.ArgumentParser(description="本地工程分模拟检查")
    parser.add_argument("--source", default=".")
    parser.add_argument("--zip", default=None, help="可选：检查已打包的提交 ZIP")
    parser.add_argument("--out", default=None, help="结果写入 JSON")
    return parser.parse_args()


def collect_source_items(root: Path):
    items = {}
    for name in SOURCE_FILES:
        f = root / name
        if f.is_file():
            items[name] = f.read_bytes()
    return items


def collect_zip_items(zip_path: Path):
    if not zip_path.is_file():
        return None, None, ["ZIP 不存在"]
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if len(names) > 500:
            return None, None, ["ZIP 条目超过 500"]
        items = {}
        warnings = []
        for name in names:
            info = zf.getinfo(name)
            if info.is_dir():
                continue
            items[name] = zf.read(name)
        expanded = sum(len(v) for v in items.values())
        zip_size = zip_path.stat().st_size
        if zip_size > 20 * 1024**2:
            warnings.append("ZIP 超过 20 MiB")
        if expanded > 100 * 1024**2:
            warnings.append("解压后超过 100 MiB")
        return items, {"zip_size": zip_size, "expanded": expanded, "entries": len(items)}, warnings


def check_required(items):
    details = []
    score = 0
    for name in ("README.md", "explain.json", "submission.py"):
        if name in items:
            score += 2
            details.append(f"通过: 必需文件 {name}")
        else:
            details.append(f"缺失: 必需文件 {name}")
    if "manifest.json" in items:
        score += 2
        details.append("通过: 根目录含 manifest.json")
    else:
        details.append("提示: 目录模式无 manifest.json，打包时会生成")
    return score, 8, details


def check_explain(items):
    details = []
    if "explain.json" not in items:
        return 0, 4, ["缺失: explain.json"]
    try:
        explain = json.loads(items["explain.json"].decode("utf-8"))
    except Exception as exc:
        return 0, 4, [f"explain.json 解析失败: {exc}"]
    ok = (
        isinstance(explain.get("method"), str)
        and bool(explain.get("method").strip())
        and isinstance(explain.get("assumptions"), list)
        and all(isinstance(v, str) for v in explain["assumptions"])
        and isinstance(explain.get("limitations"), list)
        and all(isinstance(v, str) for v in explain["limitations"])
        and bool(explain["assumptions"])
        and bool(explain["limitations"])
    )
    if ok:
        details.append("通过: explain.json 字段合法且非空")
        return 4, 4, details
    details.append("问题: explain.json 字段不完整")
    return 2, 4, details


def check_contract(items):
    details = []
    if "submission.py" not in items:
        return 0, 4, ["缺失: submission.py"]
    try:
        tree = ast.parse(items["submission.py"].decode("utf-8"))
    except Exception as exc:
        return 0, 4, [f"submission.py 语法错误: {exc}"]
    if any(isinstance(n, ast.FunctionDef) and n.name == "build_model" for n in tree.body):
        details.append("通过: submission.py 导出 build_model")
        return 4, 4, details
    details.append("问题: submission.py 未导出 build_model")
    return 0, 4, details


def check_file_safety(items):
    details = []
    warnings = []
    bad = []
    for name in items:
        posix = PurePosixPath(name)
        if posix.is_absolute() or ".." in posix.parts or "\\" in name:
            bad.append(name)
            details.append(f"拒绝: 不安全路径 {name}")
            continue
        lower = name.lower()
        if lower.endswith(tuple(FORBIDDEN_EXT)):
            bad.append(name)
            details.append(f"拒绝: 禁用扩展名 {name}")
        if any(part in lower for part in FORBIDDEN_PARTS):
            warnings.append(f"警告: 不应打包 {name}")
        ext = posix.suffix.lower()
        if ext and ext not in ALLOWED_EXT:
            bad.append(name)
            details.append(f"拒绝: 白名单外扩展名 {name}")
    score = 8 if not bad else 0
    if not bad:
        details.append("通过: 所有文件都在白名单且路径安全")
    return score, 8, details, warnings


def check_manifest(items):
    if "manifest.json" not in items:
        return 0, 4, ["目录模式跳过 manifest 校验"]
    try:
        manifest = json.loads(items["manifest.json"].decode("utf-8"))
    except Exception as exc:
        return 0, 4, [f"manifest.json 解析失败: {exc}"]
    required_fields = ["route", "contract_version", "environment_id", "seed", "files"]
    if not all(field in manifest for field in required_fields):
        return 0, 4, ["manifest.json 缺少必需字段"]
    files = manifest["files"]
    mismatch = []
    for name, digest in files.items():
        if name not in items:
            mismatch.append(f"{name}: 清单有但包内缺失")
            continue
        actual = hashlib.sha256(items[name]).hexdigest()
        if actual != digest:
            mismatch.append(f"{name}: SHA-256 不一致")
    for name in items:
        if name != "manifest.json" and name not in files:
            mismatch.append(f"{name}: 包内有但清单未记录")
    if mismatch:
        return 0, 4, mismatch[:5]
    return 4, 4, ["通过: manifest 字段与哈希一致"]


def check_stdout_hygiene(items):
    details = []
    offenders = []
    for name, data in items.items():
        if name.endswith(".py") and b"print(" in data:
            offenders.append(name)
    if offenders:
        details.append("警告: 以下文件含 print，评测要求调试信息走 stderr: " + ", ".join(offenders))
    else:
        details.append("通过: 未发现 stdout print")
    return details


def runtime_smoke(source_dir: Path):
    sys.path.insert(0, str(source_dir.resolve()))
    try:
        from acn_causal.student_api import ArrivalFeatures, Observation, VehicleObservation
        from submission import build_model
    except Exception as exc:
        return 0, 8, [f"运行时导入失败: {exc}"], None

    training = {}
    import pandas as pd

    for site in ("caltech", "jpl", "office1"):
        src = source_dir / "data" / f"training_{site}.json"
        if not src.exists():
            return 0, 8, [f"缺少训练表 data/training_{site}.json"], None
        frame = pd.DataFrame(json.loads(src.read_text(encoding="utf-8")))
        frame["day"] = pd.to_datetime(frame.day)
        training[site] = frame

    started = time.perf_counter()
    model = build_model(training)
    build_seconds = time.perf_counter() - started

    details = [f"通过: build_model 构建耗时 {build_seconds:.1f}s"]
    problems = []
    try:
        pred = model.predict(ArrivalFeatures("t1", "caltech", 9.4, 0, 6.6, None))
        if not (math.isfinite(float(pred.demand_kwh)) and math.isfinite(float(pred.dwell_hours))):
            problems.append("predict 返回非有限值")
        rate = model.arrival_rate("caltech", 9, 0)
        if not math.isfinite(float(rate)) or float(rate) < 0:
            problems.append("arrival_rate 非法")

        vehicle = VehicleObservation(ArrivalFeatures("t1", "caltech", 9.4, 0, 6.6, None), 0, 0, 0.0, False)
        observation = Observation(
            step=0,
            dt_hours=0.25,
            horizon_steps=4,
            site="caltech",
            local_hour=9.0,
            weekday=0,
            vehicles=(vehicle,),
            capacity_kw=100.0,
            background_kw=10.0,
            prices=(0.6,) * 4,
            forecast_capacity=(100.0,) * 4,
            forecast_background=(10.0,) * 4,
            historical_arrivals=(0,) * 4,
            intervention_capacity=(60.0,) * 4,
        )
        action = model.act(observation)
        if not isinstance(action, dict):
            problems.append("act 返回类型不是 dict")
        for kind in ("natural", "capacity_edf"):
            out = model.forecast(observation, 4, kind)
            if "ev_load_kw" not in out or "active_count" not in out:
                problems.append(f"forecast/{kind} 缺少字段")
            elif len(out["ev_load_kw"]) != 4 or len(out["active_count"]) != 4:
                problems.append(f"forecast/{kind} 长度错误")
        explanation = model.explain()
        if not all(k in explanation for k in ("method", "assumptions", "limitations")):
            problems.append("explain 返回结构不完整")
    except Exception as exc:
        problems.append(f"五接口运行异常: {exc}")

    total_seconds = time.perf_counter() - started
    if problems:
        details.extend(problems)
        return 0, 8, details, {"build_seconds": build_seconds, "total_seconds": total_seconds}

    details.append("通过: 五接口方法全部运行")
    score = 8
    return score, 8, details, {"build_seconds": build_seconds, "total_seconds": total_seconds}


def resource_score(timing):
    if timing is None:
        return 0, 4, ["未运行运行时冒烟"]
    total = timing["total_seconds"]
    if total <= 60:
        return 4, 4, [f"通过: 总耗时 {total:.1f}s，远低于开发评测 180s 墙钟"]
    if total <= 120:
        return 2, 4, [f"提示: 总耗时 {total:.1f}s，仍在 180s 墙钟内但建议优化"]
    return 0, 4, [f"风险: 总耗时 {total:.1f}s，接近开发评测 180s 墙钟上限"]


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()
    root = Path(args.source).resolve()
    if args.zip:
        items, meta, warnings = collect_zip_items(Path(args.zip))
        source_for_runtime = root
    else:
        items = collect_source_items(root)
        meta = {"zip_size": None, "expanded": None, "entries": len(items)}
        warnings = []
        source_for_runtime = root

    if items is None:
        print("检查失败:", warnings)
        return 1

    report = {"kind": "local-engineering-estimate", "mode": "zip" if args.zip else "source"}
    total = 0
    maximum = 0

    sections = []

    score, maxv, details = check_required(items)
    total += score
    maximum += maxv
    sections.append({"name": "打包完整性", "score": score, "max": maxv, "details": details})

    score, maxv, details = check_explain(items)
    total += score
    maximum += maxv
    sections.append({"name": "explain 文档", "score": score, "max": maxv, "details": details})

    score, maxv, details = check_contract(items)
    total += score
    maximum += maxv
    sections.append({"name": "提交契约", "score": score, "max": maxv, "details": details})

    score, maxv, details, safety_warnings = check_file_safety(items)
    total += score
    maximum += maxv
    sections.append({"name": "格式与安全", "score": score, "max": maxv, "details": details})
    warnings.extend(safety_warnings)

    score, maxv, details = check_manifest(items)
    total += score
    maximum += maxv
    sections.append({"name": "manifest 校验", "score": score, "max": maxv, "details": details})

    stdout_details = check_stdout_hygiene(items)
    if stdout_details:
        warnings.extend(stdout_details)

    score, maxv, details, timing = runtime_smoke(source_for_runtime)
    total += score
    maximum += maxv
    sections.append({"name": "五接口运行", "score": score, "max": maxv, "details": details})

    score, maxv, details = resource_score(timing)
    total += score
    maximum += maxv
    sections.append({"name": "资源表现", "score": score, "max": maxv, "details": details})

    print("本地工程分模拟（非官方分数）")
    print("-" * 50)
    for section in sections:
        print(f"{section['name']}: {section['score']}/{section['max']}")
        for line in section["details"]:
            print("  " + line)
    if warnings:
        print("警告/提示:")
        for line in warnings:
            print("  " + line)
    print("-" * 50)
    print(f"工程分估算: {total}/{maximum}")
    print("说明: 官网 40 分还含展示与人气等现场项，本脚本只模拟可机器检查的部分。")

    report["score"] = total
    report["max"] = maximum
    report["sections"] = sections
    report["warnings"] = warnings
    report["meta"] = meta
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2),
            encoding="utf-8",
        )
        print("结果已写入:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
