"""把 pipeline 的运行产物渲染成一份自包含 HTML 报告。

不依赖 matplotlib，也不依赖外网，现场断网时双击即可打开。
"""
import json
from pathlib import Path

import pandas as pd

from . import config as C
from .evaluate import decision_metrics, forecast_metrics


def load_artifacts(path=None):
    path = Path(path) if path else C.ARTIFACTS_FILE
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def summarize(df):
    rows = []
    step_hours = C.STEP_MINUTES / 60
    for station, group in df.groupby(C.STATION_COL):
        group = group.sort_values("timestamp")
        actual = group["actual_power_kw"].to_numpy(dtype=float)
        pred = group["predicted_power_kw"].to_numpy(dtype=float)
        dispatch = group["dispatch_power_kw"].to_numpy(dtype=float)
        capacity = C.STATION_CAPACITY_KW.get(station, float(dispatch.max()))

        forecast = forecast_metrics(actual, pred)
        decision = decision_metrics(actual, dispatch, capacity, step_hours)

        rows.append(
            {
                "station": station,
                "mae": round(forecast["mae"], 2),
                "rmse": round(forecast["rmse"], 2),
                "mape": round(forecast["mape"], 1),
                "peak_actual": round(decision["peak_before_kw"], 1),
                "peak_dispatch": round(decision["peak_after_kw"], 1),
                "capacity": round(decision["capacity_kw"], 1),
                "peak_reduction_pct": round(decision["peak_reduction_pct"], 1),
                "unmet_kwh": round(decision["unmet_kwh"], 1),
                "total_kwh": round(decision["total_energy_kwh"], 1),
                "before_violations": decision["before_violations"],
                "after_violations": decision["after_violations"],
            }
        )
    return rows


def _summary_table(summaries):
    head = (
        "<tr><th>场站</th><th>MAE</th><th>RMSE</th><th>MAPE%</th>"
        "<th>实际峰值</th><th>调度后峰值</th><th>容量</th><th>削峰%</th>"
        "<th>越限(前)</th><th>越限(后)</th><th>未满足kWh</th></tr>"
    )
    body = []
    for row in summaries:
        body.append(
            "<tr>"
            f"<td>{row['station']}</td>"
            f"<td>{row['mae']}</td>"
            f"<td>{row['rmse']}</td>"
            f"<td>{row['mape']}</td>"
            f"<td>{row['peak_actual']}</td>"
            f"<td>{row['peak_dispatch']}</td>"
            f"<td>{row['capacity']}</td>"
            f"<td>{row['peak_reduction_pct']}</td>"
            f"<td>{row['before_violations']}</td>"
            f"<td>{row['after_violations']}</td>"
            f"<td>{row['unmet_kwh']}</td>"
            "</tr>"
        )
    return "<table>" + head + "".join(body) + "</table>"


def _series_data(df):
    payload = {}
    for station, group in df.groupby(C.STATION_COL):
        group = group.sort_values("timestamp")
        payload[station] = {
            "labels": [str(ts) for ts in group["timestamp"]],
            "actual": [round(float(v), 2) for v in group["actual_power_kw"]],
            "predicted": [round(float(v), 2) for v in group["predicted_power_kw"]],
            "dispatch": [round(float(v), 2) for v in group["dispatch_power_kw"]],
            "capacity": C.STATION_CAPACITY_KW.get(station, 0),
        }
    return payload


TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>A 赛道运行效果报告</title>
<style>
  :root { color-scheme: light; }
  body { margin: 0; font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         background: #f6f7f9; color: #1f2937; }
  header { background: #10243e; color: #fff; padding: 22px 28px; }
  header h1 { margin: 0 0 6px; font-size: 20px; letter-spacing: 0; }
  header p { margin: 0; color: #b7c4d4; font-size: 13px; }
  main { padding: 20px 28px 40px; max-width: 1120px; margin: 0 auto; }
  .tabs { display: flex; gap: 8px; margin: 16px 0; flex-wrap: wrap; }
  .tabs button { border: 1px solid #cfd6df; background: #fff; color: #334155;
                 padding: 8px 18px; border-radius: 6px; cursor: pointer; font-size: 14px; }
  .tabs button.active { background: #155e75; border-color: #155e75; color: #fff; }
  .panel { display: none; }
  .panel.active { display: block; }
  .card { background: #fff; border: 1px solid #e5e9ef; border-radius: 8px;
          padding: 16px 18px; margin-bottom: 16px; }
  .card h2 { margin: 0 0 8px; font-size: 16px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; background: #fff; }
  th, td { border: 1px solid #e5e9ef; padding: 7px 9px; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  th { background: #f1f5f9; color: #334155; font-weight: 600; }
  .legend { display: flex; gap: 16px; margin: 10px 2px 4px; flex-wrap: wrap; font-size: 13px; }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .dot { width: 12px; height: 3px; display: inline-block; border-radius: 2px; }
  .note { color: #64748b; font-size: 12px; line-height: 1.6; }
  svg { max-width: 100%; height: auto; display: block; }
</style>
</head>
<body>
<header>
  <h1>A 赛道运行效果报告</h1>
  <p>实际负荷 vs 预测负荷 vs 调度后功率；红色虚线为变压器容量上限</p>
</header>
<main>
  <div class="tabs" id="tabs"></div>
  <div class="card">__TABLE__</div>
  <div id="panels"></div>
  <div class="note">
    高分方向：预测曲线尽量贴合实际曲线，调度后曲线压在容量线以下，削峰百分比为正，
    未满足电量接近 0，越限次数为 0。MAPE/MAE 越低越好。
  </div>
</main>
<script>
const DATA = __DATA__;
const stations = Object.keys(DATA);
const tabs = document.getElementById("tabs");
const panels = document.getElementById("panels");

function drawChart(container, station, mode) {
  const d = DATA[station];
  const W = 960, H = 280, padL = 56, padR = 18, padT = 18, padB = 48;
  const series = mode === "forecast"
    ? [ {name:"实际", color:"#2563eb", values:d.actual},
        {name:"预测", color:"#f59e0b", values:d.predicted} ]
    : [ {name:"调度后", color:"#10b981", values:d.dispatch},
        {name:"实际", color:"#2563eb", values:d.actual} ];
  const cap = d.capacity || 0;
  let maxV = Math.max(...series.flatMap(s => s.values), cap) * 1.12 || 1;
  const n = d.labels.length;
  const x = i => padL + (W - padL - padR) * (n > 1 ? i / (n - 1) : 0.5);
  const y = v => padT + (H - padT - padB) * (1 - v / maxV);

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);

  for (let g = 0; g <= 4; g++) {
    const v = maxV * g / 4;
    const yy = y(v);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", padL); line.setAttribute("x2", W - padR);
    line.setAttribute("y1", yy); line.setAttribute("y2", yy);
    line.setAttribute("stroke", "#e2e8f0"); line.setAttribute("stroke-width", "1");
    svg.appendChild(line);
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", padL - 8); text.setAttribute("y", yy + 4);
    text.setAttribute("text-anchor", "end"); text.setAttribute("font-size", "10");
    text.setAttribute("fill", "#64748b");
    text.textContent = Math.round(v);
    svg.appendChild(text);
  }

  const step = Math.max(1, Math.ceil(n / 8));
  for (let i = 0; i < n; i += step) {
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", x(i)); text.setAttribute("y", H - padB + 16);
    text.setAttribute("text-anchor", "middle"); text.setAttribute("font-size", "10");
    text.setAttribute("fill", "#64748b");
    text.textContent = d.labels[i].slice(5, 16);
    svg.appendChild(text);
  }

  if (cap > 0) {
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", padL); line.setAttribute("x2", W - padR);
    line.setAttribute("y1", y(cap)); line.setAttribute("y2", y(cap));
    line.setAttribute("stroke", "#ef4444"); line.setAttribute("stroke-width", "1.5");
    line.setAttribute("stroke-dasharray", "5 4");
    svg.appendChild(line);
  }

  series.forEach(s => {
    const points = s.values.map((v, i) => x(i) + "," + y(v)).join(" ");
    const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    poly.setAttribute("points", points);
    poly.setAttribute("fill", "none"); poly.setAttribute("stroke", s.color);
    poly.setAttribute("stroke-width", "1.7"); poly.setAttribute("stroke-linejoin", "round");
    svg.appendChild(poly);
  });

  const legend = document.createElement("div");
  legend.className = "legend";
  series.forEach(s => {
    legend.innerHTML += '<span><i class="dot" style="background:' + s.color + '"></i>' + s.name + "</span>";
  });
  legend.innerHTML += '<span><i class="dot" style="background:#ef4444"></i>容量上限</span>';

  container.appendChild(legend);
  container.appendChild(svg);
}

stations.forEach((station, idx) => {
  const btn = document.createElement("button");
  btn.textContent = station;
  btn.className = idx === 0 ? "active" : "";
  btn.onclick = () => {
    document.querySelectorAll(".tabs button").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("panel-" + station).classList.add("active");
  };
  tabs.appendChild(btn);

  const panel = document.createElement("div");
  panel.className = "panel" + (idx === 0 ? " active" : "");
  panel.id = "panel-" + station;
  panel.innerHTML =
    '<div class="card"><h2>预测效果：实际 vs 预测</h2></div>' +
    '<div class="card"><h2>调度效果：实际 vs 调度后（削峰填谷）</h2></div>';
  panels.appendChild(panel);
  drawChart(panel.children[0], station, "forecast");
  drawChart(panel.children[1], station, "dispatch");
});
</script>
</body>
</html>
"""


def render_report(df=None, path=None):
    df = df if df is not None else load_artifacts()
    summaries = summarize(df)
    data = _series_data(df)
    html = (
        TEMPLATE.replace("__TABLE__", _summary_table(summaries))
        .replace("__DATA__", json.dumps(data, ensure_ascii=False))
    )
    if path is None:
        C.SUBMISSION_DIR.mkdir(exist_ok=True)
        path = C.SUBMISSION_DIR / "report.html"
    path = Path(path)
    path.write_text(html, encoding="utf-8")
    return path


if __name__ == "__main__":
    out = render_report()
    print("报告:", out)
