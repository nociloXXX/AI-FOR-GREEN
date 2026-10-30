"""生成合成充电数据，用于开题前验证模板链路。"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
START = pd.Timestamp("2026-10-24 00:00")
DAYS = 7
STATION_PILES = {"A": 6, "B": 8, "C": 4}
POWER_OPTIONS = [7.0, 22.0, 60.0, 120.0]

rows = []
for station, n_piles in STATION_PILES.items():
    for pile_index in range(n_piles):
        t = START
        pile_id = f"{station}{pile_index:02d}"
        while t < START + pd.Timedelta(days=DAYS):
            t += pd.Timedelta(minutes=int(rng.integers(30, 240)))
            if t >= START + pd.Timedelta(days=DAYS):
                break

            duration = int(rng.integers(30, 240))
            max_power = POWER_OPTIONS[pile_index % len(POWER_OPTIONS)]
            energy = max_power * (duration / 60) * float(rng.uniform(0.45, 0.95))
            start_soc = float(rng.uniform(12, 55))
            end_soc = min(100.0, start_soc + energy * 0.55)

            end = t + pd.Timedelta(minutes=duration)
            if end > START + pd.Timedelta(days=DAYS):
                end = START + pd.Timedelta(days=DAYS)
                energy = max_power * ((end - t).total_seconds() / 3600) * 0.7
                if (end - t).total_seconds() <= 0:
                    continue

            rows.append(
                {
                    "station_id": station,
                    "pile_id": pile_id,
                    "start_time": t,
                    "end_time": end,
                    "start_soc": start_soc,
                    "end_soc": end_soc,
                    "energy_kwh": energy,
                    "max_power_kw": max_power,
                }
            )

charging = pd.DataFrame(rows).sort_values(["station_id", "start_time"])
charging.to_csv(DATA_DIR / "raw_charging.csv", index=False)

grid = pd.date_range(
    START, START + pd.Timedelta(days=DAYS) - pd.Timedelta(minutes=15), freq="15min"
)
price = pd.DataFrame(
    {
        "timestamp": grid,
        "price_yuan_per_kwh": np.where(
            (grid.hour >= 10) & (grid.hour < 15),
            1.15,
            np.where((grid.hour >= 19) & (grid.hour < 22), 1.2, 0.35),
        ),
    }
)
price.to_csv(DATA_DIR / "price.csv", index=False)

weather = pd.DataFrame(
    {
        "timestamp": grid,
        "temperature_c": 18 + 8 * np.sin((grid.hour - 9) / 24 * 2 * np.pi),
    }
)
weather.to_csv(DATA_DIR / "weather.csv", index=False)

print("sample data written to", DATA_DIR)
print("sessions:", len(charging))
