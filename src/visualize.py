"""演示用图表。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C


def plot_forecast(timestamps, actual, pred, station_id=None, path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timestamps, actual, label="actual", linewidth=1.4)
    ax.plot(timestamps, pred, label="predicted", linewidth=1.4, alpha=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel("power (kW)")
    title = f"station {station_id}" if station_id else "forecast"
    ax.set_title(title)
    ax.legend()
    fig.autofmt_xdate()

    if path is None:
        C.FIGURE_DIR.mkdir(exist_ok=True)
        path = C.FIGURE_DIR / f"{title.replace(' ', '_')}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
