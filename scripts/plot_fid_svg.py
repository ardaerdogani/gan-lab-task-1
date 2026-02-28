"""
Create an SVG plot of FID vs epoch for the report (stdlib-only).

Default input:  runs/gan/train_log.json
Default output: reports/figures/fid_vs_epoch.svg

Usage:
  python scripts/plot_fid_svg.py
  python scripts/plot_fid_svg.py --in runs/gan/train_log.json --out reports/figures/fid_vs_epoch.svg
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape


def load_fid_points(train_log_json: Path) -> list[tuple[int, float]]:
    with train_log_json.open() as f:
        log = json.load(f)
    points: list[tuple[int, float]] = []
    for item in log:
        if "fid" not in item:
            continue
        points.append((int(item["epoch"]), float(item["fid"])))
    points.sort(key=lambda x: x[0])
    return points


def nice_ticks(vmin: float, vmax: float, n: int = 5) -> list[float]:
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin == vmax:
        return [vmin]
    span = vmax - vmin
    raw_step = span / max(1, n - 1)
    power = 10 ** math.floor(math.log10(raw_step))
    for m in (1, 2, 5, 10):
        step = m * power
        if step >= raw_step:
            break
    start = math.floor(vmin / step) * step
    end = math.ceil(vmax / step) * step
    ticks = []
    t = start
    while t <= end + 1e-12:
        ticks.append(t)
        t += step
    return ticks


def render_svg(points: list[tuple[int, float]], title: str) -> str:
    if not points:
        raise ValueError("No FID points found in train log.")

    width, height = 900, 520
    margin = dict(l=80, r=30, t=60, b=70)
    plot_w = width - margin["l"] - margin["r"]
    plot_h = height - margin["t"] - margin["b"]

    epochs = [p[0] for p in points]
    fids = [p[1] for p in points]
    x_min, x_max = min(epochs), max(epochs)
    y_min, y_max = min(fids), max(fids)

    # Add a bit of padding for nicer framing.
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    y0, y1 = y_min - y_pad, y_max + y_pad

    def x_to_px(x: float) -> float:
        if x_max == x_min:
            return margin["l"] + plot_w / 2
        return margin["l"] + (x - x_min) / (x_max - x_min) * plot_w

    def y_to_px(y: float) -> float:
        if y1 == y0:
            return margin["t"] + plot_h / 2
        return margin["t"] + (y1 - y) / (y1 - y0) * plot_h

    # Polyline path
    poly = " ".join(f"{x_to_px(e):.2f},{y_to_px(fid):.2f}" for e, fid in points)

    best_epoch, best_fid = min(points, key=lambda p: p[1])
    best_x, best_y = x_to_px(best_epoch), y_to_px(best_fid)

    x_ticks = nice_ticks(x_min, x_max, n=6)
    y_ticks = nice_ticks(y0, y1, n=6)

    def fmt_tick(v: float) -> str:
        if abs(v) >= 100:
            return f"{v:.0f}"
        if abs(v) >= 10:
            return f"{v:.1f}"
        return f"{v:.2f}"

    # SVG assembly
    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')

    # Title
    parts.append(
        f'<text x="{width/2:.1f}" y="32" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="18" fill="#111">'
        f'{escape(title)}</text>'
    )
    parts.append(
        f'<text x="{width/2:.1f}" y="52" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="12" fill="#555">'
        f'Lower is better</text>'
    )

    # Grid + ticks (Y)
    for t in y_ticks:
        y = y_to_px(t)
        parts.append(f'<line x1="{margin["l"]}" y1="{y:.2f}" x2="{width - margin["r"]}" y2="{y:.2f}" stroke="#eee" stroke-width="1"/>')
        parts.append(f'<text x="{margin["l"]-10}" y="{y+4:.2f}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="11" fill="#444">{escape(fmt_tick(t))}</text>')

    # Grid + ticks (X)
    for t in x_ticks:
        x = x_to_px(t)
        parts.append(f'<line x1="{x:.2f}" y1="{margin["t"]}" x2="{x:.2f}" y2="{height - margin["b"]}" stroke="#f3f3f3" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{height - margin["b"] + 22}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="11" fill="#444">{escape(str(int(round(t))))}</text>')

    # Axes
    parts.append(f'<line x1="{margin["l"]}" y1="{margin["t"]}" x2="{margin["l"]}" y2="{height - margin["b"]}" stroke="#222" stroke-width="1.2"/>')
    parts.append(f'<line x1="{margin["l"]}" y1="{height - margin["b"]}" x2="{width - margin["r"]}" y2="{height - margin["b"]}" stroke="#222" stroke-width="1.2"/>')

    # Labels
    parts.append(
        f'<text x="{width/2:.1f}" y="{height - 24}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="13" fill="#111">Epoch</text>'
    )
    parts.append(
        f'<text x="22" y="{height/2:.1f}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="13" fill="#111" transform="rotate(-90 22 {height/2:.1f})">FID</text>'
    )

    # Curve
    parts.append(f'<polyline fill="none" stroke="#1976D2" stroke-width="2.5" points="{poly}"/>')

    # Points (subtle)
    for e, fid in points:
        x, y = x_to_px(e), y_to_px(fid)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.6" fill="#1976D2" opacity="0.7"/>')

    # Best point highlight
    parts.append(f'<circle cx="{best_x:.2f}" cy="{best_y:.2f}" r="5.2" fill="#D32F2F"/>')
    parts.append(
        f'<text x="{best_x + 10:.2f}" y="{best_y - 10:.2f}" font-family="Inter, Arial, sans-serif" font-size="12" fill="#D32F2F">'
        f'best: epoch {best_epoch}, FID {best_fid:.2f}</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=Path, default=Path("runs/gan/train_log.json"))
    parser.add_argument("--out", dest="out_path", type=Path, default=Path("reports/figures/fid_vs_epoch.svg"))
    parser.add_argument("--title", type=str, default="FID vs Epoch (CGAN Training)")
    args = parser.parse_args()

    points = load_fid_points(args.in_path)
    svg = render_svg(points, args.title)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(svg, encoding="utf-8")
    print(f"Wrote: {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

