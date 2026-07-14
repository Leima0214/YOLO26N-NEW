
from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "title": "Performance Comparison of YOLO26 Models",
    "subtitle": "Custom normalized metrics",
    "output": "runs/plots/yolo26_metrics_radar.svg",
    "backend": "svg",
    "figure": {
        "figsize": [8.0, 7.2],
        "dpi": 300,
        "font_family": "Times New Roman",
        "title_size": 16,
        "label_size": 10,
        "tick_size": 9,
        "legend_size": 9,
    },
    "radar": {
        "range": [0.0, 1.0],
        "rings": [0.2, 0.4, 0.6, 0.8, 1.0],
        "grid_color": "#d8d8d8",
        "spine_color": "#555555",
        "fill_alpha": 0.08,
    },
    "metrics": [
        {"name": "mAP@0.5:0.95", "higher_is_better": True},
        {"name": "Small Objects (AP_S)", "higher_is_better": True},
        {"name": "Medium Objects (AP_M)", "higher_is_better": True},
        {"name": "Large Objects (AP_L)", "higher_is_better": True},
        {"name": "Inference Speed (FPS)", "higher_is_better": True},
        {"name": "Parameter Efficiency", "higher_is_better": True},
    ],
    "models": [
        {
            "name": "YOLOv8s",
            "values": [0.72, 0.61, 0.74, 0.78, 0.94, 0.90],
            "color": "#1f77b4",
            "linestyle": "-",
            "marker": "o",
        },
        {
            "name": "AMSA-YOLOv8s",
            "values": [0.76, 0.72, 0.79, 0.82, 0.89, 0.84],
            "color": "#ff3b30",
            "linestyle": "--",
            "marker": "s",
        },
        {
            "name": "YOLOv9-C",
            "values": [0.77, 0.66, 0.75, 0.80, 0.84, 0.43],
            "color": "#2ca02c",
            "linestyle": "-.",
            "marker": "^",
        },
        {
            "name": "YOLO26",
            "values": [0.78, 0.68, 0.77, 0.81, 0.91, 0.86],
            "color": "#ff7f0e",
            "linestyle": ":",
            "marker": "D",
        },
    ],
}


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge update into base and return base."""
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | None) -> dict[str, Any]:
    """Load optional JSON config and merge it with DEFAULT_CONFIG."""
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if path is None:
        return config

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        user_config = json.load(f)
    return deep_update(config, user_config)


def validate_config(config: dict[str, Any]) -> None:
    """Validate metric/model lengths and radar ranges before plotting."""
    metrics = config.get("metrics", [])
    models = config.get("models", [])
    if len(metrics) < 3:
        raise ValueError("At least three metrics are required for a radar chart.")
    if not models:
        raise ValueError("At least one model is required.")

    metric_count = len(metrics)
    for model in models:
        values = model.get("values", [])
        if len(values) != metric_count:
            raise ValueError(
                f"Model '{model.get('name', '<unnamed>')}' has {len(values)} values, "
                f"but {metric_count} metrics were defined."
            )

    min_value, max_value = config["radar"]["range"]
    if min_value >= max_value:
        raise ValueError("radar.range must be [min_value, max_value] with min_value < max_value.")


def normalize_values(values: list[float], min_value: float, max_value: float) -> list[float]:
    """Clip values into the configured display range."""
    return [max(min_value, min(float(value), max_value)) for value in values]


def make_closed_angles(metric_count: int) -> list[float]:
    """Create equally spaced polar angles and repeat the first angle."""
    angles = [2 * math.pi * i / metric_count for i in range(metric_count)]
    return angles + angles[:1]


def polar_to_xy(cx: float, cy: float, radius: float, angle: float, ratio: float = 1.0) -> tuple[float, float]:
    """Convert top-start clockwise polar coordinates to SVG coordinates."""
    return cx + radius * ratio * math.sin(angle), cy - radius * ratio * math.cos(angle)


def svg_points(points: list[tuple[float, float]]) -> str:
    """Serialize a list of points for SVG polyline/polygon elements."""
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def dasharray(linestyle: str) -> str:
    """Map matplotlib-like line styles to SVG dash arrays."""
    return {
        "--": "8 5",
        "-.": "9 4 2 4",
        ":": "2 5",
    }.get(linestyle, "")


def marker_svg(x: float, y: float, marker: str, color: str, size: float) -> str:
    """Create a small SVG marker."""
    half = size / 2
    if marker == "s":
        return (
            f'<rect x="{x - half:.2f}" y="{y - half:.2f}" width="{size:.2f}" height="{size:.2f}" '
            f'fill="{escape(color)}" stroke="white" stroke-width="1"/>'
        )
    if marker == "^":
        pts = [(x, y - half), (x - half, y + half), (x + half, y + half)]
        return f'<polygon points="{svg_points(pts)}" fill="{escape(color)}" stroke="white" stroke-width="1"/>'
    if marker == "D":
        pts = [(x, y - half), (x - half, y), (x, y + half), (x + half, y)]
        return f'<polygon points="{svg_points(pts)}" fill="{escape(color)}" stroke="white" stroke-width="1"/>'
    if marker == "*":
        pts = []
        for i in range(10):
            angle = -math.pi / 2 + i * math.pi / 5
            point_radius = half if i % 2 == 0 else half * 0.45
            pts.append((x + point_radius * math.cos(angle), y + point_radius * math.sin(angle)))
        return f'<polygon points="{svg_points(pts)}" fill="{escape(color)}" stroke="white" stroke-width="1"/>'
    return (
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{half:.2f}" fill="{escape(color)}" '
        f'stroke="white" stroke-width="1"/>'
    )


def draw_radar_chart_svg(config: dict[str, Any], output: str | None = None) -> Path:
    """Draw and save the radar chart as dependency-free SVG."""
    validate_config(config)

    figure_cfg = config["figure"]
    radar_cfg = config["radar"]
    metrics = config["metrics"]
    models = config["models"]

    figsize = figure_cfg.get("figsize", [8.0, 7.2])
    width = int(config.get("svg_width", float(figsize[0]) * 120))
    height = int(config.get("svg_height", float(figsize[1]) * 120))
    cx = width * 0.43
    cy = height * 0.53
    radius = min(width * 0.31, height * 0.35)
    min_value, max_value = radar_cfg["range"]
    value_span = max_value - min_value
    metric_labels = [metric["name"] for metric in metrics]
    angles = make_closed_angles(len(metric_labels))
    font_family = figure_cfg.get("font_family", "Times New Roman")
    label_size = figure_cfg.get("label_size", 10) * 1.45
    tick_size = figure_cfg.get("tick_size", 9) * 1.35
    title_size = figure_cfg.get("title_size", 16) * 1.45
    legend_size = figure_cfg.get("legend_size", 9) * 1.35
    grid_color = radar_cfg.get("grid_color", "#d8d8d8")
    spine_color = radar_cfg.get("spine_color", "#555555")
    fill_alpha = radar_cfg.get("fill_alpha", 0.08)

    elements: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        f'<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2:.2f}" y="34" text-anchor="middle" '
            f'font-family="{escape(font_family)}" font-size="{title_size:.1f}" '
            f'font-weight="700">{escape(config.get("title", ""))}</text>'
        ),
    ]
    subtitle = config.get("subtitle", "")
    if subtitle:
        elements.append(
            f'<text x="{width / 2:.2f}" y="58" text-anchor="middle" '
            f'font-family="{escape(font_family)}" font-size="{tick_size:.1f}" fill="#444444">'
            f'{escape(subtitle)}</text>'
        )

    for tick in radar_cfg.get("rings", []):
        ratio = (float(tick) - min_value) / value_span
        ring_points = [polar_to_xy(cx, cy, radius, angle, ratio) for angle in angles[:-1]]
        elements.append(
            f'<polygon points="{svg_points(ring_points)}" fill="none" '
            f'stroke="{escape(grid_color)}" stroke-width="1"/>'
        )
        tick_x, tick_y = polar_to_xy(cx, cy, radius, 0, ratio)
        elements.append(
            f'<text x="{tick_x + 7:.2f}" y="{tick_y + 4:.2f}" font-family="{escape(font_family)}" '
            f'font-size="{tick_size:.1f}" fill="#777777">{float(tick):g}</text>'
        )

    outer_points = [polar_to_xy(cx, cy, radius, angle) for angle in angles[:-1]]
    elements.append(
        f'<polygon points="{svg_points(outer_points)}" fill="none" '
        f'stroke="{escape(spine_color)}" stroke-width="1.5"/>'
    )

    for angle in angles[:-1]:
        end_x, end_y = polar_to_xy(cx, cy, radius, angle)
        elements.append(
            f'<line x1="{cx:.2f}" y1="{cy:.2f}" x2="{end_x:.2f}" y2="{end_y:.2f}" '
            f'stroke="{escape(grid_color)}" stroke-width="1"/>'
        )

    for label, angle in zip(metric_labels, angles[:-1]):
        label_x, label_y = polar_to_xy(cx, cy, radius + 38, angle)
        sin_value = math.sin(angle)
        if sin_value > 0.25:
            anchor = "start"
        elif sin_value < -0.25:
            anchor = "end"
        else:
            anchor = "middle"
        baseline_shift = "0"
        if math.cos(angle) > 0.7:
            baseline_shift = "-0.3em"
        elif math.cos(angle) < -0.7:
            baseline_shift = "0.9em"
        elements.append(
            f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="{anchor}" '
            f'dy="{baseline_shift}" font-family="{escape(font_family)}" '
            f'font-size="{label_size:.1f}" fill="#222222">{escape(label)}</text>'
        )

    for model in models:
        color = model.get("color", "#1f77b4")
        values = normalize_values(model["values"], min_value, max_value)
        points = [
            polar_to_xy(cx, cy, radius, angle, (value - min_value) / value_span)
            for angle, value in zip(angles[:-1], values)
        ]
        closed_points = points + points[:1]
        stroke_dasharray = dasharray(model.get("linestyle", "-"))
        dash_attr = f' stroke-dasharray="{stroke_dasharray}"' if stroke_dasharray else ""
        linewidth = float(model.get("linewidth", 1.8)) * 1.3
        marker_size = float(model.get("markersize", 4.5)) * 1.7
        elements.append(
            f'<polygon points="{svg_points(points)}" fill="{escape(color)}" fill-opacity="{fill_alpha}" '
            f'stroke="none"/>'
        )
        elements.append(
            f'<polyline points="{svg_points(closed_points)}" fill="none" stroke="{escape(color)}" '
            f'stroke-width="{linewidth:.2f}"{dash_attr}/>'
        )
        for x, y in points:
            elements.append(marker_svg(x, y, model.get("marker", "o"), color, marker_size))

    legend_x = width * 0.76
    legend_y = height * 0.11
    legend_row = 25
    legend_width = width * 0.21
    legend_height = 20 + legend_row * len(models)
    elements.append(
        f'<rect x="{legend_x:.2f}" y="{legend_y:.2f}" width="{legend_width:.2f}" '
        f'height="{legend_height:.2f}" fill="white" stroke="#b0b0b0" stroke-width="1"/>'
    )
    for index, model in enumerate(models):
        y = legend_y + 20 + index * legend_row
        color = model.get("color", "#1f77b4")
        stroke_dasharray = dasharray(model.get("linestyle", "-"))
        dash_attr = f' stroke-dasharray="{stroke_dasharray}"' if stroke_dasharray else ""
        elements.append(
            f'<line x1="{legend_x + 12:.2f}" y1="{y:.2f}" x2="{legend_x + 45:.2f}" y2="{y:.2f}" '
            f'stroke="{escape(color)}" stroke-width="2"{dash_attr}/>'
        )
        elements.append(marker_svg(legend_x + 28.5, y, model.get("marker", "o"), color, 8))
        elements.append(
            f'<text x="{legend_x + 55:.2f}" y="{y + 4:.2f}" font-family="{escape(font_family)}" '
            f'font-size="{legend_size:.1f}" fill="#222222">{escape(model.get("name", "Model"))}</text>'
        )

    elements.append("</svg>")

    output_path = Path(output or config.get("output", DEFAULT_CONFIG["output"]))
    if output_path.suffix.lower() != ".svg":
        output_path = output_path.with_suffix(".svg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements), encoding="utf-8")
    return output_path


def draw_radar_chart_matplotlib(config: dict[str, Any], output: str | None = None, show: bool = False) -> Path:
    """Draw and save the radar chart with matplotlib."""
    validate_config(config)

    import matplotlib.pyplot as plt

    figure_cfg = config["figure"]
    radar_cfg = config["radar"]
    metrics = config["metrics"]
    models = config["models"]

    plt.rcParams["font.family"] = figure_cfg.get("font_family", "Times New Roman")
    plt.rcParams["axes.unicode_minus"] = False

    metric_labels = [metric["name"] for metric in metrics]
    angles = make_closed_angles(len(metric_labels))
    min_value, max_value = radar_cfg["range"]

    fig, ax = plt.subplots(
        figsize=tuple(figure_cfg.get("figsize", [8.0, 7.2])),
        subplot_kw={"polar": True},
        dpi=figure_cfg.get("dpi", 300),
    )

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(min_value, max_value)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=figure_cfg.get("label_size", 10))
    ax.set_yticks(radar_cfg.get("rings", []))
    ax.set_yticklabels(
        [f"{tick:g}" for tick in radar_cfg.get("rings", [])],
        fontsize=figure_cfg.get("tick_size", 9),
        color="#666666",
    )
    ax.yaxis.grid(True, color=radar_cfg.get("grid_color", "#d8d8d8"), linewidth=0.8)
    ax.xaxis.grid(True, color=radar_cfg.get("grid_color", "#d8d8d8"), linewidth=0.8)
    ax.spines["polar"].set_color(radar_cfg.get("spine_color", "#555555"))
    ax.spines["polar"].set_linewidth(1.2)

    for model in models:
        values = normalize_values(model["values"], min_value, max_value)
        closed_values = values + values[:1]
        color = model.get("color", None)
        ax.plot(
            angles,
            closed_values,
            label=model.get("name", "Model"),
            color=color,
            linestyle=model.get("linestyle", "-"),
            linewidth=model.get("linewidth", 1.8),
            marker=model.get("marker", "o"),
            markersize=model.get("markersize", 4.5),
        )
        if radar_cfg.get("fill_alpha", 0) > 0:
            ax.fill(angles, closed_values, color=color, alpha=radar_cfg.get("fill_alpha", 0.08))

    title_size = figure_cfg.get("title_size", 16)
    ax.set_title(config.get("title", ""), fontsize=title_size, fontweight="bold", pad=22)
    subtitle = config.get("subtitle", "")
    if subtitle:
        ax.text(0.5, 1.03, subtitle, transform=ax.transAxes, ha="center", va="center", fontsize=10)

    legend = ax.legend(
        loc=config.get("legend_loc", "upper right"),
        bbox_to_anchor=tuple(config.get("legend_bbox_to_anchor", [1.25, 1.15])),
        fontsize=figure_cfg.get("legend_size", 9),
        frameon=True,
    )
    legend.get_frame().set_edgecolor("#b0b0b0")
    legend.get_frame().set_linewidth(0.8)

    output_path = Path(output or config.get("output", DEFAULT_CONFIG["output"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def draw_radar_chart(config: dict[str, Any], output: str | None = None, show: bool = False) -> Path:
    """Draw and save the radar chart using the selected backend."""
    backend = config.get("backend", "svg").lower()
    if backend == "matplotlib":
        return draw_radar_chart_matplotlib(config, output=output, show=show)
    return draw_radar_chart_svg(config, output=output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a YOLO26 model metrics radar chart.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file.")
    parser.add_argument("--output", type=str, default=None, help="Output image path. Overrides config output.")
    parser.add_argument(
        "--backend",
        choices=["svg", "matplotlib"],
        default=None,
        help="Drawing backend. SVG uses only the Python standard library.",
    )
    parser.add_argument("--show", action="store_true", help="Show the chart window after saving with matplotlib.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.backend is not None:
        config["backend"] = args.backend
    output_path = draw_radar_chart(config, output=args.output, show=args.show)
    print(f"Saved radar chart to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
