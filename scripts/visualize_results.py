#!/usr/bin/env python3
"""
Generate comparison charts across all training experiments.

Outputs an interactive HTML dashboard with:
  - Training curves overlay (v1 vs v2 vs v3 vs v4)
  - Per-class mAP50 bar chart
  - Confusion matrices side-by-side
  - SAHI comparison results

Usage:
    python scripts/visualize_results.py
"""

import json
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "runs" / "detect"
REPORTS_DIR = BASE_DIR / "reports"

# Experiment configurations
EXPERIMENTS = {
    "v1_baseline": {
        "name": "v1 Baseline (SoccerTrack only)",
        "dir": "hoeherr_v1",
        "color": "#636EFA",
    },
    "v2_combined": {
        "name": "v2 Combined Data",
        "dir": "hoeherr_v2_combined",
        "color": "#EF553B",
    },
    "v3_unfrozen": {
        "name": "v3 Unfrozen Backbone",
        "dir": "hoeherr_v3_unfrozen",
        "color": "#00CC96",
    },
    "v4_yolov8x": {
        "name": "v4 YOLOv8x",
        "dir": "hoeherr_v4_yolov8x",
        "color": "#AB63FA",
    },
}


def find_results_csv(run_dir_name):
    """Find results.csv in various possible locations."""
    candidates = [
        RUNS_DIR / run_dir_name / "results.csv",
        RUNS_DIR / "runs" / "detect" / run_dir_name / "results.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_training_results():
    """Load training results CSVs for all experiments."""
    results = {}
    for key, exp in EXPERIMENTS.items():
        csv_path = find_results_csv(exp["dir"])
        if csv_path:
            df = pd.read_csv(csv_path)
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            results[key] = df
            print(f"Loaded {key}: {len(df)} epochs from {csv_path}")
        else:
            print(f"No results found for {key}")
    return results


def create_training_curves(results):
    """Create training curves overlay plot."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Box Loss (Train)", "Box Loss (Val)",
            "mAP@50", "mAP@50-95"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for key, df in results.items():
        exp = EXPERIMENTS[key]
        epochs = df["epoch"]
        color = exp["color"]
        name = exp["name"]

        # Train box loss
        if "train/box_loss" in df.columns:
            fig.add_trace(
                go.Scatter(x=epochs, y=df["train/box_loss"], name=name,
                          line=dict(color=color), legendgroup=key,
                          showlegend=True),
                row=1, col=1
            )

        # Val box loss
        if "val/box_loss" in df.columns:
            fig.add_trace(
                go.Scatter(x=epochs, y=df["val/box_loss"], name=name,
                          line=dict(color=color), legendgroup=key,
                          showlegend=False),
                row=1, col=2
            )

        # mAP50
        if "metrics/mAP50(B)" in df.columns:
            fig.add_trace(
                go.Scatter(x=epochs, y=df["metrics/mAP50(B)"], name=name,
                          line=dict(color=color), legendgroup=key,
                          showlegend=False),
                row=2, col=1
            )

        # mAP50-95
        if "metrics/mAP50-95(B)" in df.columns:
            fig.add_trace(
                go.Scatter(x=epochs, y=df["metrics/mAP50-95(B)"], name=name,
                          line=dict(color=color), legendgroup=key,
                          showlegend=False),
                row=2, col=2
            )

    fig.update_layout(
        height=600,
        title_text="Training Curves Comparison",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Epoch")

    return fig


def create_final_metrics_bar(results):
    """Create per-class mAP50 bar chart from final epoch."""
    metrics = []
    for key, df in results.items():
        exp = EXPERIMENTS[key]
        last = df.iloc[-1]

        metrics.append({
            "Experiment": exp["name"],
            "mAP50": last.get("metrics/mAP50(B)", 0),
            "mAP50-95": last.get("metrics/mAP50-95(B)", 0),
            "Precision": last.get("metrics/precision(B)", 0),
            "Recall": last.get("metrics/recall(B)", 0),
        })

    df_metrics = pd.DataFrame(metrics)

    fig = go.Figure()
    for col in ["mAP50", "mAP50-95", "Precision", "Recall"]:
        fig.add_trace(go.Bar(
            x=df_metrics["Experiment"],
            y=df_metrics[col],
            name=col,
            text=[f"{v:.3f}" for v in df_metrics[col]],
            textposition="auto",
        ))

    fig.update_layout(
        title="Final Metrics Comparison",
        barmode="group",
        height=450,
        template="plotly_white",
        yaxis_title="Score",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )

    return fig


def create_lr_curves(results):
    """Create learning rate schedule comparison."""
    fig = go.Figure()

    for key, df in results.items():
        exp = EXPERIMENTS[key]
        if "lr/pg0" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["epoch"], y=df["lr/pg0"],
                name=exp["name"], line=dict(color=exp["color"]),
            ))

    fig.update_layout(
        title="Learning Rate Schedule",
        xaxis_title="Epoch",
        yaxis_title="Learning Rate",
        height=350,
        template="plotly_white",
    )

    return fig


def create_loss_breakdown(results):
    """Create combined loss breakdown (box + cls + dfl)."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Box Loss", "Classification Loss", "DFL Loss"),
        horizontal_spacing=0.06,
    )

    for key, df in results.items():
        exp = EXPERIMENTS[key]
        color = exp["color"]

        for i, loss_col in enumerate(["val/box_loss", "val/cls_loss", "val/dfl_loss"], 1):
            if loss_col in df.columns:
                fig.add_trace(
                    go.Scatter(x=df["epoch"], y=df[loss_col], name=exp["name"],
                              line=dict(color=color), legendgroup=key,
                              showlegend=(i == 1)),
                    row=1, col=i
                )

    fig.update_layout(
        title="Validation Loss Breakdown",
        height=350,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )

    return fig


def embed_confusion_matrix(run_dir_name):
    """Load confusion matrix image and embed as base64."""
    candidates = [
        RUNS_DIR / run_dir_name / "confusion_matrix_normalized.png",
        RUNS_DIR / run_dir_name / "confusion_matrix.png",
        RUNS_DIR / "runs" / "detect" / run_dir_name / "confusion_matrix_normalized.png",
        RUNS_DIR / "runs" / "detect" / run_dir_name / "confusion_matrix.png",
    ]
    for c in candidates:
        if c.exists():
            with open(c, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    return None


def create_sahi_comparison():
    """Create SAHI comparison visualization if results exist."""
    sahi_path = REPORTS_DIR / "sahi_comparison.json"
    if not sahi_path.exists():
        return None

    with open(sahi_path) as f:
        data = json.load(f)

    categories = ["Player AP@50", "Ball AP@50", "mAP@50"]
    standard = [data["standard"]["player_ap50"], data["standard"]["ball_ap50"], data["standard"]["map50"]]
    sahi = [data["sahi"]["player_ap50"], data["sahi"]["ball_ap50"], data["sahi"]["map50"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories, y=standard, name="Standard",
        text=[f"{v:.4f}" for v in standard], textposition="auto",
        marker_color="#636EFA",
    ))
    fig.add_trace(go.Bar(
        x=categories, y=sahi, name="SAHI",
        text=[f"{v:.4f}" for v in sahi], textposition="auto",
        marker_color="#EF553B",
    ))

    fig.update_layout(
        title=f"SAHI vs Standard Inference (slice={data['sahi']['slice_size']}, overlap={data['sahi']['overlap']})",
        barmode="group",
        height=400,
        template="plotly_white",
        yaxis_title="AP@50",
    )

    return fig


def generate_html(figures, confusion_matrices):
    """Generate a self-contained HTML report."""
    html_parts = ["""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Results Comparison - Hoeherr Football Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               background: #f5f7fa; color: #333; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        h1 { text-align: center; padding: 20px; color: #1a1a2e;
             border-bottom: 3px solid #636EFA; margin-bottom: 30px; }
        h2 { color: #1a1a2e; margin: 30px 0 15px; padding-bottom: 8px;
             border-bottom: 2px solid #e0e0e0; }
        .chart-container { background: white; border-radius: 8px;
                          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                          padding: 20px; margin-bottom: 25px; }
        .cm-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                   gap: 20px; margin-bottom: 25px; }
        .cm-card { background: white; border-radius: 8px;
                   box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                   padding: 15px; text-align: center; }
        .cm-card h3 { margin-bottom: 10px; color: #555; }
        .cm-card img { max-width: 100%; height: auto; border-radius: 4px; }
    </style>
</head>
<body>
<div class="container">
    <h1>Training Results Comparison</h1>
"""]

    # Add each figure
    for i, (title, fig) in enumerate(figures):
        if fig is not None:
            div_id = f"chart_{i}"
            html_parts.append(f'<h2>{title}</h2>')
            html_parts.append(f'<div class="chart-container"><div id="{div_id}"></div></div>')
            html_parts.append(f'<script>Plotly.newPlot("{div_id}", {fig.to_json()});</script>')

    # Confusion matrices
    if any(v is not None for v in confusion_matrices.values()):
        html_parts.append('<h2>Confusion Matrices</h2>')
        html_parts.append('<div class="cm-grid">')
        for key, b64 in confusion_matrices.items():
            if b64:
                exp = EXPERIMENTS[key]
                html_parts.append(f'''
                <div class="cm-card">
                    <h3>{exp["name"]}</h3>
                    <img src="data:image/png;base64,{b64}" alt="Confusion Matrix - {exp['name']}">
                </div>''')
        html_parts.append('</div>')

    html_parts.append('</div></body></html>')
    return "\n".join(html_parts)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training results...")
    results = load_training_results()

    if not results:
        print("No training results found. Exiting.")
        return

    print("\nGenerating visualizations...")

    figures = []

    # Training curves
    fig = create_training_curves(results)
    figures.append(("Training Curves", fig))

    # Final metrics bar chart
    fig = create_final_metrics_bar(results)
    figures.append(("Final Metrics", fig))

    # Loss breakdown
    fig = create_loss_breakdown(results)
    figures.append(("Loss Breakdown", fig))

    # Learning rate
    fig = create_lr_curves(results)
    figures.append(("Learning Rate Schedule", fig))

    # SAHI comparison
    sahi_fig = create_sahi_comparison()
    if sahi_fig:
        figures.append(("SAHI vs Standard Inference", sahi_fig))

    # Confusion matrices
    print("Loading confusion matrices...")
    confusion_matrices = {}
    for key, exp in EXPERIMENTS.items():
        confusion_matrices[key] = embed_confusion_matrix(exp["dir"])

    # Generate HTML
    print("Generating HTML report...")
    html = generate_html(figures, confusion_matrices)

    output_path = REPORTS_DIR / "results_comparison.html"
    with open(output_path, "w") as f:
        f.write(html)

    print(f"\nReport saved to: {output_path}")
    print(f"Open in browser to view interactive charts.")


if __name__ == "__main__":
    main()
