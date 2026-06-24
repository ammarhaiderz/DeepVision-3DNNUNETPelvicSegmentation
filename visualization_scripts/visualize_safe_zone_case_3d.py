#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage.measure import marching_cubes


DATASETS = {
    9: {"bone": "LH", "dataset_name": "Dataset009_anatomy_LH"},
    10: {"bone": "RH", "dataset_name": "Dataset010_anatomy_RH"},
    11: {"bone": "S", "dataset_name": "Dataset011_anatomy_S"},
}
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize original and safe-zone-filtered binary prediction."
    )
    parser.add_argument("--dataset_id", type=int, choices=sorted(DATASETS), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--distance_mm",
        type=float,
        default=None,
        help=(
            "Safe-zone distance to visualize. If omitted, use the distance "
            "written in the experiment summary."
        ),
    )
    parser.add_argument("--step_size", type=int, default=3)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def read(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image) > 0, image.GetSpacing()[::-1]


def mesh(mask: np.ndarray, spacing, step_size: int):
    if not mask.any():
        return None
    padded = np.pad(mask.astype(np.uint8), 1)
    try:
        vertices, faces, _, _ = marching_cubes(
            padded,
            level=0.5,
            spacing=spacing,
            step_size=step_size,
            allow_degenerate=False,
        )
    except RuntimeError:
        if step_size == 1:
            return None
        try:
            vertices, faces, _, _ = marching_cubes(
                padded,
                level=0.5,
                spacing=spacing,
                step_size=1,
                allow_degenerate=False,
            )
        except RuntimeError:
            return None
    vertices -= np.asarray(spacing)
    return vertices, faces


def add_mesh(
    figure,
    row: int,
    col: int,
    mask: np.ndarray,
    spacing,
    step_size: int,
    name: str,
    color: str,
    opacity: float,
):
    import plotly.graph_objects as go

    surface = mesh(mask, spacing, step_size)
    if surface is None:
        return
    vertices, faces = surface
    figure.add_trace(
        go.Mesh3d(
            x=vertices[:, 2],
            y=vertices[:, 1],
            z=vertices[:, 0],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name=name,
            color=color,
            opacity=opacity,
            flatshading=True,
            showscale=False,
        ),
        row=row,
        col=col,
    )


def main():
    args = parse_args()
    config = DATASETS[args.dataset_id]
    case = args.case
    experiment_root = (
        PROJECT_ROOT / "safe_zone_experiments" / config["bone"]
    )
    distance_mm = args.distance_mm
    if distance_mm is None:
        summary_path = experiment_root / "experiment_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"Experiment summary not found: {summary_path}. "
                "Run evaluate_safe_zone_filter.py first or pass --distance_mm."
            )
        with summary_path.open("r", encoding="utf-8") as file:
            distance_mm = float(json.load(file)["written_distance_mm"])
    distance_name = f"{distance_mm:g}mm"
    experiment = experiment_root / distance_name
    pred_path = PROJECT_ROOT / "predictions" / config["dataset_name"] / f"{case}.nii.gz"
    gt_path = (
        PROJECT_ROOT
        / "nnUNet"
        / "nnUNet_raw"
        / config["dataset_name"]
        / "labelsTs"
        / f"{case}.nii.gz"
    )
    paths = {
        "filtered": experiment / "filtered" / f"{case}.nii.gz",
        "largest": experiment / "largest" / f"{case}.nii.gz",
        "safe_zone": experiment / "safe_zone" / f"{case}.nii.gz",
        "removed": experiment / "removed" / f"{case}.nii.gz",
    }
    for path in (pred_path, gt_path, *paths.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    original, spacing = read(pred_path)
    ground_truth, gt_spacing = read(gt_path)
    arrays = {name: read(path)[0] for name, path in paths.items()}
    if not np.allclose(spacing, gt_spacing):
        raise ValueError("Prediction and ground-truth spacing differ.")
    shell = arrays["safe_zone"] & ~arrays["largest"]

    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=1,
        cols=5,
        specs=[[{"type": "scene"}] * 5],
        subplot_titles=[
            "Ground Truth",
            "Original Prediction",
            "Largest Component + Safe Zone",
            "Filtered Prediction",
            "Removed Artifacts",
        ],
        horizontal_spacing=0.01,
    )
    add_mesh(figure, 1, 1, ground_truth, spacing, args.step_size, "GT", "#59a14f", 0.75)
    add_mesh(figure, 1, 2, original, spacing, args.step_size, "Original", "#4e79a7", 0.75)
    add_mesh(
        figure, 1, 3, arrays["largest"], spacing, args.step_size,
        "Largest component", "#f28e2b", 0.9,
    )
    add_mesh(
        figure, 1, 3, shell, spacing, args.step_size,
        "Safe-zone shell", "#edc948", 0.18,
    )
    add_mesh(
        figure, 1, 4, arrays["filtered"], spacing, args.step_size,
        "Filtered", "#59a14f", 0.8,
    )
    add_mesh(
        figure, 1, 5, arrays["removed"], spacing, args.step_size,
        "Removed", "#e15759", 0.9,
    )
    figure.update_layout(
        title=(
            f"{config['bone']} Safe-Zone Postprocessing: {case}, "
            f"{distance_mm:g} mm"
        ),
        height=850,
        width=2200,
        margin=dict(l=10, r=10, t=70, b=10),
    )
    for index in range(1, 6):
        scene = "scene" if index == 1 else f"scene{index}"
        figure.update_layout(
            {
                scene: dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z",
                    aspectmode="data",
                )
            }
        )
    output = (
        Path(args.out)
        if args.out
        else PROJECT_ROOT
        / "visualization_outputs"
        / f"{case}_{config['bone']}_safe_zone_{distance_name}.html"
    )
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output), include_plotlyjs="cdn")
    print(f"Saved interactive visualization: {output}")


if __name__ == "__main__":
    main()
