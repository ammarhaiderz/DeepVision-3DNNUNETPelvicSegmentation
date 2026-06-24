#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage.measure import marching_cubes


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare a fine-tuned LH prediction on erased and normal inputs "
            "against the binary ground truth."
        )
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--step_size", type=int, default=3)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def read_mask(path: Path):
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image) > 0, image.GetSpacing()[::-1]


def dice(prediction: np.ndarray, reference: np.ndarray) -> float:
    denominator = prediction.sum() + reference.sum()
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(prediction, reference).sum() / denominator)


def mesh_from_mask(mask: np.ndarray, spacing, step_size: int):
    if not mask.any():
        return None
    padded = np.pad(mask.astype(np.uint8), 1)
    vertices, faces, _, _ = marching_cubes(
        padded,
        level=0.5,
        spacing=spacing,
        step_size=step_size,
        allow_degenerate=False,
    )
    vertices -= np.asarray(spacing)
    return vertices, faces


def add_mesh(
    figure,
    col: int,
    mask: np.ndarray,
    spacing,
    step_size: int,
    name: str,
    color: str,
    opacity: float,
):
    import plotly.graph_objects as go

    surface = mesh_from_mask(mask, spacing, step_size)
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
        row=1,
        col=col,
    )


def main():
    args = parse_args()
    case = args.case
    ground_truth_path = (
        PROJECT_ROOT
        / "nnUNet"
        / "nnUNet_raw"
        / "Dataset017_PENGWIN_erased_LH"
        / "labelsTs"
        / f"{case}.nii.gz"
    )
    prediction_root = (
        PROJECT_ROOT / "lh_pretraining_smoke_test" / "artifacts" / "predictions"
    )
    erased_path = prediction_root / "erased_input_single" / f"{case}.nii.gz"
    normal_path = prediction_root / "normal_input_single" / f"{case}.nii.gz"
    for path in (ground_truth_path, erased_path, normal_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    ground_truth, spacing = read_mask(ground_truth_path)
    erased, erased_spacing = read_mask(erased_path)
    normal, normal_spacing = read_mask(normal_path)
    if not (
        ground_truth.shape == erased.shape == normal.shape
        and np.allclose(spacing, erased_spacing)
        and np.allclose(spacing, normal_spacing)
    ):
        raise ValueError("Prediction and ground-truth geometries do not match.")

    erased_dice = dice(erased, ground_truth)
    normal_dice = dice(normal, ground_truth)

    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=1,
        cols=5,
        specs=[[{"type": "scene"}] * 5],
        subplot_titles=[
            "Ground Truth",
            f"Erased Input Prediction<br>Dice {erased_dice:.4f}",
            "GT + Erased Prediction",
            f"Normal Input Prediction<br>Dice {normal_dice:.4f}",
            "GT + Normal Prediction",
        ],
        horizontal_spacing=0.01,
    )
    add_mesh(
        figure, 1, ground_truth, spacing, args.step_size,
        "Ground truth", "#59a14f", 0.82,
    )
    add_mesh(
        figure, 2, erased, spacing, args.step_size,
        "Erased prediction", "#4e79a7", 0.82,
    )
    add_mesh(
        figure, 3, ground_truth, spacing, args.step_size,
        "Ground truth", "#59a14f", 0.48,
    )
    add_mesh(
        figure, 3, erased, spacing, args.step_size,
        "Erased prediction", "#4e79a7", 0.48,
    )
    add_mesh(
        figure, 4, normal, spacing, args.step_size,
        "Normal prediction", "#e15759", 0.82,
    )
    add_mesh(
        figure, 5, ground_truth, spacing, args.step_size,
        "Ground truth", "#59a14f", 0.48,
    )
    add_mesh(
        figure, 5, normal, spacing, args.step_size,
        "Normal prediction", "#e15759", 0.48,
    )

    figure.update_layout(
        title=f"Fine-Tuned LH Comparison: {case}",
        height=850,
        width=2200,
        margin=dict(l=10, r=10, t=85, b=10),
        legend=dict(itemsizing="constant"),
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
        / f"{case}_finetuned_LH_erased_vs_normal.html"
    )
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output), include_plotlyjs="cdn")
    print(f"Erased-input Dice: {erased_dice:.4f}")
    print(f"Normal-input Dice: {normal_dice:.4f}")
    print(f"Saved interactive visualization: {output}")


if __name__ == "__main__":
    main()
