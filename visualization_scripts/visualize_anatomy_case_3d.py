#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage.measure import marching_cubes


LABELS = {
    1: {"name": "LH", "color": "#e15759"},
    2: {"name": "RH", "color": "#4e79a7"},
    3: {"name": "S", "color": "#59a14f"},
}


def find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "nnUNet").exists() or (parent / "README.md").exists():
            return parent
    raise FileNotFoundError("Could not find project root.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create an interactive 3D HTML view for one anatomy prediction case."
    )
    parser.add_argument("--case", required=True, help="Case id, for example dataset8_007.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output HTML path. Defaults to visualization_outputs/<case>_anatomy_3d.html.",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=3,
        help="Marching-cubes step size. Higher is faster/lighter, lower is more detailed.",
    )
    parser.add_argument(
        "--gt_dir",
        default="nnUNet/nnUNet_raw/Dataset002_anatomy_network/labelsTs",
    )
    parser.add_argument("--lh_pred_dir", default="predictions/Dataset009_anatomy_LH")
    parser.add_argument("--rh_pred_dir", default="predictions/Dataset010_anatomy_RH")
    parser.add_argument("--s_pred_dir", default="predictions/Dataset011_anatomy_S")
    parser.add_argument("--merged_pred_dir", default="predictions/merged_binary_anatomy")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def image_path(folder: Path, case_id: str) -> Path:
    for suffix in (".nii.gz", ".nii", ".mha"):
        path = folder / f"{case_id}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find {case_id} in {folder}")


def read_label(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    spacing_zyx = img.GetSpacing()[::-1]
    return arr, spacing_zyx


def mesh_from_mask(mask: np.ndarray, spacing_zyx, step_size: int):
    if not np.any(mask):
        return None

    padded = np.pad(mask.astype(np.uint8), pad_width=1, mode="constant")
    verts, faces, _, _ = marching_cubes(
        padded,
        level=0.5,
        spacing=spacing_zyx,
        step_size=step_size,
        allow_degenerate=False,
    )

    # Undo the one-voxel pad in physical coordinates.
    verts[:, 0] -= spacing_zyx[0]
    verts[:, 1] -= spacing_zyx[1]
    verts[:, 2] -= spacing_zyx[2]
    return verts, faces


def add_mesh_trace(fig, row, col, arr, spacing_zyx, label_value, name_prefix, step_size, opacity=0.72):
    import plotly.graph_objects as go

    info = LABELS[label_value]
    mesh = mesh_from_mask(arr == label_value, spacing_zyx, step_size)
    if mesh is None:
        return

    verts, faces = mesh
    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 2],
            y=verts[:, 1],
            z=verts[:, 0],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name=f"{name_prefix} {info['name']}",
            color=info["color"],
            opacity=opacity,
            flatshading=True,
            showscale=False,
        ),
        row=row,
        col=col,
    )


def binary_to_label(arr: np.ndarray, label_value: int) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.uint8)
    out[arr == 1] = label_value
    return out


def make_figure(case_id, gt, lh, rh, sacrum, merged, spacing_zyx, step_size):
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=5,
        specs=[[{"type": "scene"}] * 5],
        subplot_titles=[
            "Ground Truth",
            "LH Binary Prediction",
            "RH Binary Prediction",
            "S Binary Prediction",
            "Merged Prediction",
        ],
        horizontal_spacing=0.01,
    )

    for label_value in LABELS:
        add_mesh_trace(fig, 1, 1, gt, spacing_zyx, label_value, "GT", step_size)

    add_mesh_trace(fig, 1, 2, lh, spacing_zyx, 1, "Pred", step_size)
    add_mesh_trace(fig, 1, 3, rh, spacing_zyx, 2, "Pred", step_size)
    add_mesh_trace(fig, 1, 4, sacrum, spacing_zyx, 3, "Pred", step_size)

    for label_value in LABELS:
        add_mesh_trace(fig, 1, 5, merged, spacing_zyx, label_value, "Merged", step_size)

    fig.update_layout(
        title=f"3D Anatomy Check: {case_id}",
        height=850,
        width=2200,
        legend=dict(itemsizing="constant"),
        margin=dict(l=10, r=10, t=70, b=10),
    )

    for idx in range(1, 6):
        scene_name = "scene" if idx == 1 else f"scene{idx}"
        fig.update_layout(
            {
                scene_name: dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z",
                    aspectmode="data",
                )
            }
        )

    return fig


def main():
    args = parse_args()
    case_id = args.case

    gt_path = image_path(resolve(args.gt_dir), case_id)
    lh_path = image_path(resolve(args.lh_pred_dir), case_id)
    rh_path = image_path(resolve(args.rh_pred_dir), case_id)
    s_path = image_path(resolve(args.s_pred_dir), case_id)
    merged_path = image_path(resolve(args.merged_pred_dir), case_id)

    gt, spacing_zyx = read_label(gt_path)
    lh_raw, _ = read_label(lh_path)
    rh_raw, _ = read_label(rh_path)
    s_raw, _ = read_label(s_path)
    merged, _ = read_label(merged_path)

    lh = binary_to_label(lh_raw, 1)
    rh = binary_to_label(rh_raw, 2)
    sacrum = binary_to_label(s_raw, 3)

    fig = make_figure(case_id, gt, lh, rh, sacrum, merged, spacing_zyx, args.step_size)

    out_path = resolve(args.out) if args.out else PROJECT_ROOT / "visualization_outputs" / f"{case_id}_anatomy_3d.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")

    print(f"Saved interactive 3D view: {out_path}")


if __name__ == "__main__":
    main()
