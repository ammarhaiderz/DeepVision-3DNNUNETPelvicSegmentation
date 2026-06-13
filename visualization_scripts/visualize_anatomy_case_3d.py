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
    parser.add_argument(
        "--show_probabilities",
        action="store_true",
        help="Color the normal 3D prediction surfaces by foreground probability.",
    )
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


def load_probability_map(folder: Path, case_id: str) -> np.ndarray | None:
    """Load probability map from .npz file if it exists."""
    npz_path = folder / f"{case_id}.npz"
    if not npz_path.is_file():
        return None

    try:
        with np.load(npz_path) as data:
            key = "probabilities" if "probabilities" in data else "softmax"
            if key not in data:
                print(
                    f"Warning: No probability array found in {npz_path}; "
                    f"available keys: {data.files}"
                )
                return None

            probs = data[key]
            if probs.ndim != 4:
                raise ValueError(
                    f"Expected (C, Z, Y, X), got probability shape {probs.shape}"
                )
            if probs.shape[0] < 2:
                raise ValueError(
                    f"Expected background and foreground channels, got {probs.shape[0]}"
                )

            return probs[1].copy()
    except Exception as e:
        print(f"Warning: Could not load probability from {npz_path}: {e}")
        return None


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


def probabilities_for_surface_vertices(
    vertices: np.ndarray,
    mask: np.ndarray,
    probabilities: np.ndarray,
    spacing_zyx,
) -> np.ndarray:
    voxel_vertices = vertices / np.asarray(spacing_zyx, dtype=np.float32)
    lower = np.floor(voxel_vertices).astype(np.int32)
    upper = np.ceil(voxel_vertices).astype(np.int32)

    best_distance = np.full(len(vertices), np.inf, dtype=np.float32)
    values = np.full(len(vertices), np.nan, dtype=np.float32)

    for z_choice in (0, 1):
        for y_choice in (0, 1):
            for x_choice in (0, 1):
                indices = np.column_stack(
                    (
                        upper[:, 0] if z_choice else lower[:, 0],
                        upper[:, 1] if y_choice else lower[:, 1],
                        upper[:, 2] if x_choice else lower[:, 2],
                    )
                )
                indices = np.clip(indices, 0, np.asarray(mask.shape) - 1)
                is_foreground = mask[
                    indices[:, 0], indices[:, 1], indices[:, 2]
                ]
                distance = np.sum((voxel_vertices - indices) ** 2, axis=1)
                use = is_foreground & (distance < best_distance)
                best_distance[use] = distance[use]
                values[use] = probabilities[
                    indices[use, 0], indices[use, 1], indices[use, 2]
                ]

    if np.isnan(values).any():
        raise RuntimeError("Could not map every mesh vertex to a foreground voxel.")
    return values


def add_mesh_trace(
    fig,
    row,
    col,
    arr,
    spacing_zyx,
    label_value,
    name_prefix,
    step_size,
    opacity=0.72,
    probability_map=None,
    show_probability_scale=False,
):
    import plotly.graph_objects as go

    info = LABELS[label_value]
    mesh = mesh_from_mask(arr == label_value, spacing_zyx, step_size)
    if mesh is None:
        return

    verts, faces = mesh

    trace = go.Mesh3d(
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
    )

    if probability_map is not None:
        values = probabilities_for_surface_vertices(
            verts,
            arr == label_value,
            probability_map,
            spacing_zyx,
        )
        trace.update(
            color=None,
            intensity=values,
            intensitymode="vertex",
            colorscale="RdYlGn",
            cmin=0.5,
            cmax=1.0,
            showscale=show_probability_scale,
            colorbar=dict(title="P(class)", x=1.01)
            if show_probability_scale
            else None,
            customdata=values,
            hovertemplate="P(class)=%{customdata:.4f}<extra></extra>",
        )

    fig.add_trace(trace, row=row, col=col)


def binary_to_label(arr: np.ndarray, label_value: int) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.uint8)
    out[arr == 1] = label_value
    return out


def make_figure(
    case_id,
    gt,
    lh,
    rh,
    sacrum,
    merged,
    spacing_zyx,
    step_size,
    lh_prob=None,
    rh_prob=None,
    s_prob=None,
):
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

    prediction_step = 1 if lh_prob is not None else step_size
    add_mesh_trace(
        fig, 1, 2, lh, spacing_zyx, 1, "Pred", prediction_step,
        probability_map=lh_prob,
    )
    add_mesh_trace(
        fig, 1, 3, rh, spacing_zyx, 2, "Pred", prediction_step,
        probability_map=rh_prob,
    )
    add_mesh_trace(
        fig, 1, 4, sacrum, spacing_zyx, 3, "Pred", prediction_step,
        probability_map=s_prob,
        show_probability_scale=s_prob is not None,
    )

    for label_value in LABELS:
        class_probability = {
            1: lh_prob,
            2: rh_prob,
            3: s_prob,
        }[label_value]
        add_mesh_trace(
            fig, 1, 5, merged, spacing_zyx, label_value, "Merged",
            prediction_step, probability_map=class_probability,
        )

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

    image_shapes = {
        "ground truth": gt.shape,
        "LH prediction": lh_raw.shape,
        "RH prediction": rh_raw.shape,
        "S prediction": s_raw.shape,
        "merged prediction": merged.shape,
    }
    if len(set(image_shapes.values())) != 1:
        raise ValueError(f"Image shapes do not match: {image_shapes}")

    lh_prob = None
    rh_prob = None
    s_prob = None
    if args.show_probabilities:
        print("Loading probability maps...")
        lh_prob = load_probability_map(resolve(args.lh_pred_dir), case_id)
        rh_prob = load_probability_map(resolve(args.rh_pred_dir), case_id)
        s_prob = load_probability_map(resolve(args.s_pred_dir), case_id)

        if lh_prob is not None:
            print(f"  LH probabilities: shape {lh_prob.shape}")
        if rh_prob is not None:
            print(f"  RH probabilities: shape {rh_prob.shape}")
        if s_prob is not None:
            print(f"  S probabilities: shape {s_prob.shape}")

        probability_maps = (lh_prob, rh_prob, s_prob)
        if not all(prob is not None for prob in probability_maps):
            raise FileNotFoundError(
                "Probability mode requires matching .npz files for LH, RH, and S."
            )

        expected_shape = gt.shape
        if any(prob.shape != expected_shape for prob in probability_maps):
            shapes = [prob.shape for prob in probability_maps]
            raise ValueError(
                f"Probability shapes {shapes} do not match image shape "
                f"{expected_shape}"
            )

    lh = binary_to_label(lh_raw, 1)
    rh = binary_to_label(rh_raw, 2)
    sacrum = binary_to_label(s_raw, 3)
    fig = make_figure(
        case_id,
        gt,
        lh,
        rh,
        sacrum,
        merged,
        spacing_zyx,
        args.step_size,
        lh_prob=lh_prob,
        rh_prob=rh_prob,
        s_prob=s_prob,
    )
    default_name = (
        f"{case_id}_anatomy_3d_probabilities.html"
        if args.show_probabilities
        else f"{case_id}_anatomy_3d.html"
    )

    out_path = (
        resolve(args.out)
        if args.out
        else PROJECT_ROOT / "visualization_outputs" / default_name
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")

    print(f"Saved interactive visualization: {out_path}")


if __name__ == "__main__":
    main()
