import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run nnU-Net prediction (foreground, no nohup)."
    )

    parser.add_argument("--dataset_id", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--save_probabilities",
        action="store_true",
        help="Save nnU-Net probability maps (.npz) next to hard predictions.",
    )

    return parser.parse_args()


def find_dataset_folder(dataset_id: int, root: Path):
    prefix = f"Dataset{dataset_id:03d}_"
    for f in root.iterdir():
        if f.is_dir() and f.name.startswith(prefix):
            return f
    raise RuntimeError(f"Dataset {dataset_id} not found in {root}")


def detect_trainer(dataset_folder: Path):
    if "anatomy" in dataset_folder.name.lower():
        return "nnUNetTrainer_no_rotation"
    return "nnUNetTrainer_1000"


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent

    nnunet_root = project_root / "nnUNet"
    nnunet_raw = nnunet_root / "nnUNet_raw"
    nnunet_preprocessed = nnunet_root / "nnUNet_preprocessed"
    nnunet_results = nnunet_root / "nnUNet_results"

    dataset_folder = find_dataset_folder(args.dataset_id, nnunet_preprocessed)
    dataset_name = dataset_folder.name
    trainer = detect_trainer(dataset_folder)

    input_dir = nnunet_raw / dataset_name / "imagesTs"
    if not input_dir.exists():
        raise RuntimeError(f"Missing imagesTs folder: {input_dir}")

    output_dir = project_root / "predictions" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = (
        f'nnUNetv2_predict '
        f'-i "{input_dir}" '
        f'-o "{output_dir}" '
        f'-d {args.dataset_id} '
        f'-c 3d_fullres '
        f'-f 0 '
        f'-tr {trainer} '
        f'-chk checkpoint_best.pth'
    )
    if args.save_probabilities:
        base_cmd = f"{base_cmd} --save_probabilities"

    if args.gpu is not None:
        base_cmd = f'CUDA_VISIBLE_DEVICES={args.gpu} {base_cmd}'

    cmd = f"""
    export nnUNet_raw=\"{nnunet_raw}\" && \\
    export nnUNet_preprocessed=\"{nnunet_preprocessed}\" && \\
    export nnUNet_results=\"{nnunet_results}\" && \\
    {base_cmd}
    """

    print("\n=== PREDICTION START ===")
    print(f"Project root: {project_root}")
    print(f"Dataset: {args.dataset_id}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}\n")

    subprocess.run(["bash", "-c", cmd], check=True)

    print("\n=== PREDICTION DONE ===\n")


if __name__ == "__main__":
    main()
