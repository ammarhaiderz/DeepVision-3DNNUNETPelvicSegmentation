import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run nnU-Net training (foreground, no nohup)."
    )

    parser.add_argument("--dataset_id", type=int, required=True)
    parser.add_argument("--fold", type=str, default="0")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--config", type=str, default="3d_fullres")
    parser.add_argument("--trainer", type=str, default=None)
    parser.add_argument(
        "--continue_training",
        action="store_true",
        help="Resume from checkpoint_latest.pth if it exists.",
    )

    return parser.parse_args()


def detect_trainer(dataset_id: int, nnunet_preprocessed: Path) -> str:
    prefix = f"Dataset{dataset_id:03d}_"

    for folder in nnunet_preprocessed.iterdir():
        if folder.is_dir() and folder.name.startswith(prefix):
            name = folder.name.lower()
            if "anatomy" in name:
                return "nnUNetTrainer_no_rotation"
            return "nnUNetTrainer_1000"

    raise RuntimeError(f"Dataset {dataset_id} not found in {nnunet_preprocessed}")


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent

    nnunet_raw = project_root / "nnUNet" / "nnUNet_raw"
    nnunet_preprocessed = project_root / "nnUNet" / "nnUNet_preprocessed"
    nnunet_results = project_root / "nnUNet" / "nnUNet_results"

    trainer = args.trainer or detect_trainer(args.dataset_id, nnunet_preprocessed)

    base_cmd = f'nnUNetv2_train {args.dataset_id} {args.config} {args.fold} -tr {trainer}'
    if args.continue_training:
        base_cmd = f"{base_cmd} --c"

    if args.gpu is not None:
        base_cmd = f'CUDA_VISIBLE_DEVICES={args.gpu} {base_cmd}'

    cmd = f"""
    export nnUNet_raw=\"{nnunet_raw}\" && \\
    export nnUNet_preprocessed=\"{nnunet_preprocessed}\" && \\
    export nnUNet_results=\"{nnunet_results}\" && \\
    {base_cmd}
    """

    print("\n=== TRAINING START ===")
    print(f"Project root: {project_root}")
    print(f"Dataset: {args.dataset_id}")
    print(f"Fold: {args.fold}")
    print(f"Trainer: {trainer}\n")
    print(f"Continue training: {args.continue_training}\n")

    subprocess.run(["bash", "-c", cmd], check=True)

    print("\n=== TRAINING DONE ===\n")


if __name__ == "__main__":
    main()
