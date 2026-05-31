import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run nnU-Net preprocessing (foreground, no nohup)."
    )
    parser.add_argument("--dataset_id", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_id = args.dataset_id

    project_root = Path(__file__).resolve().parent.parent

    nnunet_raw = project_root / "nnUNet" / "nnUNet_raw"
    nnunet_preprocessed = project_root / "nnUNet" / "nnUNet_preprocessed"
    nnunet_results = project_root / "nnUNet" / "nnUNet_results"

    cmd = f"""
    export nnUNet_raw=\"{nnunet_raw}\" && \\
    export nnUNet_preprocessed=\"{nnunet_preprocessed}\" && \\
    export nnUNet_results=\"{nnunet_results}\" && \\
    export OMP_NUM_THREADS=1 && \\
    export MKL_NUM_THREADS=1 && \\
    export OPENBLAS_NUM_THREADS=1 && \\
    nnUNetv2_plan_and_preprocess -d {dataset_id} -c 3d_fullres --verify_dataset_integrity -np 1 -npfp 1
    """

    print(f"\n=== PREPROCESS START ===")
    print(f"Project root: {project_root}")
    print(f"Dataset: {dataset_id}\n")

    subprocess.run(["bash", "-c", cmd], check=True)

    print("\n=== PREPROCESS DONE ===\n")


if __name__ == "__main__":
    main()
