#!/usr/bin/env python3

import argparse
import subprocess
import sys


STEPS = [
    ["python", "data_preparation_scripts/align_and_convert_data.py"],
    ["python", "data_preparation_scripts/data_splitting.py"],
    ["python", "data_preparation_scripts/create_nnunet_structure.py"],
    ["python", "label_engineering_scripts/generate_dataset_anatomy.py"],
    ["python", "train_val_test_nnunet_scripts/create_train_val_split.py", "--dataset_id", "2"],
    ["python", "train_val_test_nnunet_scripts/preprocess.py", "--dataset_id", "2"],
    ["python", "train_val_test_nnunet_scripts/train.py", "--dataset_id", "2", "--fold", "0", "--gpu", "0"],
    ["python", "train_val_test_nnunet_scripts/predict.py", "--dataset_id", "2", "--gpu", "0"],
    ["python", "evaluation_scripts/eval_anatomy_folder.py"],
]


def run(cmd):
    print("\n" + "=" * 80)
    print("RUN:", " ".join(cmd))
    print("=" * 80 + "\n")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run anatomy-only nnU-Net pipeline.")
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    steps = STEPS.copy()

    if args.skip_prepare:
        steps = steps[3:]

    if args.skip_train:
        steps = [s for s in steps if "train.py" not in s]

    for step in steps:
        step = [args.gpu if x == "0" and "--gpu" in step else x for x in step]
        run(step)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nPipeline failed at: {' '.join(e.cmd)}")
        sys.exit(e.returncode)