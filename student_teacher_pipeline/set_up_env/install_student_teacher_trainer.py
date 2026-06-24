#!/usr/bin/env python3
from __future__ import annotations

import inspect
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    trainer = script_dir / "nnUNetTrainer_student_teacher.py"
    if not trainer.is_file():
        raise FileNotFoundError(f"Missing trainer file: {trainer}")

    code = """
import inspect
import nnunetv2
from pathlib import Path
print(Path(inspect.getfile(nnunetv2)).resolve().parent)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    nnunet_dir = Path(result.stdout.strip())
    target_dir = nnunet_dir / "training" / "nnUNetTrainer"
    if not target_dir.is_dir():
        raise RuntimeError(f"Could not find nnU-Net trainer directory: {target_dir}")

    shutil.copyfile(trainer, target_dir / trainer.name)

    test_code = """
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_student_teacher import nnUNetTrainer_student_teacher
print("Student-teacher trainer imported successfully.")
"""
    subprocess.run([sys.executable, "-c", test_code], check=True)
    print(f"Installed {trainer.name} into {target_dir}")


if __name__ == "__main__":
    main()

