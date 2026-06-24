import os
import subprocess
import sys
from pathlib import Path


def run(cmd, shell=False):
    print(f"\n[RUN] {cmd if isinstance(cmd, str) else ' '.join(cmd)}\n")
    subprocess.run(cmd, shell=shell, check=True)


def main():
    print("=== nnU-Net environment setup (Python) ===")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    env_dir = project_root / "nnunet_env"

    requirements = script_dir / "requirements.txt"
    trainer_1000 = script_dir / "nnUNetTrainer_1000.py"
    trainer_no_rot = script_dir / "nnUNetTrainer_no_rotation.py"

    for f in [requirements, trainer_1000, trainer_no_rot]:
        if not f.exists():
            raise FileNotFoundError(f"Missing required file: {f}")

    if not env_dir.exists():
        print("Creating virtual environment...")
        run([sys.executable, "-m", "venv", str(env_dir)])
    else:
        print("Virtual environment already exists, reusing it.")

    python_bin = env_dir / "bin" / "python"
    pip_bin = env_dir / "bin" / "pip"

    if not python_bin.exists():
        raise RuntimeError("Could not find python inside venv.")

    run([str(python_bin), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    print("Installing PyTorch with the CUDA 12.1 runtime...")
    run([
        str(pip_bin),
        "install",
        "--index-url",
        "https://download.pytorch.org/whl/cu121",
        "torch==2.5.1",
        "torchvision==0.20.1",
    ])

    print("Installing requirements...")
    run([str(pip_bin), "install", "-r", str(requirements)])

    print("Locating nnunetv2 package...")

    code = """
import inspect
import nnunetv2
from pathlib import Path
print(Path(inspect.getfile(nnunetv2)).resolve().parent)
"""

    result = subprocess.run(
        [str(python_bin), "-c", code],
        capture_output=True,
        text=True,
        check=True
    )

    nnunet_dir = Path(result.stdout.strip())

    if not nnunet_dir.exists():
        raise RuntimeError("Could not locate nnunetv2 package")

    trainer_target = nnunet_dir / "training" / "nnUNetTrainer"

    if not trainer_target.exists():
        raise RuntimeError(f"Trainer folder not found: {trainer_target}")

    print(f"nnunetv2 located at: {nnunet_dir}")
    print(f"Trainer target dir: {trainer_target}")

    print("Copying custom trainers...")

    import shutil
    shutil.copy(trainer_1000, trainer_target / "nnUNetTrainer_1000.py")
    shutil.copy(trainer_no_rot, trainer_target / "nnUNetTrainer_no_rotation.py")

    print("Verifying trainer import...")

    test_code = """
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_1000 import nnUNetTrainer_1000
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_no_rotation import nnUNetTrainer_no_rotation
print("Custom trainers imported successfully.")
"""

    run([str(python_bin), "-c", test_code])

    activate_script = project_root / "activate_nnunet_env.sh"

    activate_script.write_text(f"""#!/usr/bin/env bash
source "{env_dir}/bin/activate"
""")

    os.chmod(activate_script, 0o755)

    print("\n=== SETUP COMPLETE ===")
    print(f"Project root: {project_root}")
    print(f"Environment: {env_dir}")
    print("Activate with:")
    print(f"  source {env_dir}/bin/activate")
    print("or:")
    print(f"  source {activate_script}")


if __name__ == "__main__":
    main()
