#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ENV = PROJECT_ROOT / "nnunet_env"
PROJECT_PYTHON = PROJECT_ENV / "bin" / "python"

if Path(sys.prefix).resolve() != PROJECT_ENV.resolve():
    if not PROJECT_PYTHON.is_file():
        raise FileNotFoundError(
            f"Project Python environment not found: {PROJECT_PYTHON}"
        )
    os.execv(
        str(PROJECT_PYTHON),
        [str(PROJECT_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
    )

from common import nnunet_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve local nnU-Net TensorBoard logs for VS Code forwarding."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()

    _, _, results = nnunet_paths()
    command = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(results),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    print("RUN:", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
