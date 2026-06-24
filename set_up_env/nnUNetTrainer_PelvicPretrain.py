from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_no_rotation import (
    nnUNetTrainer_no_rotation,
)


class nnUNetTrainer_PelvicPretrain(nnUNetTrainer_no_rotation):
    """Pelvic pretraining trainer with durable checkpoints and TensorBoard."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.save_every = 10

    def on_train_start(self):
        super().on_train_start()
        if self.local_rank != 0:
            return
        metadata = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "trainer": self.__class__.__name__,
            "dataset": self.plans_manager.dataset_name,
            "configuration": self.configuration_name,
            "fold": self.fold,
            "num_epochs": self.num_epochs,
            "save_every": self.save_every,
            "checkpoint_policy": {
                "best": "checkpoint_best.pth",
                "latest": "checkpoint_latest.pth every 10 epochs",
                "final": "checkpoint_final.pth",
            },
        }
        output = Path(self.output_folder)
        with (output / "experiment_manifest.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(metadata, file, indent=2)
        if self.tb_writer is not None:
            self.tb_writer.add_text(
                "experiment/configuration",
                "\n".join(f"{key}: {value}" for key, value in metadata.items()),
                0,
            )
            self.tb_writer.flush()

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.local_rank != 0 or self.tb_writer is None:
            return
        epoch = self.current_epoch - 1
        log = self.logger.my_fantastic_logging
        epoch_seconds = (
            log["epoch_end_timestamps"][-1]
            - log["epoch_start_timestamps"][-1]
        )
        self.tb_writer.add_scalar(
            "dice/ema_mean_fg", float(log["ema_fg_dice"][-1]), epoch
        )
        self.tb_writer.add_scalar("time/epoch_seconds", epoch_seconds, epoch)
        self.tb_writer.flush()

    def on_train_end(self):
        super().on_train_end()
        if self.local_rank != 0:
            return
        output = Path(self.output_folder)
        final_checkpoint = output / "checkpoint_final.pth"
        latest_checkpoint = output / "checkpoint_latest.pth"
        if final_checkpoint.is_file():
            shutil.copyfile(final_checkpoint, latest_checkpoint)
