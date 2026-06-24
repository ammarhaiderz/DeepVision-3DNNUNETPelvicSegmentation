from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_no_rotation import nnUNetTrainer_no_rotation
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainer_student_teacher(nnUNetTrainer_no_rotation):
    """Student trainer with soft teacher-prior regularization.

    The student dataset has four input channels:
    0. CT
    1. LH teacher probability prior
    2. RH teacher probability prior
    3. sacrum teacher probability prior

    The network input receives channels 1-3 as zeros, so the deployed model does
    not depend on teacher predictions. Channels 1-3 are used only as a soft
    distillation target during training.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.teacher_distill_weight = float(os.environ.get("STUDENT_TEACHER_DISTILL_WEIGHT", "0.2"))
        self.teacher_distill_temperature = float(os.environ.get("STUDENT_TEACHER_TEMPERATURE", "1.0"))

    @staticmethod
    def _renormalize_prior_patch(priors: torch.Tensor) -> torch.Tensor:
        """Bring preprocessed prior channels back into a stable 0..1 range per patch.

        nnU-Net preprocesses image channels. Teacher priors are stored as image
        channels so they go through the same spatial crop/augmentation as CT.
        This min-max step keeps the auxiliary target bounded for the MSE loss.
        """
        flat = priors.flatten(2)
        mins = flat.min(dim=2).values[:, :, None, None, None]
        maxs = flat.max(dim=2).values[:, :, None, None, None]
        denom = torch.clamp(maxs - mins, min=1e-6)
        normalized = (priors - mins) / denom
        constant_patch = (maxs - mins) < 1e-6
        return torch.where(constant_patch, torch.zeros_like(normalized), normalized)

    def _student_input_and_teacher_priors(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if data.shape[1] < 4:
            return data, None

        teacher_priors = self._renormalize_prior_patch(data[:, 1:4])
        student_input = data.clone()
        student_input[:, 1:4] = 0
        return student_input, teacher_priors

    def _distillation_loss(self, output, teacher_priors: torch.Tensor | None) -> torch.Tensor:
        if teacher_priors is None or self.teacher_distill_weight <= 0:
            if isinstance(output, (list, tuple)):
                return output[0].sum() * 0
            return output.sum() * 0

        logits = output[0] if isinstance(output, (list, tuple)) else output
        if logits.shape[1] < 4:
            raise RuntimeError(
                "Student-teacher distillation expects background + 3 anatomy output channels."
            )

        probs = torch.softmax(logits / self.teacher_distill_temperature, dim=1)[:, 1:4]
        if teacher_priors.shape[2:] != probs.shape[2:]:
            teacher_priors = F.interpolate(
                teacher_priors,
                size=probs.shape[2:],
                mode="trilinear" if probs.ndim == 5 else "bilinear",
                align_corners=False,
            )
        return F.mse_loss(probs, teacher_priors)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        data, teacher_priors = self._student_input_and_teacher_priors(data)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            segmentation_loss = self.loss(output, target)
            distillation_loss = self._distillation_loss(output, teacher_priors)
            total_loss = segmentation_loss + self.teacher_distill_weight * distillation_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {
            "loss": total_loss.detach().cpu().numpy(),
            "segmentation_loss": segmentation_loss.detach().cpu().numpy(),
            "distillation_loss": distillation_loss.detach().cpu().numpy(),
        }

