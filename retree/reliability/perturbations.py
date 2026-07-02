from __future__ import annotations

import torch


class ModalityPerturbation:
    """Controlled perturbations used for reliability analysis and stability regularization."""

    def __init__(self, text_mask_ratio: float = 0.15, image_mask_ratio: float = 0.20):
        self.text_mask_ratio = text_mask_ratio
        self.image_mask_ratio = image_mask_ratio

    def mask_representation(self, x: torch.Tensor, ratio: float) -> torch.Tensor:
        if ratio <= 0:
            return x
        mask = torch.rand_like(x) > ratio
        return x * mask.to(x.dtype)

    def noisy_text(self, text_rep: torch.Tensor) -> torch.Tensor:
        return self.mask_representation(text_rep, self.text_mask_ratio)

    def noisy_visual(self, visual_rep: torch.Tensor) -> torch.Tensor:
        return self.mask_representation(visual_rep, self.image_mask_ratio)

    def noisy_fused(self, fused_rep: torch.Tensor, ratio: float = 0.10) -> torch.Tensor:
        noise = torch.randn_like(fused_rep) * ratio
        return fused_rep + noise
