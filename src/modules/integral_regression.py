from __future__ import annotations

from typing import Dict, Optional, Tuple

import segmentation_models_pytorch as smp
import torch
from torch import nn


class IntegralRegression(nn.Module):
    """Integral regression layer for heatmap-based keypoints.

    This layer converts a stack of keypoint heatmaps into coordinates by
    normalizing the spatial logits with softmax and computing the expected
    value over the x/y grids. Optionally, coordinates can be normalized to the
    ``[0, 1]`` range.
    """

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize

    def forward(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Convert heatmaps to ``(B, K, 2)`` keypoint coordinates.

        Args:
            heatmaps: Tensor of shape ``(B, K, H, W)`` where ``K`` is the number
                of keypoints.

        Returns:
            Tensor of shape ``(B, K, 2)`` with ``(x, y)`` coordinates for each
            keypoint. Coordinates are normalized to ``[0, 1]`` if ``normalize``
            is set.
        """
        if heatmaps.ndim != 4:
            raise ValueError(
                f"Expected heatmaps of shape (B, K, H, W), got {heatmaps.shape}"
            )

        batch_size, num_kpts, height, width = heatmaps.shape
        # Softmax over spatial dimensions to obtain probability maps.
        probs = torch.softmax(heatmaps.flatten(2), dim=-1).view(
            batch_size, num_kpts, height, width
        )

        x_grid = torch.arange(width, device=heatmaps.device, dtype=heatmaps.dtype)
        y_grid = torch.arange(height, device=heatmaps.device, dtype=heatmaps.dtype)
        x_grid = x_grid.view(1, 1, 1, width)
        y_grid = y_grid.view(1, 1, height, 1)

        x_coord = (probs * x_grid).sum(dim=(2, 3))
        y_coord = (probs * y_grid).sum(dim=(2, 3))
        coords = torch.stack((x_coord, y_coord), dim=-1)

        if self.normalize:
            # Divide by the maximal pixel index to keep coordinates in ``[0, 1]``.
            scale = torch.tensor(
                [max(width - 1, 1), max(height - 1, 1)],
                device=heatmaps.device,
                dtype=heatmaps.dtype,
            )
            coords = coords / scale

        return coords


class UNetIntegralRegression(nn.Module):
    """U-Net backbone followed by integral regression for keypoints.

    The model predicts a keypoint heatmap using a configurable U-Net and then
    converts those heatmaps into coordinates via :class:`IntegralRegression`.
    """

    def __init__(
        self,
        num_keypoints: int,
        unet_kwargs: Optional[Dict] = None,
        normalize_coords: bool = True,
    ) -> None:
        super().__init__()
        unet_kwargs = unet_kwargs or {}

        # Ensure the decoder outputs one channel per keypoint heatmap.
        unet_args = {**unet_kwargs, "classes": num_keypoints}
        self.backbone = smp.Unet(**unet_args)
        self.integral = IntegralRegression(normalize=normalize_coords)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return heatmaps and corresponding keypoint coordinates.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            A tuple ``(heatmaps, coords)`` where ``heatmaps`` has shape
            ``(B, K, H, W)`` and ``coords`` has shape ``(B, K, 2)``.
        """
        heatmaps = self.backbone(x)
        coords = self.integral(heatmaps)
        return heatmaps, coords
