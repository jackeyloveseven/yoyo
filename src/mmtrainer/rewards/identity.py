"""Differentiable 5-point face alignment and ArcFace identity preservation loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

ARCFACE_REFERENCE_LANDMARKS_112: tuple[tuple[float, float], ...] = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)


def estimate_similarity_transform_5pt(
    src_pts: torch.Tensor,
    dst_pts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Least-squares 2x3 similarity transform mapping 5 facial landmarks to ArcFace reference."""
    if src_pts.shape != (5, 2):
        raise ValueError(f"Expected src_pts shape (5, 2), got {tuple(src_pts.shape)}")
    src = src_pts.to(dtype=torch.float64)
    if dst_pts is None:
        dst = torch.tensor(
            ARCFACE_REFERENCE_LANDMARKS_112, device=src.device, dtype=torch.float64
        )
    else:
        dst = dst_pts.to(device=src.device, dtype=torch.float64)
    src_mean = src.mean(dim=0, keepdim=True)
    dst_mean = dst.mean(dim=0, keepdim=True)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    src_var = (src_centered.pow(2).sum(dim=1)).mean()
    cov = (dst_centered.t() @ src_centered) / 5.0
    u, s, vh = torch.linalg.svd(cov)
    d = torch.ones(2, device=src.device, dtype=torch.float64)
    if torch.linalg.det(u) * torch.linalg.det(vh) < 0:
        d[-1] = -1.0
    rot = u @ torch.diag(d) @ vh
    scale = (s * d).sum() / src_var.clamp_min(1e-12)
    trans = dst_mean.squeeze(0) - scale * (rot @ src_mean.squeeze(0))
    transform = torch.zeros((2, 3), device=src.device, dtype=torch.float32)
    transform[:, :2] = (scale * rot).to(dtype=torch.float32)
    transform[:, 2] = trans.to(dtype=torch.float32)
    return transform


def _pixel_affine_to_normalized_theta(
    matrix_2x3: torch.Tensor,
    *,
    src_h: int,
    src_w: int,
    dst_size: int = 112,
) -> torch.Tensor:
    m33 = torch.eye(3, device=matrix_2x3.device, dtype=torch.float32)
    m33[:2, :] = matrix_2x3.to(dtype=torch.float32)
    inv_m = torch.linalg.inv(m33)
    src_norm_to_pix = torch.tensor(
        [[(src_w - 1) / 2.0, 0.0, (src_w - 1) / 2.0],
         [0.0, (src_h - 1) / 2.0, (src_h - 1) / 2.0],
         [0.0, 0.0, 1.0]],
        device=matrix_2x3.device,
        dtype=torch.float32,
    )
    dst_norm_to_pix = torch.tensor(
        [[(dst_size - 1) / 2.0, 0.0, (dst_size - 1) / 2.0],
         [0.0, (dst_size - 1) / 2.0, (dst_size - 1) / 2.0],
         [0.0, 0.0, 1.0]],
        device=matrix_2x3.device,
        dtype=torch.float32,
    )
    theta_3x3 = torch.linalg.inv(src_norm_to_pix) @ inv_m @ dst_norm_to_pix
    return theta_3x3[:2, :]


def align_face(
    image: torch.Tensor,
    landmarks_5pt: torch.Tensor,
    *,
    out_size: int = 112,
) -> torch.Tensor:
    """Differentiably warp a ``[1, C, H, W]`` or ``[C, H, W]`` tensor to ``out_size x out_size``."""
    squeezed = False
    if image.ndim == 3:
        image = image.unsqueeze(0)
        squeezed = True
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected image shape [1, C, H, W], got {tuple(image.shape)}")
    _, _, src_h, src_w = image.shape
    sim_2x3 = estimate_similarity_transform_5pt(landmarks_5pt)
    theta = _pixel_affine_to_normalized_theta(
        sim_2x3, src_h=src_h, src_w=src_w, dst_size=out_size
    ).unsqueeze(0)
    grid = F.affine_grid(
        theta.to(dtype=image.dtype),
        size=[1, image.shape[1], out_size, out_size],
        align_corners=True,
    )
    aligned = F.grid_sample(
        image, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return aligned.squeeze(0) if squeezed else aligned


def cosine_identity_loss(
    pred_embedding: torch.Tensor,
    target_embedding: torch.Tensor,
) -> torch.Tensor:
    """Compute ``1 - cos(pred, target)`` for identity-preserving diffusion alignment."""
    pred_norm = F.normalize(pred_embedding.float(), p=2, dim=-1)
    target_norm = F.normalize(target_embedding.detach().float(), p=2, dim=-1)
    cos_sim = (pred_norm * target_norm).sum(dim=-1)
    return (1.0 - cos_sim).mean()
