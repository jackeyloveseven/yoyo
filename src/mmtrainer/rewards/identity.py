"""Differentiable 5-point face alignment and ArcFace identity preservation loss."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

ARCFACE_REFERENCE_LANDMARKS_112: tuple[tuple[float, float], ...] = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)

__all__ = [
    "ARCFACE_REFERENCE_LANDMARKS_112",
    "IdentityScorer",
    "align_face",
    "cosine_identity_loss",
    "estimate_similarity_transform_5pt",
]


def estimate_similarity_transform_5pt(
    src_pts: torch.Tensor,
    dst_pts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Least-squares 2x3 similarity transform mapping 5 facial landmarks to ArcFace reference."""
    if src_pts.shape != (5, 2):
        raise ValueError(f"Expected src_pts shape (5, 2), got {tuple(src_pts.shape)}")
    src = src_pts.to(dtype=torch.float64)
    if dst_pts is None:
        dst = torch.tensor(ARCFACE_REFERENCE_LANDMARKS_112, device=src.device, dtype=torch.float64)
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
        [
            [(src_w - 1) / 2.0, 0.0, (src_w - 1) / 2.0],
            [0.0, (src_h - 1) / 2.0, (src_h - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        device=matrix_2x3.device,
        dtype=torch.float32,
    )
    dst_norm_to_pix = torch.tensor(
        [
            [(dst_size - 1) / 2.0, 0.0, (dst_size - 1) / 2.0],
            [0.0, (dst_size - 1) / 2.0, (dst_size - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
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
    aligned = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
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


def _normalize_tensor_shape(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 2 and t.shape[0] == 1:
        return t.squeeze(0)
    return t


def _is_embedding_tuple(first: Any) -> bool:
    if not isinstance(first, (tuple, list)) or len(first) != 2:
        return False
    elem = first[0]
    if isinstance(elem, torch.Tensor):
        return True
    if isinstance(elem, (list, tuple)) and len(elem) > 0 and isinstance(elem[0], (int, float)):
        return True
    return False


class IdentityScorer:
    """Composable identity reward scorer evaluating cosine similarity against target identity."""

    def __init__(
        self,
        target_embedding: torch.Tensor | None = None,
        feature_extractor: Any | None = None,
        name: str = "identity_scorer",
        default_score: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.target_embedding = target_embedding
        self.feature_extractor = feature_extractor
        self.name = name
        self.default_score = float(default_score)
        self.config = kwargs

    def score_embeddings(
        self,
        pred_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute cosine similarity between predicted and target embeddings."""
        if target_embeddings is None:
            if self.target_embedding is None:
                raise ValueError(
                    "target_embeddings must be provided in score_embeddings or at initialization"
                )
            target_embeddings = self.target_embedding

        if not isinstance(pred_embeddings, torch.Tensor):
            pred_embeddings = torch.as_tensor(pred_embeddings)
        if not isinstance(target_embeddings, torch.Tensor):
            target_embeddings = torch.as_tensor(target_embeddings)

        if not torch.isfinite(pred_embeddings).all():
            raise ValueError("pred_embeddings contains non-finite values (NaN or Inf)")
        if not torch.isfinite(target_embeddings).all():
            raise ValueError("target_embeddings contains non-finite values (NaN or Inf)")

        pred_norm = F.normalize(pred_embeddings.float(), p=2, dim=-1)
        target_norm = F.normalize(target_embeddings.float(), p=2, dim=-1)
        cos_sim = (pred_norm * target_norm).sum(dim=-1)
        return torch.clamp(cos_sim, -1.0, 1.0)

    def _warp_faces(
        self,
        faces: torch.Tensor,
        landmarks: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if landmarks is None:
            return faces
        if faces.ndim == 4 and faces.shape[0] > 1:
            if landmarks.ndim == 3 and landmarks.shape[0] == faces.shape[0]:
                return torch.stack(
                    [align_face(faces[i], landmarks[i], **kwargs) for i in range(faces.shape[0])],
                    dim=0,
                )
            return torch.stack(
                [align_face(faces[i], landmarks, **kwargs) for i in range(faces.shape[0])],
                dim=0,
            )
        return align_face(faces, landmarks, **kwargs)

    def score_aligned_faces(
        self,
        pred_faces: torch.Tensor,
        target_embedding: torch.Tensor | None = None,
        landmarks: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Warp faces if landmarks provided, extract embeddings, and return score_embeddings."""
        target_emb = target_embedding if target_embedding is not None else self.target_embedding
        if target_emb is None:
            raise ValueError(
                "target_embedding must be provided in score_aligned_faces or at initialization"
            )
        faces = self._warp_faces(pred_faces, landmarks=landmarks, **kwargs)
        if self.feature_extractor is not None:
            feat_input = faces.unsqueeze(0) if faces.ndim == 3 else faces
            pred_emb = self.feature_extractor(feat_input)
            if (
                faces.ndim == 3
                and isinstance(pred_emb, torch.Tensor)
                and pred_emb.ndim == 2
                and pred_emb.shape[0] == 1
            ):
                pred_emb = pred_emb.squeeze(0)
        else:
            pred_emb = faces.flatten(1) if faces.ndim >= 3 else faces
        return self.score_embeddings(pred_emb, target_emb)

    def _score_dict_embeddings(self, samples: Sequence[dict[str, Any]]) -> list[float]:
        preds: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for item in samples:
            if not isinstance(item, dict) or "pred_embedding" not in item:
                raise ValueError("Inconsistent sample format in batch")
            t = item.get("target_embedding", self.target_embedding)
            if t is None:
                raise ValueError("target_embedding must be provided in dict or at initialization")
            preds.append(_normalize_tensor_shape(torch.as_tensor(item["pred_embedding"])))
            targets.append(_normalize_tensor_shape(torch.as_tensor(t)))
        scores = self.score_embeddings(torch.stack(preds), torch.stack(targets))
        return [float(s.item()) for s in scores.view(-1)]

    def _score_dict_faces(self, samples: Sequence[dict[str, Any]]) -> list[float]:
        scores_list: list[float] = []
        for item in samples:
            if not isinstance(item, dict):
                raise ValueError("Inconsistent sample format in batch")
            face = item.get("pred_faces", item.get("image"))
            lm = item.get("landmarks")
            tgt = item.get("target_embedding", self.target_embedding)
            res = self.score_aligned_faces(torch.as_tensor(face), tgt, landmarks=lm)
            scores_list.extend([float(s.item()) for s in res.view(-1)])
        return scores_list

    def _score_dict_samples(self, samples: Sequence[dict[str, Any]]) -> list[float]:
        first = samples[0]
        if "pred_embedding" in first:
            return self._score_dict_embeddings(samples)
        if "pred_faces" in first or "image" in first:
            return self._score_dict_faces(samples)
        return [float(self.default_score) for _ in samples]

    def _score_face_pairs(
        self, preds: Sequence[torch.Tensor], targets: Sequence[torch.Tensor]
    ) -> list[float]:
        scores_list: list[float] = []
        for p, t in zip(preds, targets, strict=True):
            res = self.score_aligned_faces(p, t)
            scores_list.extend([float(s.item()) for s in res.view(-1)])
        return scores_list

    def _score_tuple_pair(self, samples: Sequence[tuple[Any, ...] | list[Any]]) -> list[float]:
        preds: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for item in samples:
            if not (isinstance(item, (tuple, list)) and len(item) >= 2):
                raise ValueError("Inconsistent sample format in batch")
            preds.append(torch.as_tensor(item[0]))
            targets.append(torch.as_tensor(item[1]))
        if preds[0].ndim >= 3:
            return self._score_face_pairs(preds, targets)
        preds_t = torch.stack([_normalize_tensor_shape(p) for p in preds])
        targets_t = torch.stack([_normalize_tensor_shape(t) for t in targets])
        scores = self.score_embeddings(preds_t, targets_t)
        return [float(s.item()) for s in scores.view(-1)]

    def _score_tuple_triplet(self, samples: Sequence[tuple[Any, ...] | list[Any]]) -> list[float]:
        scores_list: list[float] = []
        for item in samples:
            p = torch.as_tensor(item[0])
            lm = torch.as_tensor(item[1]) if item[1] is not None else None
            tgt = torch.as_tensor(item[2]) if item[2] is not None else self.target_embedding
            res = self.score_aligned_faces(p, tgt, landmarks=lm)
            scores_list.extend([float(s.item()) for s in res.view(-1)])
        return scores_list

    def _score_tuple_single(self, samples: Sequence[tuple[Any, ...] | list[Any]]) -> list[float]:
        preds = [_normalize_tensor_shape(torch.as_tensor(item[0])) for item in samples]
        scores = self.score_embeddings(torch.stack(preds), self.target_embedding)
        return [float(s.item()) for s in scores.view(-1)]

    def _score_tuple_samples(self, samples: Sequence[tuple[Any, ...] | list[Any]]) -> list[float]:
        first = samples[0]
        if _is_embedding_tuple(first):
            return self._score_tuple_pair(samples)
        if len(first) == 3 and isinstance(first[0], torch.Tensor):
            return self._score_tuple_triplet(samples)
        if (
            len(first) == 1
            and isinstance(first[0], torch.Tensor)
            and self.target_embedding is not None
        ):
            return self._score_tuple_single(samples)
        return [float(self.default_score) for _ in samples]

    def __call__(self, samples: Sequence[Any]) -> list[float]:
        """Score samples, returning a list of float rewards."""
        if not samples:
            return []
        first = samples[0]
        if isinstance(first, dict):
            return self._score_dict_samples(samples)
        if isinstance(first, (tuple, list)):
            return self._score_tuple_samples(samples)
        if isinstance(first, torch.Tensor) and self.target_embedding is not None:
            if first.ndim >= 3:
                res = [self.score_aligned_faces(s, self.target_embedding) for s in samples]
                return [float(s.item()) for r in res for s in r.view(-1)]
            preds_t = torch.stack(list(samples))
            scores = self.score_embeddings(preds_t, self.target_embedding)
            return [float(s.item()) for s in scores.view(-1)]
        return [float(self.default_score) for _ in samples]
