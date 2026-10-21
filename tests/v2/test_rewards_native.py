"""Unit tests for native Identity Reward Scorer and IdentityRewardPlugin."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from mmtrainer.config import ComponentSpec
from mmtrainer.registry import build_builtin_registries
from mmtrainer.rewards.catalog import IdentityRewardPlugin
from mmtrainer.rewards.identity import (
    ARCFACE_REFERENCE_LANDMARKS_112,
    IdentityScorer,
    align_face,
)


def test_identity_scorer_orthogonal_identical_opposite() -> None:
    scorer = IdentityScorer()

    unit_x = torch.tensor([1.0, 0.0, 0.0])
    unit_y = torch.tensor([0.0, 1.0, 0.0])
    unit_z = torch.tensor([0.0, 0.0, 1.0])

    # Orthogonal vectors -> 0.0
    score_ortho = scorer.score_embeddings(unit_x, unit_y)
    assert score_ortho.ndim == 0
    assert score_ortho.item() == pytest.approx(0.0, abs=1e-6)

    # Identical vectors -> 1.0
    score_ident = scorer.score_embeddings(unit_x, unit_x)
    assert score_ident.item() == pytest.approx(1.0, abs=1e-6)

    # Opposite vectors -> -1.0
    score_opp = scorer.score_embeddings(unit_x, -unit_x)
    assert score_opp.item() == pytest.approx(-1.0, abs=1e-6)

    # Normalization check with unnormalized vectors
    scaled_x = torch.tensor([3.5, 0.0, 0.0])
    neg_scaled_x = torch.tensor([-7.2, 0.0, 0.0])
    assert scorer.score_embeddings(scaled_x, neg_scaled_x).item() == pytest.approx(-1.0, abs=1e-6)
    assert scorer.score_embeddings(scaled_x, unit_z).item() == pytest.approx(0.0, abs=1e-6)

    # 45-degree angle -> cos(pi/4) = sqrt(2)/2
    vec_45 = torch.tensor([1.0, 1.0, 0.0])
    assert scorer.score_embeddings(unit_x, vec_45).item() == pytest.approx(math.sqrt(0.5), abs=1e-5)


def test_identity_scorer_batch_scoring() -> None:
    scorer = IdentityScorer()

    pred = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    target = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
    )

    batch_scores = scorer.score_embeddings(pred, target)
    assert batch_scores.shape == (4,)
    assert batch_scores.tolist() == pytest.approx([1.0, 1.0, -1.0, 0.0], abs=1e-6)

    # Broadcast single target vector across batch
    single_target = torch.tensor([1.0, 0.0, 0.0])
    broadcast_scores = scorer.score_embeddings(pred, single_target)
    assert broadcast_scores.shape == (4,)
    assert broadcast_scores.tolist() == pytest.approx([1.0, 0.0, 1.0, 0.0], abs=1e-6)


def test_identity_scorer_call_dict_and_tuple_inputs() -> None:
    scorer = IdentityScorer(default_score=-0.5)

    v1 = torch.tensor([1.0, 0.0])
    v2 = torch.tensor([0.0, 1.0])
    v3 = torch.tensor([-1.0, 0.0])

    # Dict inputs with pred_embedding and target_embedding
    dict_samples = [
        {"pred_embedding": v1, "target_embedding": v1},
        {"pred_embedding": v1, "target_embedding": v2},
        {"pred_embedding": v1, "target_embedding": v3},
    ]
    dict_res = scorer(dict_samples)
    assert isinstance(dict_res, list)
    assert all(isinstance(val, float) for val in dict_res)
    assert dict_res == pytest.approx([1.0, 0.0, -1.0], abs=1e-6)

    # Tuple inputs (pred_emb, target_emb)
    tuple_samples = [
        (v1, v1),
        (v1, v2),
        (v1, v3),
    ]
    tuple_res = scorer(tuple_samples)
    assert isinstance(tuple_res, list)
    assert all(isinstance(val, float) for val in tuple_res)
    assert tuple_res == pytest.approx([1.0, 0.0, -1.0], abs=1e-6)

    # Scorer initialized with default target_embedding
    scorer_with_target = IdentityScorer(target_embedding=v1)
    single_pred_dicts = [{"pred_embedding": v1}, {"pred_embedding": v3}]
    assert scorer_with_target(single_pred_dicts) == pytest.approx([1.0, -1.0], abs=1e-6)

    single_pred_tuples = [(v1,), (v3,)]
    assert scorer_with_target(single_pred_tuples) == pytest.approx([1.0, -1.0], abs=1e-6)

    # Empty samples returns empty list
    assert scorer([]) == []


def test_identity_scorer_non_finite_input_detection() -> None:
    scorer = IdentityScorer()

    nan_vec = torch.tensor([float("nan"), 1.0])
    inf_vec = torch.tensor([float("inf"), 0.0])
    neginf_vec = torch.tensor([float("-inf"), 0.0])
    valid_vec = torch.tensor([1.0, 0.0])

    with pytest.raises(ValueError, match="non-finite|NaN|Inf"):
        scorer.score_embeddings(nan_vec, valid_vec)

    with pytest.raises(ValueError, match="non-finite|NaN|Inf"):
        scorer.score_embeddings(valid_vec, nan_vec)

    with pytest.raises(ValueError, match="non-finite|NaN|Inf"):
        scorer.score_embeddings(inf_vec, valid_vec)

    with pytest.raises(ValueError, match="non-finite|NaN|Inf"):
        scorer.score_embeddings(valid_vec, neginf_vec)

    # Non-finite detection through __call__ with dict
    with pytest.raises(ValueError, match="non-finite|NaN|Inf"):
        scorer([{"pred_embedding": nan_vec, "target_embedding": valid_vec}])

    # Non-finite detection through __call__ with tuple
    with pytest.raises(ValueError, match="non-finite|NaN|Inf"):
        scorer([(valid_vec, inf_vec)])


def test_identity_reward_plugin_backward_compatibility() -> None:
    plugin = IdentityRewardPlugin(ComponentSpec(plugin_id="identity"))
    scorer = plugin.build_scorer(default_score=0.75)
    assert scorer(["sample"]) == [0.75]
    assert scorer(["sample1", "sample2"]) == [0.75, 0.75]

    # Verify registry resolution
    registries = build_builtin_registries()
    resolved_cls = registries.rewards.resolve("identity")
    assert resolved_cls is IdentityRewardPlugin
    reg_plugin = resolved_cls(ComponentSpec(plugin_id="identity"))
    reg_scorer = reg_plugin.build_scorer(default_score=0.42)
    assert reg_scorer(["arbitrary_object"]) == [0.42]


def test_align_face_integration_with_identity_scorer() -> None:
    landmarks = torch.tensor(ARCFACE_REFERENCE_LANDMARKS_112, dtype=torch.float32)

    # Single face without extractor (raw flattened comparison)
    face_img = torch.randn(1, 3, 112, 112)
    aligned_face = align_face(face_img, landmarks, out_size=112)
    target_flat = aligned_face.flatten(1)

    scorer_raw = IdentityScorer()
    score_raw = scorer_raw.score_aligned_faces(face_img, target_flat, landmarks=landmarks)
    assert score_raw.shape == (1,)
    assert score_raw.item() == pytest.approx(1.0, abs=1e-5)

    # Face scoring with mock feature extractor
    def mock_extractor(imgs: torch.Tensor) -> torch.Tensor:
        flat = imgs.flatten(1)[:, :64]
        return F.normalize(flat, p=2, dim=-1)

    extractor_scorer = IdentityScorer(feature_extractor=mock_extractor)
    target_feature = mock_extractor(aligned_face)

    feat_score = extractor_scorer.score_aligned_faces(face_img, target_feature, landmarks=landmarks)
    assert feat_score.shape == (1,)
    assert feat_score.item() == pytest.approx(1.0, abs=1e-5)

    # Batch of faces
    batch_faces = torch.randn(2, 3, 112, 112)
    aligned_batch_0 = align_face(batch_faces[0], landmarks, out_size=112)
    aligned_batch_1 = align_face(batch_faces[1], landmarks, out_size=112)
    aligned_batch = torch.stack([aligned_batch_0, aligned_batch_1], dim=0)
    batch_targets = mock_extractor(aligned_batch)

    batch_scores = extractor_scorer.score_aligned_faces(
        batch_faces, batch_targets, landmarks=landmarks
    )
    assert batch_scores.shape == (2,)
    assert batch_scores.tolist() == pytest.approx([1.0, 1.0], abs=1e-5)

    # Integration through __call__ with dict containing pred_faces and landmarks
    dict_call_samples = [
        {
            "pred_faces": face_img,
            "landmarks": landmarks,
            "target_embedding": target_feature,
        }
    ]
    call_scores = extractor_scorer(dict_call_samples)
    assert isinstance(call_scores, list)
    assert call_scores == pytest.approx([1.0], abs=1e-5)
