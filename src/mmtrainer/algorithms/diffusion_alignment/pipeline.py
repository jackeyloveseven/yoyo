"""State-only plan for sequential DPO then DiffusionNFT post-training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AlignmentStage(str, Enum):
    """Valid stages in the FireRed-style sequential alignment plan."""

    DPO = "dpo"
    DIFFUSION_NFT = "diffusion_nft"
    COMPLETE = "complete"


@dataclass
class DPOThenNFTPlan:
    """Track checkpoint handoff without combining DPO and NFT losses."""

    stage: AlignmentStage = AlignmentStage.DPO
    dpo_checkpoint: str | None = None
    nft_policy_checkpoint: str | None = None
    nft_old_checkpoint: str | None = None
    nft_reference_checkpoint: str | None = None
    final_checkpoint: str | None = None

    def complete_dpo(self, checkpoint: str) -> None:
        """Enter NFT and initialize policy, old, and reference from DPO."""

        if self.stage is not AlignmentStage.DPO:
            raise RuntimeError("DPO can only be completed from the DPO stage")
        checkpoint = checkpoint.strip()
        if not checkpoint:
            raise ValueError("DPO checkpoint must not be empty")
        self.dpo_checkpoint = checkpoint
        self.nft_policy_checkpoint = checkpoint
        self.nft_old_checkpoint = checkpoint
        self.nft_reference_checkpoint = checkpoint
        self.stage = AlignmentStage.DIFFUSION_NFT

    def complete_nft(self, checkpoint: str) -> None:
        """Mark the sequential plan complete with the final NFT checkpoint."""

        if self.stage is not AlignmentStage.DIFFUSION_NFT:
            raise RuntimeError("DiffusionNFT can only be completed after DPO")
        checkpoint = checkpoint.strip()
        if not checkpoint:
            raise ValueError("DiffusionNFT checkpoint must not be empty")
        self.final_checkpoint = checkpoint
        self.stage = AlignmentStage.COMPLETE
