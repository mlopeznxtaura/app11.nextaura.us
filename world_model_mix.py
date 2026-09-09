"""MCF v2 world-model V-JEPA stream mix — single source of truth."""

from __future__ import annotations

# Original HLS latent stack (CS2 + ego + sokoban).
VJEPA_MIX_HLS: list[str] = [
    "cbctr/cs2-10k-vjepa2-latents-300",
    "rookierufus/ego10k-vjepa-latents",
    "qinglinhou/sokoban-10k-vjepa2-tokenized",
]

# Full world-model mix for MCF v2 5 GB runs (HLS + robotics + physics + gameplay).
VJEPA_MIX_WORLD: list[str] = [
    *VJEPA_MIX_HLS,
    # Robotics & JEPA-native
    # Gated on HF Hub — request access separately; excluded from live mix until approved.
    # "facebook/jepa-wms",
    "jialei02/libero_merged_no_noops_20hz",
    "quastAI/behavior-1k-2025-challenge-vjepa2-vitg-demo-embeddings",
    # Synthetic physics / GCT-friendly
    "Silicon23/mujoco-kinematics-probing",
    "Swastikr/PhysSim-VLM-Dataset",
    "Swastikr/PhysSim-VLM-SFT-R2-Data",
    # Embodied failure + human gameplay
    "agibot-world/AgiBotWorld2026",
    "HuberyLL/nms_hitl_world_model",
]

# Default for start/resume scripts (override via MCF_VJEPA_MIX env comma-list).
DEFAULT_VJEPA_MIX: list[str] = VJEPA_MIX_WORLD
