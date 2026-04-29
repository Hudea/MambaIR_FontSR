"""
fontsr_mambair_utils — FontSR data adapters and MambaIRv2Light model helpers.

Uses FontSR's build_dataset(data_cfg, split) public API — does NOT reach into
private attributes. Produces MambaIR-compatible [0,1] grayscale tensors:
LR at native 32×32, HR at 128×128.
"""

from __future__ import annotations

import csv
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

# ----------------------------------------------------------------------
# FontSR root resolution
# ----------------------------------------------------------------------


def _resolve_fontsr_root() -> Path:
    repo_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []
    if os.environ.get("FONTSR_ROOT"):
        candidates.append(Path(os.environ["FONTSR_ROOT"]))
    candidates.extend(
        [
            repo_dir.parent / "FontSR",
            repo_dir.parent.parent / "FontSR",
            Path("/Users/butterflies/Project/FontSR"),
        ]
    )
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "datasets" / "dataset.py").exists():
            return root
    searched = ", ".join(str(p) for p in candidates)
    raise RuntimeError(f"Cannot locate FontSR root. Set FONTSR_ROOT. Searched: {searched}")


FONTSR_ROOT = _resolve_fontsr_root()
if str(FONTSR_ROOT) not in sys.path:
    sys.path.insert(0, str(FONTSR_ROOT))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_fontsr_path(path: str | Path) -> str:
    """Resolve a path that may be relative to FONTSR_ROOT."""
    candidate = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if candidate.is_absolute():
        if candidate.exists():
            return str(candidate)
        parts = candidate.parts
        if "FontSR" in parts:
            rel = Path(*parts[parts.index("FontSR") + 1 :])
            relocated = FONTSR_ROOT / rel
            if relocated.exists():
                return str(relocated)
        return str(candidate)
    return str(FONTSR_ROOT / candidate)


def apply_difficulty_profile(config: dict[str, Any], profile: str) -> None:
    config["data"]["level"] = profile


def resolve_device(requested: str | None) -> torch.device:
    if not requested or requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested cuda, but CUDA is not available.")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("Requested mps, but MPS is not available.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unknown device: {requested}")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def denormalize(t: torch.Tensor) -> torch.Tensor:
    """FontSR normalizes to [-1,1] (mean=0.5, std=0.5); convert back to [0,1]."""
    return (t + 1.0) / 2.0


def save_grayscale_tensor(path: str | Path, tensor: torch.Tensor) -> None:
    """Save a [1, H, W] or [H, W] tensor as grayscale PNG."""
    import cv2

    if tensor.dim() == 3:
        tensor = tensor[0]
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    cv2.imwrite(str(path), arr)


def build_export_filename(index: int, semantic_hex: str, variant_hex: str) -> str:
    return f"{index:06d}_sem-{semantic_hex}_var-{variant_hex}.png"


def write_manifest(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "image_path",
        "gt_hex_code",
        "gt_semantic_hex_code",
        "gt_base_char",
        "split",
        "sample_index",
        "checkpoint",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------
# FontSRMambaIRAdapter — wraps build_dataset; denormalizes [-1,1] → [0,1]
# ----------------------------------------------------------------------


class FontSRMambaIRAdapter(Dataset):
    """
    Wraps FontSR's build_dataset() public API.

    Returns MambaIR-compatible dict:
        lq : [1, 32, 32], float32, range [0, 1]   (LR, native 32×32)
        gt : [1, 128, 128], float32, range [0, 1]  (HR ground-truth 128×128)
        semantic_hex_code, hex_code, base_char, split, sample_index

    FontSR's OnlineRenderDataset.__getitem__ returns:
        hr_image : tensor in [-1, 1]
        lr_image : tensor in [-1, 1]
    We denormalize to [0, 1] for MambaIR (img_range=1.0, mean-based norm).
    """

    def __init__(self, config: dict[str, Any], split: str):
        super().__init__()
        self._config = config
        self._split = split
        self._is_train = split in ("train", "train_split")

        # Resolve font_path and char_map_path to absolute paths before
        # build_dataset / OnlineRenderDataset sees them (they don't do path resolution).
        data_cfg = config["data"]
        spec = data_cfg.get("spec", {})
        if "font" in spec:
            spec["font"] = dict(spec["font"])
            spec["font"]["font_path"] = resolve_fontsr_path(spec["font"]["font_path"])
        if "charset" in spec:
            spec["charset"] = dict(spec["charset"])
            spec["charset"]["char_map_path"] = resolve_fontsr_path(spec["charset"]["char_map_path"])

        from datasets import build_dataset
        from torchvision import transforms

        self._dataset = build_dataset(data_cfg, split)

        # Override _lr_transform to skip the bicubic-to-HR resize that
        # OnlineRenderDataset bakes in for diffusion-style models.
        self._dataset._lr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = self._dataset[index]

        hr_tensor = denormalize(raw["hr_image"])   # [1, 128, 128] in [0,1]
        lr_tensor = denormalize(raw["lr_image"])   # [1, 32, 32] in [0,1] (native LR)

        if lr_tensor.dim() == 2:
            lr_tensor = lr_tensor.unsqueeze(0)
        if hr_tensor.dim() == 2:
            hr_tensor = hr_tensor.unsqueeze(0)

        return {
            "lq": lr_tensor,
            "gt": hr_tensor,
            "semantic_hex_code": raw["semantic_hex_code"],
            "hex_code": raw["hex_code"],
            "base_char": raw["base_char"],
            "split": self._split,
            "sample_index": index,
        }


# ----------------------------------------------------------------------
# MambaIR model builder (lazy import to avoid mamba_ssm/basicsr complexity at --check-config)
# ----------------------------------------------------------------------


def build_mambairv2light_model(model_cfg: dict[str, Any]) -> torch.nn.Module:
    """
    Build MambaIRv2Light model from config dict.
    Lazy import keeps --check-config fast (no mamba_ssm / CUDA needed).
    """
    mambair_root = Path(__file__).resolve().parent
    if str(mambair_root) not in sys.path:
        sys.path.insert(0, str(mambair_root))
    
    from basicsr.archs.mambairv2light_arch import MambaIRv2Light
    
    model = MambaIRv2Light(
        img_size=model_cfg.get("img_size", 32),
        in_chans=model_cfg.get("in_chans", 1),
        embed_dim=model_cfg.get("embed_dim", 48),
        d_state=model_cfg.get("d_state", 8),
        depths=model_cfg.get("depths", [4, 4, 4, 4]),
        num_heads=model_cfg.get("num_heads", [4, 4, 4, 4]),
        window_size=model_cfg.get("window_size", 8),
        inner_rank=model_cfg.get("inner_rank", 32),
        num_tokens=model_cfg.get("num_tokens", 64),
        convffn_kernel_size=model_cfg.get("convffn_kernel_size", 5),
        mlp_ratio=model_cfg.get("mlp_ratio", 1.0),
        upscale=model_cfg.get("upscale", 4),
        upsampler=model_cfg.get("upsampler", "pixelshuffledirect"),
        img_range=model_cfg.get("img_range", 1.0),
        resi_connection=model_cfg.get("resi_connection", "1conv"),
    )
    return model
