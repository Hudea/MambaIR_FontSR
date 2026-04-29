"""
export_fontsr_predictions_mambair.py — Export MambaIRv2Light SR predictions + optional retrieval eval.

Mirrors HAT/SwinIR export pattern but for MambaIRv2Light.

Usage:
    python export_fontsr_predictions_mambair.py \
        --config configs/fontsr_mambair_lr32_level1.yaml \
        --checkpoint experiments/mambair_fontsr_level1/checkpoints/best.pt \
        --split validation \
        --batch-size 1 \
        --max-samples 2 \
        --device cpu \
        --output-dir experiments/mambair_fontsr_level1/export_best \
        --save-lr-hr \
        --run-retrieval-eval \
        --retrieval-device cpu \
        --retrieval-metric hybrid \
        --top-k 3
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from fontsr_mambair_utils import (
    FONTSR_ROOT,
    FontSRMambaIRAdapter,
    apply_difficulty_profile,
    build_export_filename,
    build_mambairv2light_model,
    load_config,
    resolve_device,
    resolve_fontsr_path,
    save_grayscale_tensor,
    write_manifest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--difficulty-profile", type=str, default=None)
    parser.add_argument(
        "--split", type=str, default="validation", choices=["validation", "val", "train"]
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-lr-hr", action="store_true")
    parser.add_argument("--run-retrieval-eval", action="store_true")
    parser.add_argument("--retrieval-device", type=str, default="cpu")
    parser.add_argument("--retrieval-metric", type=str, default="hybrid")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.difficulty_profile:
        apply_difficulty_profile(config, args.difficulty_profile)

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)

    # Normalize split name
    split = "val" if args.split in ("val", "validation") else "train"

    # Build dataset
    ds = FontSRMambaIRAdapter(config, split=split)
    logger.info("Export split '%s': %d samples", split, len(ds))

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Load MambaIR model from checkpoint
    model_cfg = config["model"]
    model = build_mambairv2light_model(model_cfg)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    # Create output directories
    sr_dir = output_dir / "sr"
    sr_dir.mkdir(parents=True, exist_ok=True)
    if args.save_lr_hr:
        lr_dir = output_dir / "lr"
        hr_dir = output_dir / "hr"
        lr_dir.mkdir(parents=True, exist_ok=True)
        hr_dir.mkdir(parents=True, exist_ok=True)

    # Export loop
    manifest_rows = []
    global_sample_idx = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Exporting")
        for batch in pbar:
            lq_batch = batch["lq"].to(device)
            gt_batch = batch["gt"]
            sem_hex_list = batch["semantic_hex_code"]
            hex_list = batch["hex_code"]
            base_list = batch["base_char"]
            split_list = batch["split"]
            sample_idx_list = batch["sample_index"]

            sr_batch = model(lq_batch)

            for i in range(sr_batch.size(0)):
                sem_hex = sem_hex_list[i] if isinstance(sem_hex_list[i], str) else sem_hex_list[i].item()
                var_hex = hex_list[i] if isinstance(hex_list[i], str) else hex_list[i].item()
                base_char = base_list[i] if isinstance(base_list[i], str) else base_list[i].item()
                split_val = split_list[i] if isinstance(split_list[i], str) else split_list[i].item()
                sample_idx = int(sample_idx_list[i].item()) if isinstance(sample_idx_list[i], torch.Tensor) else int(sample_idx_list[i])

                filename = build_export_filename(global_sample_idx, str(sem_hex), str(var_hex))

                # Save SR
                save_grayscale_tensor(sr_dir / filename, sr_batch[i].cpu())

                # Save LR/HR if requested
                if args.save_lr_hr:
                    save_grayscale_tensor(lr_dir / filename, lq_batch[i].cpu())
                    save_grayscale_tensor(hr_dir / filename, gt_batch[i].cpu())

                manifest_rows.append({
                    "image_path": f"sr/{filename}",
                    "gt_hex_code": str(var_hex),
                    "gt_semantic_hex_code": str(sem_hex),
                    "gt_base_char": str(base_char),
                    "split": str(split_val),
                    "sample_index": sample_idx,
                    "checkpoint": args.checkpoint,
                })

                global_sample_idx += 1

                if args.max_samples and global_sample_idx >= args.max_samples:
                    break

            if args.max_samples and global_sample_idx >= args.max_samples:
                break

    # Write manifest
    manifest_path = output_dir / "predictions_manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    logger.info("Wrote manifest: %s (%d rows)", manifest_path, len(manifest_rows))

    # Write export summary
    summary = {
        "split": split,
        "num_samples": global_sample_idx,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "tensor_range": "[0,1]",
        "adapter_note": (
            "FontSR build_dataset public API; tensors denormalized from [-1,1] to [0,1] "
            "for MambaIR img_range=1.0. LR=32×32, HR=128×128."
        ),
    }
    with open(output_dir / "export_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save used config
    with open(output_dir / "used_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)

    logger.info("Export complete: %d samples", global_sample_idx)

    # Run retrieval eval if requested
    if args.run_retrieval_eval:
        eval_dir = output_dir / "retrieval_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        retrieval_cmd = [
            sys.executable,
            str(FONTSR_ROOT / "scripts/eval_variant_retrieval.py"),
            "--manifest", str(manifest_path),
            "--label-mode", "manifest",
            "--font-path", resolve_fontsr_path(config["data"]["spec"]["font"]["font_path"]),
            "--char-map-path", resolve_fontsr_path(config["data"]["spec"]["charset"]["char_map_path"]),
            "--target-size", "128",
            "--gallery-font-size", "128",
            "--gallery-hinting", "mac",
            "--gallery-subpixel", "0.0",
            "--metric", args.retrieval_metric,
            "--top-k", str(args.top_k),
            "--device", args.retrieval_device,
            "--output-dir", str(eval_dir),
            "--pad-mode", "center",
            "--batch-size", str(args.batch_size),
        ]
        
        logger.info("Running retrieval eval: %s", " ".join(retrieval_cmd))
        subprocess.run(retrieval_cmd, check=True)
        logger.info("Retrieval eval complete.")


if __name__ == "__main__":
    main()
