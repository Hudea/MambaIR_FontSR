"""
train_fontsr_mambair.py — MambaIRv2Light training entry point for FontSR LR32→HR128 comparison.

Mirrors the HAT/SwinIR training scripts but uses MambaIRv2Light architecture
and FontSR's build_dataset public API.

Usage:
    # Config check (no model built)
    python train_fontsr_mambair.py --config configs/fontsr_mambair_lr32_level1.yaml --check-config

    # L1 training (full)
    python train_fontsr_mambair.py --config configs/fontsr_mambair_lr32_level1.yaml \
        --difficulty-profile level1 \
        --output-dir experiments/mambair_fontsr_level1 \
        --device cuda --epochs 200 --batch-size 16
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from fontsr_mambair_utils import (
    FontSRMambaIRAdapter,
    apply_difficulty_profile,
    build_mambairv2light_model,
    load_config,
    resolve_device,
    set_seed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def maybe_wrap_data_parallel(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        logger.info("Using DataParallel across %d CUDA devices", torch.cuda.device_count())
        return torch.nn.DataParallel(model)
    return model


def psnr_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Simple PSNR between two [0,1] tensors."""
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--difficulty-profile", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save milestone checkpoints every N epochs; best.pt and last.pt are always updated.",
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.difficulty_profile:
        apply_difficulty_profile(config, args.difficulty_profile)

    device = resolve_device(args.device)

    # ------------------------------------------------------------------
    # --check-config: validate datasets only (no model)
    # ------------------------------------------------------------------
    if args.check_config:
        logger.info("Running --check-config (dataset validation only)")
        train_ds = FontSRMambaIRAdapter(config, split="train")
        val_ds = FontSRMambaIRAdapter(config, split="val")
        logger.info("  train samples: %d", len(train_ds))
        logger.info("  val samples: %d", len(val_ds))

        sample = train_ds[0]
        lq = sample["lq"]
        gt = sample["gt"]
        logger.info("  train[0] lq shape: %s, gt shape: %s", lq.shape, gt.shape)
        assert lq.shape == (1, 32, 32), f"lq shape mismatch: {lq.shape}"
        assert gt.shape == (1, 128, 128), f"gt shape mismatch: {gt.shape}"

        val_sample = val_ds[0]
        lq_val = val_sample["lq"]
        gt_val = val_sample["gt"]
        assert lq_val.shape == (1, 32, 32), f"val lq shape mismatch: {lq_val.shape}"
        assert gt_val.shape == (1, 128, 128), f"val gt shape mismatch: {gt_val.shape}"

        logger.info("CHECK_CONFIG_OK")
        print("CHECK_CONFIG_OK")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Normal training
    # ------------------------------------------------------------------
    if not args.output_dir:
        parser.error("--output-dir is required for training (omit for --check-config)")

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.get("training", {}).get("seed", 42))

    train_ds = FontSRMambaIRAdapter(config, split="train")
    val_ds = FontSRMambaIRAdapter(config, split="val")
    logger.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

    pin_memory = device.type == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    # Build MambaIR model
    model_cfg = config["model"]
    model = build_mambairv2light_model(model_cfg)
    model = model.to(device)

    start_epoch = 1
    resume_checkpoint = None
    if args.resume_from:
        resume_checkpoint = torch.load(
            args.resume_from, map_location=device, weights_only=False
        )
        model.load_state_dict(resume_checkpoint["model"])
        start_epoch = resume_checkpoint["epoch"] + 1
        logger.info("Loaded model weights from epoch %d", resume_checkpoint["epoch"])

    model = maybe_wrap_data_parallel(model, device)

    # Optimizer: Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("training", {}).get("lr", 2e-4))

    # Scheduler: MultiStepLR
    train_cfg = config.get("training", {})
    milestones = train_cfg.get("lr_milestones", [100, 150, 175])
    gamma = train_cfg.get("gamma", 0.5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma
    )

    criterion = torch.nn.L1Loss()

    if resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        logger.info("Resumed optimizer and scheduler from epoch %d", resume_checkpoint["epoch"])

    # Save config snapshot
    used_config_path = output_dir / "used_config.yaml"
    with open(used_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)

    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        # ---------- Training ----------
        model.train()
        train_loss = 0.0
        train_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} train")
        for batch in pbar:
            lq_batch = batch["lq"].to(device)
            gt_batch = batch["gt"].to(device)

            optimizer.zero_grad()
            sr_batch = model(lq_batch)
            loss = criterion(sr_batch, gt_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({"l1": f"{loss.item():.4f}"})

            if args.max_train_batches and train_batches >= args.max_train_batches:
                break

        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
        scheduler.step()

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_batches = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch} val")
            for batch in pbar_val:
                lq_batch = batch["lq"].to(device)
                gt_batch = batch["gt"].to(device)
                sr_batch = model(lq_batch)
                loss = torch.nn.functional.l1_loss(sr_batch, gt_batch)
                val_loss += loss.item()
                val_psnr += psnr_metric(sr_batch, gt_batch)
                val_batches += 1
                pbar_val.set_postfix({"l1": f"{loss.item():.4f}"})

                if args.max_val_batches and val_batches >= args.max_val_batches:
                    break

        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        avg_val_psnr = val_psnr / val_batches if val_batches > 0 else 0.0

        logger.info(
            "Epoch %d: train_l1=%.4f val_l1=%.4f val_psnr=%.2f lr=%.2e",
            epoch, avg_train_loss, avg_val_loss, avg_val_psnr, scheduler.get_last_lr()[0]
        )

        # ---------- Checkpointing ----------
        checkpoint_payload = {
            "epoch": epoch,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
        }

        last_path = checkpoint_dir / "last.pt"
        torch.save(checkpoint_payload, last_path)
        logger.info("Saved checkpoint: %s", last_path)

        save_every = args.save_every if args.save_every is not None else config.get("training", {}).get("save_every_n_epochs", 10)
        save_milestone = epoch == args.epochs or (save_every > 0 and epoch % save_every == 0)
        if save_milestone:
            ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            torch.save(checkpoint_payload, ckpt_path)
            logger.info("Saved milestone checkpoint: %s", ckpt_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint_payload, checkpoint_dir / "best.pt")
            logger.info("New best model (val_l1=%.4f)", best_val_loss)


if __name__ == "__main__":
    main()
