"""
training/trainer.py
====================
Training loop for the multilingual Transformer NMT system.

Covers (in order of execution):
    • Model + optimizer + scheduler construction from config
    • Mixed-precision training via torch.cuda.amp  (beyond-paper speedup)
    • Gradient accumulation for simulating large batches on small GPUs
    • Gradient clipping (max_norm=1.0, paper Section 5.3 footnote)
    • Checkpoint saving: best-BLEU model + rolling last-N checkpoints
    • Wandb logging: loss, LR, BLEU per language pair, attention heatmaps
    • Per-epoch validation with BLEU evaluation via sacrebleu

Paper references:
    Optimizer:  Section 5.3 — Adam β₁=0.9, β₂=0.98, ε=1e-9
    Regularisation: Section 5.4 — dropout=0.1, label smoothing ε=0.1
    Gradient clipping: not in the paper but universal practice; harmless
    Mixed precision: beyond the paper — purely a speed optimisation
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from data.tokenizer import MultilingualTokenizer
from evaluation.bleu import compute_corpus_bleu
from evaluation.beam_search import greedy_decode
from model.transformer import Transformer
from training.losses import LabelSmoothingLoss
from training.scheduler import build_noam_scheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path:         Path,
    model:        Transformer,
    optimizer:    torch.optim.Optimizer,
    scheduler:    torch.optim.lr_scheduler.LambdaLR,
    scaler:       Optional[GradScaler],
    global_step:  int,
    epoch:        int,
    best_bleu:    float,
    config:       dict,
) -> None:
    """
    Save all training state to a `.pt` file for resuming or inference.

    Parameters
    ----------
    path        : Path   Where to write the checkpoint.
    model       : Transformer
    optimizer   : Adam optimizer
    scheduler   : Noam LR scheduler
    scaler      : AMP GradScaler (or None if not using AMP)
    global_step : int   Total steps taken so far.
    epoch       : int   Last completed epoch (0-indexed).
    best_bleu   : float Best validation BLEU seen so far.
    config      : dict  Full Hydra config (for reproducibility).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state":    scaler.state_dict() if scaler else None,
        "global_step":     global_step,
        "epoch":           epoch,
        "best_bleu":       best_bleu,
        "config":          config,
    }
    torch.save(payload, path)
    logger.info("Checkpoint saved → %s  (step=%d, epoch=%d)", path, global_step, epoch)


def load_checkpoint(
    path:      Path,
    model:     Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    scaler:    Optional[GradScaler] = None,
    device:    str = "cpu",
) -> Tuple[int, int, float]:
    """
    Load a checkpoint into model (and optionally optimizer/scheduler).

    Parameters
    ----------
    path      : Path   Checkpoint file.
    model     : Transformer   Will receive the saved weights.
    optimizer : optional      If provided, optimizer state is restored.
    scheduler : optional      If provided, scheduler state is restored.
    scaler    : optional      If provided, AMP scaler state is restored.
    device    : str           Where to map the tensors ('cpu', 'cuda', etc.)

    Returns
    -------
    (global_step, epoch, best_bleu)
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and ckpt.get("optimizer_state"):
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler and ckpt.get("scaler_state"):
        scaler.load_state_dict(ckpt["scaler_state"])
    logger.info(
        "Loaded checkpoint '%s'  step=%d  epoch=%d  best_bleu=%.2f",
        path, ckpt["global_step"], ckpt["epoch"], ckpt["best_bleu"],
    )
    return ckpt["global_step"], ckpt["epoch"], ckpt["best_bleu"]


def _prune_checkpoints(ckpt_dir: Path, keep: int) -> None:
    """
    Delete oldest epoch checkpoints, keeping only the `keep` most recent.
    Best-BLEU checkpoint is never deleted (its filename starts with 'best').
    """
    epoch_ckpts = sorted(
        [f for f in ckpt_dir.glob("epoch_*.pt")],
        key=lambda f: f.stat().st_mtime,
    )
    for old in epoch_ckpts[:-keep]:
        old.unlink()
        logger.debug("Pruned old checkpoint: %s", old)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Encapsulates the full training + validation loop.

    Parameters
    ----------
    config    : dict                  Full Hydra config.
    model     : Transformer
    tokenizer : MultilingualTokenizer
    train_loader : DataLoader         Token-bucket batched train set.
    val_loader   : DataLoader         Validation set.
    device    : str                   'cuda', 'mps', or 'cpu'.
    resume_from : Path | None         Optional checkpoint to resume from.
    """

    def __init__(
        self,
        config:       dict,
        model:        Transformer,
        tokenizer:    MultilingualTokenizer,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        device:       str = "cuda",
        resume_from:  Optional[Path] = None,
    ) -> None:
        self.config     = config
        self.model      = model.to(device)
        self.tokenizer  = tokenizer
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device     = device

        train_cfg   = config["Training"]
        model_cfg   = config["Modelling"]

        # --- Optimizer (paper Section 5.3) ---
        # Base lr=1.0: the Noam schedule carries all the scaling
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1.0,
            betas=(train_cfg["adam_beta1"], train_cfg["adam_beta2"]),
            eps=train_cfg["adam_eps"],
        )

        # --- LR scheduler (paper eq. 3) ---
        self.scheduler = build_noam_scheduler(
            self.optimizer,
            d_model=model_cfg["d_model"],
            warmup_steps=train_cfg["warmup_steps"],
        )

        # --- Loss (paper Section 5.4) ---
        self.criterion = LabelSmoothingLoss(
            vocab_size=model_cfg["tgt_vocab_size"],
            pad_idx=tokenizer.pad_id,
            smoothing=train_cfg["label_smoothing"],
        )

        # --- Mixed precision (beyond-paper) ---
        self.use_amp   = train_cfg["use_amp"] and device == "cuda"
        self.scaler    = GradScaler("cuda") if self.use_amp else None

        # --- Training state ---
        self.global_step  = 0
        self.start_epoch  = 0
        self.best_bleu    = 0.0
        self.grad_accum   = train_cfg["grad_accum_steps"]
        self.max_grad_norm = train_cfg["gradient_clip"]

        # --- Config shortcuts ---
        self.max_epochs          = train_cfg["max_epochs"]
        self.max_steps           = train_cfg["max_steps"]
        self.log_every           = train_cfg["log_every_n_steps"]
        self.eval_every_epoch    = train_cfg["eval_every_n_epochs"]
        self.save_every_epoch    = train_cfg["save_every_n_epochs"]
        self.keep_n_ckpts        = train_cfg["keep_last_n_checkpoints"]
        self.ckpt_dir            = Path(train_cfg["checkpoint_dir"])

        # --- Wandb (optional) ---
        self._init_wandb()

        # --- Resume from checkpoint ---
        if resume_from is not None:
            self.global_step, self.start_epoch, self.best_bleu = load_checkpoint(
                resume_from, self.model, self.optimizer, self.scheduler, self.scaler, device
            )
            self.start_epoch += 1   # resume AFTER the saved epoch

        logger.info(
            "Trainer ready | device=%s  amp=%s  grad_accum=%d  max_epochs=%d",
            device, self.use_amp, self.grad_accum, self.max_epochs,
        )

    # ------------------------------------------------------------------
    # wandb
    # ------------------------------------------------------------------

    def _init_wandb(self) -> None:
        """Initialise wandb if a project name is configured."""
        wandb_cfg = self.config.get("Wandb", {})
        project   = wandb_cfg.get("project", "")
        self._wandb = None

        if not project:
            logger.info("Wandb project not set — skipping wandb logging.")
            return

        try:
            import wandb
            self._wandb = wandb
            wandb.init(
                project=project,
                entity=wandb_cfg.get("entity") or None,
                config=self.config,
                resume="allow",
            )
            # Log model parameter count
            n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            wandb.run.summary["n_params"] = n_params
            logger.info("Wandb initialised: project=%s", project)
        except ImportError:
            logger.warning("wandb not installed — install with: pip install wandb")

    def _wandb_log(self, payload: dict) -> None:
        """Log a dict to wandb if available, otherwise no-op."""
        if self._wandb:
            self._wandb.log(payload, step=self.global_step)

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def _train_step(self, batch: Dict) -> float:
        """
        Forward + backward for one batch.

        Returns
        -------
        float   Loss value (for logging; detached from graph).
        """
        src    = batch["src"].to(self.device)
        tgt_in = batch["tgt_in"].to(self.device)
        tgt_out = batch["tgt_out"].to(self.device)

        # Build masks
        src_mask = Transformer.make_src_mask(src, self.tokenizer.pad_id)
        tgt_mask = Transformer.make_tgt_mask(tgt_in, self.tokenizer.pad_id)

        # Forward pass — wrapped in autocast for AMP
        with autocast("cuda", enabled=self.use_amp):
            logits = self.model(src, tgt_in, src_mask, tgt_mask)
            # Reshape for loss: (batch * seq_len, vocab_size) vs (batch * seq_len,)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                tgt_out.view(-1),
            )
            # Scale loss by accumulation factor so gradients are correctly
            # normalised when we call optimizer.step() after N micro-batches
            loss = loss / self.grad_accum

        # Backward
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.grad_accum   # return unscaled value for logging

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """
        Run greedy-decode on the validation set and compute BLEU per language pair.

        Returns
        -------
        dict mapping "bleu_{src}-{tgt}" → float
        """
        self.model.eval()
        # Collect hypotheses and references grouped by language pair
        hyp_by_pair: Dict[str, list] = {}
        ref_by_pair: Dict[str, list] = {}
        MAX_VAL_BATCHES = 200
        for batch_idx, batch in enumerate(self.val_loader):
            if batch_idx >= MAX_VAL_BATCHES:
                break
            if batch_idx % 20 == 0:
                logger.info("  Validating batch %d/%d ...", batch_idx, MAX_VAL_BATCHES)
            src    = batch["src"].to(self.device)
            src_langs = batch["src_langs"]
            tgt_langs = batch["tgt_langs"]
            ref_texts = batch["tgt_texts"]

            src_mask = Transformer.make_src_mask(src, self.tokenizer.pad_id)
            print("STTTTTTTTTTTTTTTTARRRRRRRRRT OOOOOFFFFFFFF GGGGGRRREDDDDY DEEEEEEEEECCCCCCCCODER")
            # Greedy decode (beam search used at final eval — see evaluation/)
            pred_ids = greedy_decode(
                model=self.model,
                src=src,
                src_mask=src_mask,
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id,
                max_len=self.config["Evaluation"]["max_decode_steps"],
                device=self.device,
            )
            print("ENDDDDDDDDDDDDDDDDDDDD OOOOOFFFFFFFF GGGGGRRREDDDDY DEEEEEEEEECCCCCCCCODER")
            
            for i, pred in enumerate(pred_ids):
                pair_key = f"{src_langs[i]}-{tgt_langs[i]}"
                hyp = self.tokenizer.decode(pred, skip_special_tokens=True)
                ref = ref_texts[i]
                hyp_by_pair.setdefault(pair_key, []).append(hyp)
                ref_by_pair.setdefault(pair_key, []).append([ref])


        print("==================================================================================================")

        bleu_scores: Dict[str, float] = {}
        for pair_key in hyp_by_pair:
            bleu = compute_corpus_bleu(hyp_by_pair[pair_key], ref_by_pair[pair_key])
            bleu_scores[f"bleu/{pair_key}"] = bleu
            logger.info("  [%s]  BLEU = %.2f", pair_key, bleu)

        self.model.train()
        return bleu_scores

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """
        Run the full training loop from `start_epoch` to `max_epochs`.

        Loop structure:
            for epoch in range(max_epochs):
                for batch in train_loader:
                    forward → loss → backward
                    every grad_accum steps: clip + optimizer.step + scheduler.step
                    every log_every steps:  log loss + LR to wandb
                if epoch % eval_every: validate → log BLEU → maybe save best
                if epoch % save_every: save epoch checkpoint
        """
        logger.info("=== Training start  epochs=%d  steps=%d ===",
                    self.max_epochs, self.max_steps)

        self.model.train()
        self.optimizer.zero_grad()

        for epoch in range(self.start_epoch, self.max_epochs):
            epoch_loss   = 0.0
            epoch_tokens = 0
            t0 = time.time()

            for step_in_epoch, batch in enumerate(self.train_loader):

                # ---- Forward + backward ----
                loss = self._train_step(batch)
                epoch_loss += loss
                epoch_tokens += batch["src"].numel() + batch["tgt_in"].numel()
                print(loss)
                # ---- Optimizer step (every grad_accum micro-batches) ----
                micro_step = (step_in_epoch + 1)
                if micro_step % self.grad_accum == 0:
                    # Unscale before clipping so the clip threshold is in fp32
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)

                    # Gradient clipping — prevents rare exploding-gradient spikes
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    # Noam schedule: one step per optimizer update (not per epoch)
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # ---- Logging ----
                    if self.global_step % self.log_every == 0:
                        current_lr = self.scheduler.get_last_lr()[0]
                        logger.info(
                            "epoch=%d  step=%d  loss=%.4f  lr=%.2e",
                            epoch, self.global_step, loss, current_lr,
                        )
                        self._wandb_log({
                            "train/loss": loss,
                            "train/lr":   current_lr,
                            "train/epoch": epoch,
                        })

                    # Hard step ceiling
                    if self.global_step >= self.max_steps:
                        logger.info("Reached max_steps=%d — stopping.", self.max_steps)
                        self._end_of_epoch(epoch, epoch_loss, epoch_tokens, t0)
                        return

            # ---- End of epoch ----
            self._end_of_epoch(epoch, epoch_loss, epoch_tokens, t0)

        logger.info("=== Training complete ===")
        if self._wandb:
            self._wandb.finish()

    def _end_of_epoch(
        self,
        epoch:        int,
        epoch_loss:   float,
        epoch_tokens: int,
        t0:           float,
    ) -> None:
        """Validation, BLEU logging, and checkpointing at epoch end."""
        elapsed   = time.time() - t0
        avg_loss  = epoch_loss / max(len(self.train_loader), 1)
        tok_per_s = epoch_tokens / elapsed
        print(f"qqqqqqqqqqqqqqqqqqqqqqqqqqq{avg_loss}")
        print(f"qqqqqqqqqqqqqqqqqqqqqqqqqqq{epoch_loss}")
        print(f"qqqqqqqqqqqqqqqqqqqqqqqqqqq{len(self.train_loader)}")

        logger.info(
            "── Epoch %d done  avg_loss=%.4f  tokens/s=%.0f  time=%.1fs",
            epoch, avg_loss, tok_per_s, elapsed,
        )

        # Validation + BLEU
        if (epoch + 1) % self.eval_every_epoch == 0:
            bleu_scores = self._validate(epoch)
            mean_bleu   = sum(bleu_scores.values()) / max(len(bleu_scores), 1)

            self._wandb_log({
                "val/avg_bleu": mean_bleu,
                "val/epoch": epoch,
                **bleu_scores,
            })

            # Save best-BLEU checkpoint
            if mean_bleu > self.best_bleu:
                self.best_bleu = mean_bleu
                save_checkpoint(
                    path=self.ckpt_dir / "best_bleu.pt",
                    model=self.model, optimizer=self.optimizer,
                    scheduler=self.scheduler, scaler=self.scaler,
                    global_step=self.global_step, epoch=epoch,
                    best_bleu=self.best_bleu, config=self.config,
                )
                logger.info("  ★ New best BLEU: %.2f", self.best_bleu)

        # Rolling epoch checkpoint
        if (epoch + 1) % self.save_every_epoch == 0:
            save_checkpoint(
                path=self.ckpt_dir / f"epoch_{epoch:03d}.pt",
                model=self.model, optimizer=self.optimizer,
                scheduler=self.scheduler, scaler=self.scaler,
                global_step=self.global_step, epoch=epoch,
                best_bleu=self.best_bleu, config=self.config,
            )
            _prune_checkpoints(self.ckpt_dir, self.keep_n_ckpts)