"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sacrebleu
from tqdm import tqdm
import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import (
    PAD_IDX, SOS_IDX, EOS_IDX,
    Vocabulary,
    get_dataloaders,
)
from lr_scheduler import NoamScheduler


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        p(correct class) = 1 - eps
        p(other classes) = eps / (vocab_size - 2)   # -1 true, -1 pad
        p(<pad>)         = 0

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # Flatten if 3-D input
        if logits.dim() == 3:
            B, T, V = logits.shape
            logits = logits.reshape(B * T, V)
            target = target.reshape(B * T)

        V = logits.size(-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)   # (N, V)

        # Build smooth target distribution
        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, self.smoothing / (V - 2))
            smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            smooth_dist[:, self.pad_idx] = 0.0   # pad never gets probability mass

        # KL-divergence style: -sum(p_smooth * log_q)
        loss = -(smooth_dist * log_probs).sum(dim=-1)   # (N,)

        # Exclude padding positions from the mean
        non_pad_mask = (target != self.pad_idx)
        loss = loss[non_pad_mask].mean()
        return loss


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).
    """
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        pbar = tqdm(
            data_iter,
            desc=f"{'Train' if is_train else 'Val  '} epoch {epoch_num:03d}",
            leave=False,
        )
        for src, tgt in pbar:
            src = src.to(device)   # (B, S)
            tgt = tgt.to(device)   # (B, T)

            # Teacher-forcing: decoder input drops last token,
            # target for loss drops first (<sos>) token
            tgt_in  = tgt[:, :-1]   # (B, T-1)
            tgt_out = tgt[:, 1:]    # (B, T-1)

            src_mask = make_src_mask(src, pad_idx=PAD_IDX).to(device)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx=PAD_IDX).to(device)

            logits = model(src, tgt_in, src_mask, tgt_mask)  # (B, T-1, V)
            loss   = loss_fn(logits, tgt_out)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_batches, 1)


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.
    """
    model.eval()
    src      = src.to(device)
    src_mask = src_mask.to(device)

    # Encode source once
    memory = model.encode(src, src_mask)   # (1, src_len, d_model)

    # Initialise decoder input with <sos>
    ys = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=PAD_IDX).to(device)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)  # (1, t, V)
            # Greedy: pick argmax at the last position
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
            ys = torch.cat([ys, next_tok], dim=1)                      # (1, t+1)
            if next_tok.item() == end_symbol:
                break

    return ys   # (1, out_len)


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab: Vocabulary,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Supports  tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    model.eval()
    hypotheses: list[str] = []
    references: list[str] = []

    special_ids = {PAD_IDX, SOS_IDX, EOS_IDX}

    for src, tgt in tqdm(test_dataloader, desc="BLEU eval", leave=False):
        for i in range(src.size(0)):
            single_src      = src[i].unsqueeze(0).to(device)      # (1, S)
            single_src_mask = make_src_mask(single_src, pad_idx=PAD_IDX).to(device)

            pred_ids = greedy_decode(
                model, single_src, single_src_mask,
                max_len=max_len,
                start_symbol=SOS_IDX,
                end_symbol=EOS_IDX,
                device=device,
            )  # (1, out_len)

            # Decode hypothesis — strip specials, stop at EOS
            hyp_tokens = []
            for idx in pred_ids.squeeze(0).tolist():
                if idx == EOS_IDX:
                    break
                if idx not in special_ids:
                    hyp_tokens.append(tgt_vocab.lookup_token(idx))

            # Decode reference — strip specials
            ref_tokens = [
                tgt_vocab.lookup_token(idx)
                for idx in tgt[i].tolist()
                if idx not in special_ids
            ]

            hypotheses.append(" ".join(hyp_tokens))
            references.append(" ".join(ref_tokens))

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'
    """
    # Collect constructor kwargs so the autograder can rebuild the model
    model_config = {
        "src_vocab_size": model.src_embedding.num_embeddings,
        "tgt_vocab_size": model.tgt_embedding.num_embeddings,
        "d_model":        model.d_model,
        "N":              len(model.encoder.layers),
        "num_heads":      model.encoder.layers[0].self_attn.num_heads,
        "d_ff":           model.encoder.layers[0].ffn.linear1.out_features,
        "dropout":        model.encoder.layers[0].dropout.p,
    }

    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "model_config":         model_config,
            "src_vocab":            getattr(model, "src_vocab", None),
            "tgt_vocab":            getattr(model, "tgt_vocab", None),
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).
    """
    ckpt = torch.load(path, map_location="cpu")

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return ckpt["epoch"]


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler
        7. Instantiate LabelSmoothingLoss
        8. Training loop
        9. Final BLEU on test set
    """

    # ── Hyperparameters ────────────────────────────────────────────────
    CONFIG = {
        # Architecture
        "d_model":        256,
        "N":              3,
        "num_heads":      8,
        "d_ff":           512,
        "dropout":        0.1,
        # Training
        "num_epochs":     20,
        "batch_size":     128,
        "warmup_steps":   4000,
        "label_smoothing": 0.1,
        "min_freq":       2,
        "max_len":        150,
    }

    # ── Device ─────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # ── 1. W&B init ────────────────────────────────────────────────────
    wandb.init(project="da6401-a3", config=CONFIG)
    cfg = wandb.config

    # ── 2 & 3. Data ────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=cfg.batch_size,
        min_freq=cfg.min_freq,
        max_len=cfg.max_len,
    )

    # ── 4. Model ───────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg.d_model,
        N=cfg.N,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    # Attach vocabs so save_checkpoint can store them
    # and model.infer() can tokenise without external setup
    model.src_vocab = src_vocab
    model.tgt_vocab = tgt_vocab

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    wandb.log({"n_params": n_params})

    # ── 5. Optimizer ───────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,               # Noam scheduler sets actual LR
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    # ── 6. Scheduler ───────────────────────────────────────────────────
    scheduler = NoamScheduler(
        optimizer,
        d_model=cfg.d_model,
        warmup_steps=cfg.warmup_steps,
    )

    # ── 7. Loss ────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=PAD_IDX,
        smoothing=cfg.label_smoothing,
    )

    # ── 8. Training loop ───────────────────────────────────────────────
    best_val_bleu = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()

        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device,
        )

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f}  ppl={math.exp(train_loss):.2f} | "
            f"val_loss={val_loss:.4f}  ppl={math.exp(val_loss):.2f} | "
            f"{elapsed:.0f}s"
        )

        # BLEU every 2 epochs (expensive — greedy decodes entire val set)
        val_bleu = 0.0
        if epoch % 2 == 0 or epoch == cfg.num_epochs:
            val_bleu = evaluate_bleu(model, val_loader, tgt_vocab, device=device)
            print(f"          val_bleu={val_bleu:.2f}")

            if val_bleu > best_val_bleu:
                best_val_bleu = val_bleu
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    path="checkpoints/best_model.pt",
                )
                print(f"          ✓ Saved best checkpoint (BLEU {best_val_bleu:.2f})")

        # Save latest checkpoint every epoch
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            path="checkpoints/latest.pt",
        )

        # W&B logging
        log_dict = {
            "epoch":      epoch,
            "train_loss": train_loss,
            "train_ppl":  math.exp(train_loss),
            "val_loss":   val_loss,
            "val_ppl":    math.exp(val_loss),
            "lr":         optimizer.param_groups[0]["lr"],
        }
        if val_bleu:
            log_dict["val_bleu"] = val_bleu
        wandb.log(log_dict)

    # ── 9. Final test BLEU ─────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation …")
    load_checkpoint("checkpoints/best_model.pt", model)
    test_bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"Test BLEU: {test_bleu:.2f}")
    wandb.log({"test_bleu": test_bleu})

    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_training_experiment()
