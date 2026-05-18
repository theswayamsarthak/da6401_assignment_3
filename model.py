"""
model.py — Transformer Architecture Implementation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  1. SCALED DOT-PRODUCT ATTENTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : (..., seq_q, d_k)
        K    : (..., seq_k, d_k)
        V    : (..., seq_k, d_v)
        mask : BoolTensor broadcastable to (..., seq_q, seq_k)
               True → masked out (set to -inf before softmax)
    Returns:
        output : (..., seq_q, d_v)
        attn_w : (..., seq_q, seq_k)
    """
    d_k    = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_w = F.softmax(scores, dim=-1)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)   # guard all-masked rows

    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  2. MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Padding mask for encoder.
    Returns [batch, 1, 1, src_len].  True = PAD (mask out).
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Combined causal + padding mask for decoder.
    Returns [batch, 1, tgt_len, tgt_len].  True = mask out.
    """
    tgt_len  = tgt.size(1)
    causal   = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)                              # (1,1,T,T)
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)    # (B,1,1,T)
    return causal | pad_mask                                  # (B,1,T,T)


# ══════════════════════════════════════════════════════════════════════
#  3. MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention — §3.2.2.
    Does NOT use torch.nn.MultiheadAttention.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

        self.attn_weights: Optional[torch.Tensor] = None

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """(B, S, d_model) → (B, H, S, d_k)"""
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, S, d_k) → (B, S, d_model)"""
        B, _, S, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Q = self._split(self.w_q(query))
        K = self._split(self.w_k(key))
        V = self._split(self.w_v(value))
        ctx, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        return self.w_o(self._merge(ctx))


# ══════════════════════════════════════════════════════════════════════
#  4. POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal PE — §3.5.
    Registered as a buffer (not a trainable parameter).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  5. FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, xW₁+b₁)W₂+b₂  — §3.3"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  6. ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """Pre-LN: x → [Self-Attn → Add] → [FFN → Add]"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        r = x; x = self.norm1(x)
        x = r + self.dropout(self.self_attn(x, x, x, src_mask))
        r = x; x = self.norm2(x)
        x = r + self.dropout(self.ffn(x))
        return x


# ══════════════════════════════════════════════════════════════════════
#  7. DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """Pre-LN: x → [Masked Self-Attn] → [Cross-Attn] → [FFN]"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        r = x; x = self.norm1(x)
        x = r + self.dropout(self.self_attn(x, x, x, tgt_mask))
        r = x; x = self.norm2(x)
        x = r + self.dropout(self.cross_attn(x, memory, memory, src_mask))
        r = x; x = self.norm3(x)
        x = r + self.dropout(self.ffn(x))
        return x


# ══════════════════════════════════════════════════════════════════════
#  8. ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  9. FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

# Google Drive file ID for best_model.pt
# Checkpoint contains: model_state_dict, model_config, src_vocab, tgt_vocab
_GDRIVE_FILE_ID = "15yxkutUFRKzq0VJXjZGiAN0LdWtLLMNN"
_DEFAULT_CKPT   = "best_model.pt"


class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer.

    When checkpoint_path is provided (or on first infer() call),
    weights + vocabs are loaded from Google Drive automatically.
    """

    def __init__(
        self,
        src_vocab_size: int   = 8000,
        tgt_vocab_size: int   = 8000,
        d_model:        int   = 512,
        N:              int   = 6,
        num_heads:      int   = 8,
        d_ff:           int   = 2048,
        dropout:        float = 0.1,
        checkpoint_path: str  = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.src_embedding     = nn.Embedding(src_vocab_size, d_model, padding_idx=1)
        self.tgt_embedding     = nn.Embedding(tgt_vocab_size, d_model, padding_idx=1)
        self.src_pe            = PositionalEncoding(d_model, dropout)
        self.tgt_pe            = PositionalEncoding(d_model, dropout)

        enc_layer    = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer    = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Runtime state for infer()
        self.src_vocab = None
        self.tgt_vocab = None
        self._de_nlp   = None
        self._ready    = False

        # Download and load if path provided
        if checkpoint_path is not None:
            self._load(checkpoint_path)

    # ── Download + load weights ───────────────────────────────────────

    def _download(self, path: str) -> None:
        """Download checkpoint from Google Drive if not already present."""
        if not os.path.exists(path):
            print(f"Downloading checkpoint → {path}")
            gdown.download(id=_GDRIVE_FILE_ID, output=path, quiet=False)

    def _load(self, path: str) -> None:
        """Load weights and vocabs from checkpoint file."""
        self._download(path)
        ckpt = torch.load(path, map_location="cpu")

        # Resize embeddings/projection if checkpoint vocab sizes differ
        cfg    = ckpt.get("model_config", {})
        src_vs = cfg.get("src_vocab_size", self.src_embedding.num_embeddings)
        tgt_vs = cfg.get("tgt_vocab_size", self.tgt_embedding.num_embeddings)

        if src_vs != self.src_embedding.num_embeddings:
            self.src_embedding = nn.Embedding(src_vs, self.d_model, padding_idx=1)
        if tgt_vs != self.tgt_embedding.num_embeddings:
            self.tgt_embedding     = nn.Embedding(tgt_vs, self.d_model, padding_idx=1)
            self.output_projection = nn.Linear(self.d_model, tgt_vs)

        self.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.src_vocab = ckpt.get("src_vocab")
        self.tgt_vocab = ckpt.get("tgt_vocab")
        self._ready    = True
        print(f"Loaded checkpoint from {path} ✓")

    def _ensure_ready(self) -> None:
        """Called at the start of infer() — loads everything if not done yet."""
        if self._ready:
            return

        # Try to find checkpoint locally first, then download
        for path in [
            _DEFAULT_CKPT,
            "checkpoints/best_model.pt",
            "checkpoint.pt",
            "checkpoints/latest.pt",
        ]:
            if os.path.exists(path):
                self._load(path)
                break
        else:
            # Nothing found locally — download to default path
            self._load(_DEFAULT_CKPT)

        # Load spaCy German tokeniser
        if self._de_nlp is None:
            try:
                import spacy
                self._de_nlp = spacy.load(
                    "de_core_news_sm",
                    disable=["ner", "parser", "tagger", "lemmatizer"],
                )
            except Exception:
                class _WS:
                    def __call__(self, text):
                        class _T:
                            def __init__(self, w): self.text = w
                        return [_T(w) for w in text.split()]
                self._de_nlp = _WS()

    # ── AUTOGRADER HOOKS ──────────────────────────────────────────────

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src      : [batch, src_len]
            src_mask : [batch, 1, 1, src_len]
        Returns:
            memory   : [batch, src_len, d_model]
        """
        x = self.src_pe(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            memory   : [batch, src_len, d_model]
            src_mask : [batch, 1, 1, src_len]
            tgt      : [batch, tgt_len]
            tgt_mask : [batch, 1, tgt_len, tgt_len]
        Returns:
            logits   : [batch, tgt_len, tgt_vocab_size]
        """
        x = self.tgt_pe(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        return self.output_projection(self.decoder(x, memory, src_mask, tgt_mask))

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            src      : [batch, src_len]
            tgt      : [batch, tgt_len]
            src_mask : [batch, 1, 1, src_len]
            tgt_mask : [batch, 1, tgt_len, tgt_len]
        Returns:
            logits   : [batch, tgt_len, tgt_vocab_size]
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        """
        Translate a raw German sentence to English using greedy decoding.

        Fully self-contained:
          - Downloads checkpoint from Google Drive if not present locally
          - Loads vocab from checkpoint
          - Loads spaCy de_core_news_sm for tokenisation

        Args:
            src_sentence : Raw German text string.
        Returns:
            Translated English sentence as a plain string.
        """
        self.eval()
        self._ensure_ready()

        device = next(self.parameters()).device
        sv     = self.src_vocab
        tv     = self.tgt_vocab

        # Special token indices
        src_UNK = sv.stoi.get("<unk>", 0)
        src_SOS = sv.stoi.get("<sos>", 2)
        src_EOS = sv.stoi.get("<eos>", 3)
        src_PAD = sv.stoi.get("<pad>", 1)
        tgt_SOS = tv.stoi.get("<sos>", 2)
        tgt_EOS = tv.stoi.get("<eos>", 3)
        tgt_PAD = tv.stoi.get("<pad>", 1)

        # 1. Tokenise + encode source
        tokens  = [t.text.lower() for t in self._de_nlp(src_sentence.strip())]
        src_ids = (
            [src_SOS]
            + [sv.stoi.get(t, src_UNK) for t in tokens]
            + [src_EOS]
        )
        src      = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = (src == src_PAD).unsqueeze(1).unsqueeze(2)

        # 2. Greedy decode
        memory = self.encode(src, src_mask)
        ys     = torch.tensor([[tgt_SOS]], dtype=torch.long, device=device)

        with torch.no_grad():
            for _ in range(128):
                T        = ys.size(1)
                causal   = torch.triu(
                    torch.ones(T, T, device=device, dtype=torch.bool),
                    diagonal=1,
                ).unsqueeze(0).unsqueeze(0)
                tgt_mask = causal | (ys == tgt_PAD).unsqueeze(1).unsqueeze(2)
                logits   = self.decode(memory, src_mask, ys, tgt_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys       = torch.cat([ys, next_tok], dim=1)
                if next_tok.item() == tgt_EOS:
                    break

        # 3. Decode indices → string
        special = {tgt_SOS, tgt_EOS, tgt_PAD}
        out     = [
            tv.itos[i]
            for i in ys.squeeze(0).tolist()
            if i not in special and i < len(tv.itos)
        ]
        return " ".join(out)
