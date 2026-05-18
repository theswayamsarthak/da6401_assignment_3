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
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  1. STANDALONE ATTENTION FUNCTION
#     Exposed at module level so the autograder can import and test it
#     independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)

    # (..., seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # mask=True  → masked out → fill with -inf before softmax
        scores = scores.masked_fill(mask, float("-inf"))

    attn_w = F.softmax(scores, dim=-1)
    # Guard against NaN from all-masked rows (softmax of all -inf → NaN)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)

    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  2. MASK HELPERS
#     Exposed at module level so they can be tested independently and
#     reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    # (batch, 1, 1, src_len) — True where pad
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    tgt_len = tgt.size(1)

    # Causal mask: upper-triangular (above diagonal) = True = masked future
    # shape (tgt_len, tgt_len) → (1, 1, tgt_len, tgt_len)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)                              # (1, 1, T, T)

    # Padding mask: True where pad — shape (batch, 1, 1, tgt_len)
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)   # (B, 1, 1, T)

    # Combined: mask position if it is causal-future OR padding
    tgt_mask = causal_mask | pad_mask                        # (B, 1, T, T)
    return tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  3. MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head

        # Four projection matrices (no bias, per paper)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)

        # Stored for external visualisation (attention head ablation)
        self.attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) → (batch, heads, seq, d_k)"""
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, heads, seq, d_k) → (batch, seq, d_model)"""
        B, _, S, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out

        Returns:
            output : shape [batch, seq_q, d_model]
        """
        Q = self._split_heads(self.w_q(query))   # (B, H, S_q, d_k)
        K = self._split_heads(self.w_k(key))     # (B, H, S_k, d_k)
        V = self._split_heads(self.w_v(value))   # (B, H, S_k, d_k)

        context, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        output = self.w_o(self._merge_heads(context))
        return output


# ══════════════════════════════════════════════════════════════════════
#  4. POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    The encoding table is registered as a buffer (not a trainable parameter).

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, d_model)                       # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()       # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                               # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)   # odd  dimensions

        pe = pe.unsqueeze(0)                            # (1, max_len, d_model)
        self.register_buffer("pe", pe)                 # ← buffer, never a gradient param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  5. FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  6. ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Uses Pre-LayerNorm (normalise input before sub-layer) for training
    stability.  See Xiong et al. (2020) for justification.

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]
        """
        # Pre-LN self-attention + residual connection
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.self_attn(x, x, x, src_mask))

        # Pre-LN FFN + residual connection
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))

        return x


# ══════════════════════════════════════════════════════════════════════
#  7. DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

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
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        # 1. Masked self-attention (causal)
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.self_attn(x, x, x, tgt_mask))

        # 2. Cross-attention over encoder memory
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.cross_attn(x, memory, memory, src_mask))

        # 3. Position-wise FFN
        residual = x
        x = self.norm3(x)
        x = residual + self.dropout(self.ffn(x))

        return x


# ══════════════════════════════════════════════════════════════════════
#  8. ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

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
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  9. FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
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
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Token embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=1)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=1)

        # Sinusoidal positional encodings (registered as buffers)
        self.src_pe = PositionalEncoding(d_model, dropout)
        self.tgt_pe = PositionalEncoding(d_model, dropout)

        # Encoder / Decoder stacks
        enc_layer    = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer    = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        # Final linear projection to target vocabulary
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # Xavier uniform weight initialisation
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
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
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        x = self.tgt_pe(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        dec_out = self.decoder(x, memory, src_mask, tgt_mask)
        return self.output_projection(dec_out)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_text: str, max_len: int = 128) -> str:
        """
        Translate a raw German string to English.

        Fully self-contained — loads spaCy and vocabs from the checkpoint
        that was used to restore this model. No external setup required.

        The autograder calls:  model.infer(src_text) -> str

        Args:
            src_text : Raw German sentence string.
            max_len  : Maximum output length tokens.

        Returns:
            Translated English sentence as a plain string.
        """
        import spacy, torch, os

        self.eval()
        device = next(self.parameters()).device

        # ── Load vocabs from checkpoint if not already attached ───────
        if not hasattr(self, "_infer_ready"):
            # Find checkpoint — try common locations
            for ckpt_path in [
                "checkpoints/best_model.pt",
                "checkpoint.pt",
                "best_model.pt",
                "checkpoints/latest.pt",
            ]:
                if os.path.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    self.src_vocab = ckpt.get("src_vocab")
                    self.tgt_vocab = ckpt.get("tgt_vocab")
                    break

            # Load spaCy German tokeniser
            try:
                self._de_nlp = spacy.load(
                    "de_core_news_sm",
                    disable=["ner", "parser", "tagger", "lemmatizer"],
                )
            except OSError:
                # Fallback: simple whitespace tokeniser
                class _WS:
                    def __call__(self, text):
                        class _Tok:
                            def __init__(self, t): self.text = t
                        return [_Tok(w) for w in text.split()]
                self._de_nlp = _WS()

            self._infer_ready = True

        # ── Special token indices ─────────────────────────────────────
        sv = self.src_vocab
        tv = self.tgt_vocab

        src_UNK = sv.stoi.get("<unk>", 0)
        src_SOS = sv.stoi.get("<sos>", 2)
        src_EOS = sv.stoi.get("<eos>", 3)
        src_PAD = sv.stoi.get("<pad>", 1)

        tgt_SOS = tv.stoi.get("<sos>", 2)
        tgt_EOS = tv.stoi.get("<eos>", 3)
        tgt_PAD = tv.stoi.get("<pad>", 1)

        # ── 1. Tokenise + encode source ───────────────────────────────
        tokens  = [t.text.lower() for t in self._de_nlp(src_text.strip())]
        src_ids = (
            [src_SOS]
            + [sv.stoi.get(t, src_UNK) for t in tokens]
            + [src_EOS]
        )
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = (src == src_PAD).unsqueeze(1).unsqueeze(2)   # (1,1,1,S)

        # ── 2. Greedy decode ──────────────────────────────────────────
        memory = self.encode(src, src_mask)
        ys = torch.tensor([[tgt_SOS]], dtype=torch.long, device=device)

        with torch.no_grad():
            for _ in range(max_len - 1):
                T = ys.size(1)
                causal = torch.triu(
                    torch.ones(T, T, device=device, dtype=torch.bool),
                    diagonal=1,
                ).unsqueeze(0).unsqueeze(0)                      # (1,1,T,T)
                pad_m    = (ys == tgt_PAD).unsqueeze(1).unsqueeze(2)
                tgt_mask = causal | pad_m

                logits   = self.decode(memory, src_mask, ys, tgt_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_tok], dim=1)
                if next_tok.item() == tgt_EOS:
                    break

        # ── 3. Decode indices to string ───────────────────────────────
        special = {tgt_SOS, tgt_EOS, tgt_PAD}
        out = [
            tv.itos[i]
            for i in ys.squeeze(0).tolist()
            if i not in special and i < len(tv.itos)
        ]
        return " ".join(out)
