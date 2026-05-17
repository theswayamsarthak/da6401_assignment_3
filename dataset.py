"""
dataset.py — Multi30k Dataset Pipeline
DA6401 Assignment 3

Provides:
  - Multi30kDataset : loads data, builds vocab, processes tokens
  - get_dataloaders : convenience wrapper returning DataLoaders + vocabs
  - collate_fn      : dynamic padding for DataLoader batches
"""

from __future__ import annotations

from collections import Counter
from functools import partial
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset   # pip install datasets
import spacy                         # pip install spacy


# ──────────────────────────────────────────────────────────────────────
# Special-token constants  (pad=1 matches skeleton default pad_idx=1)
# ──────────────────────────────────────────────────────────────────────

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

# These indices match pad_idx=1 expected by make_src_mask / make_tgt_mask
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


# ──────────────────────────────────────────────────────────────────────
# Vocabulary helper
# ──────────────────────────────────────────────────────────────────────

class Vocabulary:
    """
    Bidirectional token ↔ index mapping.

    Special tokens always at fixed indices:
        0: <unk>
        1: <pad>
        2: <sos>
        3: <eos>
    """

    def __init__(self):
        self.itos: list[str] = [UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(self.itos)}

    # ── torchtext-compatible aliases ──────────────────────────────────
    def lookup_token(self, idx: int) -> str:
        """Return the token string for a given index."""
        if idx < len(self.itos):
            return self.itos[idx]
        return UNK_TOKEN

    def lookup_indices(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, UNK_IDX) for t in tokens]

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, UNK_IDX)

    def _add(self, token: str):
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)


# ──────────────────────────────────────────────────────────────────────
# Main dataset class  (matches the skeleton API exactly)
# ──────────────────────────────────────────────────────────────────────

class Multi30kDataset(Dataset):
    """
    Loads the Multi30k dataset (bentrevett/multi30k) from HuggingFace
    and prepares spaCy tokenizers for German → English NMT.

    After construction call:
        dataset.build_vocab()   # once, on the training split
        dataset.process_data()  # converts sentences to integer lists

    Args:
        split    : "train", "validation", or "test"
        src_vocab: Optional pre-built source Vocabulary (for val/test splits)
        tgt_vocab: Optional pre-built target Vocabulary (for val/test splits)
        min_freq : Minimum token frequency for vocab inclusion (train only)
        max_len  : Drop sentence pairs longer than this (tokens)
    """

    def __init__(
        self,
        split: str = "train",
        src_vocab: Optional[Vocabulary] = None,
        tgt_vocab: Optional[Vocabulary] = None,
        min_freq: int = 2,
        max_len: int = 150,
    ):
        self.split     = split
        self.min_freq  = min_freq
        self.max_len   = max_len

        # ── Load spaCy models ──────────────────────────────────────────
        try:
            self._de_nlp = spacy.load(
                "de_core_news_sm",
                disable=["ner", "parser", "tagger", "lemmatizer"],
            )
        except OSError:
            raise OSError(
                "spaCy German model not found. "
                "Run: python -m spacy download de_core_news_sm"
            )
        try:
            self._en_nlp = spacy.load(
                "en_core_web_sm",
                disable=["ner", "parser", "tagger", "lemmatizer"],
            )
        except OSError:
            raise OSError(
                "spaCy English model not found. "
                "Run: python -m spacy download en_core_web_sm"
            )

        # ── Load raw HuggingFace dataset ──────────────────────────────
        raw = load_dataset("bentrevett/multi30k", split=split)
        self._raw_de: list[str] = [ex["de"] for ex in raw]
        self._raw_en: list[str] = [ex["en"] for ex in raw]

        # ── Tokenise once up-front ────────────────────────────────────
        self._tok_de: list[list[str]] = [
            [t.text.lower() for t in self._de_nlp(s.strip())]
            for s in self._raw_de
        ]
        self._tok_en: list[list[str]] = [
            [t.text.lower() for t in self._en_nlp(s.strip())]
            for s in self._raw_en
        ]

        # Vocab placeholders
        self.src_vocab: Optional[Vocabulary] = src_vocab
        self.tgt_vocab: Optional[Vocabulary] = tgt_vocab

        # Processed integer sequences (filled by process_data)
        self.src_data: list[list[int]] = []
        self.tgt_data: list[list[int]] = []

        # If vocabs are supplied (val/test) we can process immediately
        if src_vocab is not None and tgt_vocab is not None:
            self.process_data()

    # ------------------------------------------------------------------
    def tokenize_src(self, text: str) -> list[str]:
        return [t.text.lower() for t in self._de_nlp(text.strip())]

    def tokenize_tgt(self, text: str) -> list[str]:
        return [t.text.lower() for t in self._en_nlp(text.strip())]

    # ------------------------------------------------------------------
    def build_vocab(self) -> tuple[Vocabulary, Vocabulary]:
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>

        Should only be called on the training split.

        Returns:
            (src_vocab, tgt_vocab)
        """
        src_counter: Counter = Counter()
        tgt_counter: Counter = Counter()

        for tokens in self._tok_de:
            src_counter.update(tokens)
        for tokens in self._tok_en:
            tgt_counter.update(tokens)

        src_vocab = Vocabulary()
        for token, freq in src_counter.most_common():
            if freq >= self.min_freq:
                src_vocab._add(token)

        tgt_vocab = Vocabulary()
        for token, freq in tgt_counter.most_common():
            if freq >= self.min_freq:
                tgt_vocab._add(token)

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        return src_vocab, tgt_vocab

    # ------------------------------------------------------------------
    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary.

        Drops pairs where either side exceeds max_len tokens.
        Adds <sos> and <eos> to each sequence.
        """
        assert self.src_vocab is not None and self.tgt_vocab is not None, \
            "Call build_vocab() before process_data(), or pass vocabs to constructor."

        self.src_data = []
        self.tgt_data = []

        for src_tokens, tgt_tokens in zip(self._tok_de, self._tok_en):
            if len(src_tokens) > self.max_len or len(tgt_tokens) > self.max_len:
                continue

            src_ids = (
                [SOS_IDX]
                + [self.src_vocab.stoi.get(t, UNK_IDX) for t in src_tokens]
                + [EOS_IDX]
            )
            tgt_ids = (
                [SOS_IDX]
                + [self.tgt_vocab.stoi.get(t, UNK_IDX) for t in tgt_tokens]
                + [EOS_IDX]
            )
            self.src_data.append(src_ids)
            self.tgt_data.append(tgt_ids)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.src_data[idx], dtype=torch.long),
            torch.tensor(self.tgt_data[idx], dtype=torch.long),
        )


# ──────────────────────────────────────────────────────────────────────
# Collate — dynamic padding per batch
# ──────────────────────────────────────────────────────────────────────

def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    pad_idx: int = PAD_IDX,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded


# ──────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ──────────────────────────────────────────────────────────────────────

def get_dataloaders(
    batch_size: int = 128,
    min_freq:   int = 2,
    max_len:    int = 150,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, Vocabulary, Vocabulary]:
    """
    Build and return train/val/test DataLoaders plus the two vocabularies.

    Vocabularies are built from the training split only, then reused for
    val and test.

    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    print("Building training dataset & vocabulary …")
    train_ds = Multi30kDataset("train", min_freq=min_freq, max_len=max_len)
    src_vocab, tgt_vocab = train_ds.build_vocab()
    train_ds.process_data()
    print(f"  src_vocab: {len(src_vocab):,} tokens")
    print(f"  tgt_vocab: {len(tgt_vocab):,} tokens")
    print(f"  train sentences: {len(train_ds):,}")

    print("Building validation dataset …")
    val_ds = Multi30kDataset(
        "validation", src_vocab=src_vocab, tgt_vocab=tgt_vocab, max_len=max_len
    )
    print(f"  val sentences: {len(val_ds):,}")

    print("Building test dataset …")
    test_ds = Multi30kDataset(
        "test", src_vocab=src_vocab, tgt_vocab=tgt_vocab, max_len=max_len
    )
    print(f"  test sentences: {len(test_ds):,}")

    _collate = partial(collate_fn, pad_idx=PAD_IDX)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_collate, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate, num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
