# DA6401 Assignment 3 — Transformer for Neural Machine Translation

Implementation of the Transformer architecture from ["Attention Is All You Need"](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (Vaswani et al., 2017) from scratch in PyTorch, trained on the Multi30k German→English translation dataset.

> **Test BLEU: 39.31** (Beam-8) · **37.85** (Greedy) · ~9M parameters

---

## Results

| Decode Strategy | Epochs | Test BLEU |
|---|---|---|
| Greedy | 20 | 37.85 |
| Beam-4 | 20 | 38.90 |
| **Beam-8** | **20** | **39.31** |

---

## Project Structure

```
da6401_assignment_3/
├── model.py          # Full Transformer architecture
├── lr_scheduler.py   # Noam learning rate scheduler
├── dataset.py        # Multi30k dataset + spaCy tokenization + Vocabulary
├── train.py          # Training loop, greedy decode, BLEU eval, checkpoints
└── requirements.txt  # Dependencies
```

---

## Architecture

Built entirely from `nn.Module`, `nn.Linear`, and `nn.LayerNorm` — no use of `torch.nn.MultiheadAttention`.

| Component | Details |
|---|---|
| Attention | Scaled dot-product + Multi-Head Attention (8 heads) |
| Positional encoding | Sinusoidal — registered as buffer, not trainable |
| Feed-forward | FFN(x) = max(0, xW₁+b₁)W₂+b₂ |
| Layer norm | Pre-LayerNorm placement |
| Loss | Label smoothing cross-entropy (ε = 0.1) |
| LR schedule | Noam (warmup = 4000 steps) |
| Inference | Greedy decoding + Beam search (beam = 4 or 8) |

**Model config:**

| d_model | heads | layers | d_ff | dropout | params |
|---|---|---|---|---|---|
| 256 | 8 | 3+3 | 512 | 0.1 | ~9M |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/MiRL-IITM/da6401_assignment_3
cd da6401_assignment_3
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

### 3. Train

```python
from train import run_training_experiment
run_training_experiment()
```

Or if running on Google Colab (recommended — see [Colab setup](#colab-setup)):

```python
# After uploading all files to /content/
from train import run_training_experiment
run_training_experiment()
```

Checkpoints are saved to `checkpoints/best_model.pt` (keyed on best validation BLEU).

### 4. Evaluate

```python
import torch
from model import Transformer
from train import evaluate_bleu, load_checkpoint
from dataset import get_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"
_, _, test_loader, src_vocab, tgt_vocab = get_dataloaders()

ckpt  = torch.load("checkpoints/best_model.pt", map_location=device)
model = Transformer(**ckpt["model_config"]).to(device)
model.load_state_dict(ckpt["model_state_dict"])

bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
print(f"Test BLEU: {bleu:.2f}")
```

---

## Colab Setup

> Recommended over local — ~4× faster on T4 GPU vs MacBook M4.

**Step 1** — Enable GPU: `Runtime → Change runtime type → T4 GPU`

**Step 2** — Install:
```python
!pip install torch torchvision datasets spacy sacrebleu wandb tqdm --quiet
!python -m spacy download de_core_news_sm --quiet
!python -m spacy download en_core_web_sm --quiet
```

**Step 3** — W&B login:
```python
import wandb
wandb.login()  # paste API key from wandb.ai/authorize
```

**Step 4** — Train:
```python
from train import run_training_experiment
run_training_experiment()
```

---

## Ablation Experiments

Five experiments were run to validate key design decisions from the paper.
Full analysis in the [W&B Report](https://wandb.ai/YOUR_USERNAME/da6401-a3/reports/YOUR_REPORT_LINK).

| Experiment | Condition A | Condition B | Key metric | Gap |
|---|---|---|---|---|
| 2.1 Noam scheduler | Noam warmup | Fixed LR=1e-4 | Val BLEU | **+3.08** |
| 2.2 Scaling 1/√dₖ | With scaling | Without scaling | Q grad norm @step 999 | **0.145 vs ~0** |
| 2.3 Head specialisation | — | — | 8 heads, 5 distinct roles | No severe redundancy |
| 2.4 Positional encoding | Sinusoidal | Learned PE | Test BLEU | **+0.13, zero params** |
| 2.5 Label smoothing | ε = 0.1 | ε = 0.0 | Test BLEU | **+0.99** |

---

## Key Implementation Notes

**Why Pre-LayerNorm?**
Pre-LN (normalise input before the sub-layer) provides O(1) gradient norm at initialisation vs O(N) for Post-LN. This makes training stable without careful tuning, which is why it is the default in GPT-2, T5, and most modern architectures. Post-LN is the original paper's choice but requires longer warmup and is more sensitive to learning rate.

**Mask convention**
`True` = masked out (set to −∞ before softmax). `make_src_mask` masks padding tokens. `make_tgt_mask` combines causal (upper-triangular) + padding masks.

**Noam scheduler**
Implemented as a subclass of `torch.optim.lr_scheduler.LRScheduler`. Stores `d_model` and `warmup_steps` before calling `super().__init__()` because the parent immediately calls `get_lr()`.

**Label smoothing**
Custom `nn.Module` — not `nn.CrossEntropyLoss` with label_smoothing kwarg. Spreads ε / (V−2) probability mass across all non-pad, non-correct tokens and assigns 0 to `<pad>`.

---

## Dependencies

```
torch
datasets
spacy
sacrebleu
wandb
tqdm
numpy
matplotlib
```

---

## Dataset

[Multi30k](https://huggingface.co/datasets/bentrevett/multi30k) — multilingual image caption dataset for NMT:

| Split | Sentences |
|---|---|
| Train | 29,000 |
| Validation | 1,014 |
| Test | 1,000 |

Tokenised using spaCy (`de_core_news_sm` for German, `en_core_web_sm` for English). Vocabulary built on training split only with minimum frequency threshold of 2.

---

## References

1. Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
2. Xiong et al. (2020). *On Layer Normalization in the Transformer Architecture.* ICML.
3. Clark et al. (2019). *What Does BERT Look At?* ACL BlackboxNLP.
4. Müller et al. (2019). *When Does Label Smoothing Help?* NeurIPS.

---

*Course: DA6401 · IIT Madras · Assignment 3*
