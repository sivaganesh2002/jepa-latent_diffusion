# jepa-diffusion-lm

**A self-supervised text generation framework combining Joint Embedding Predictive Architecture (JEPA) with a conditional diffusion decoder for abstractive summarization.**

This project explores an alternative to autoregressive language modeling: instead of next-token prediction, a JEPA predictor learns document-to-summary alignment entirely in latent embedding space, and a transformer-based diffusion decoder reconstructs token sequences from those latent targets. No cross-entropy over vocabulary. No teacher forcing.

---

## Motivation

Standard seq2seq and autoregressive LMs are trained with exposure bias and operate in a high-dimensional discrete token space. JEPA-style objectives, as proposed by LeCun (2022), instead predict representations in a continuous latent space — avoiding the degeneracies of pixel/token-level prediction. This work applies that principle to text:

- **Stage 1** learns a structured embedding space where document representations are predictive of summary representations, via InfoNCE contrastive learning.
- **Stage 2** learns to invert that embedding back to token sequences via a score-based diffusion process conditioned on JEPA latents.

This separates *semantic alignment* (Stage 1) from *surface realization* (Stage 2), making each component independently analyzable — a property useful for interpretability and modular scaling research.

---

## Architecture

### Stage 1 — JEPA Predictor

```
Document ──► [MiniLM encoder, frozen]
               │
             [Linear + LayerNorm]          input projection
               │
             [8 × TransformerEncoderLayer] pre-norm, 12-head, GELU
             (d_model=768, ff_dim=3072)
               │
             [Linear → GELU → Linear]      output projection
               │
             L2-normalize ──► ŝ ∈ ℝ¹⁰²⁴

Summary ──► [MiniLM encoder, trainable]
              │
            [Linear → GELU → LayerNorm → Linear]
              │
            L2-normalize ──► s ∈ ℝ¹⁰²⁴  ◄── EMA shadow target
```

The base encoder (`all-MiniLM-L6-v2`, `d=384`) is frozen in the predictor and only the stacked transformer layers are optimized. This follows the standard JEPA design: a powerful but fixed feature extractor feeds a learnable predictor head.

An **EMA (Exponential Moving Average)** copy of the YEncoder provides slowly-drifting, stable targets — analogous to the momentum encoder in MoCo and the target network in I-JEPA.

EMA decay ramps from `0.999 → 0.9999` over the first 10k steps:

```python
decay(step) = decay_min + (decay_max - decay_min) * min(step / 10000, 1.0)
```

Early in training this allows fast adaptation; later it stabilizes the target distribution.

---

### Stage 2 — Conditional Diffusion Decoder

```
hs = YEncoder(summary)    ∈ ℝ¹⁰²⁴        frozen condition

x₀ = WordEmb(tokenize(summary))  ∈ ℝ^{32 × 384}

Forward:  xₜ = √ᾱₜ · x₀ + √(1−ᾱₜ) · ε,   ε ~ N(0, I)

DiffusionDecoder(xₜ, t, hs):
  ┌─────────────────────────────────────────┐
  │  t ──► SinusoidalEmb ──► 3-layer MLP   │  timestep embedding
  │                                          │
  │  hs ──► Linear ──► reshape              │  condition as sequence
  │         ℝ¹⁰²⁴ → ℝ^{8 × 512}           │  (8 context tokens)
  │                                          │
  │  xₜ ──► Linear + SinPE                 │  noisy input
  │                                          │
  │  10 × DiffDecoderLayer:                 │
  │    PreNorm → SelfAttn                   │
  │    PreNorm → CrossAttn(memory)          │
  │    FiLM(t_emb) → FFN                   │
  └─────────────────────────────────────────┘
           │
           ▼
           ε̂ ∈ ℝ^{32 × 384}     predicted noise
```

The condition vector `hs` is **expanded into a sequence of 8 tokens** before cross-attention rather than used as a single pooled vector. This gives the decoder spatially-distributed context analogous to how encoder-decoder attention operates in T5/BART.

---

## Mathematical Formulation

### InfoNCE with Label Smoothing

Given a batch of B (document, summary) pairs, embeddings `{ŝᵢ}` and `{sᵢ}` are L2-normalized. The contrastive objective is:

$$\mathcal{L}_{\text{JEPA}} = - \frac{1}{B} \sum_{i=1}^{B} \sum_{j=1}^{B} \tilde{y}_{ij} \log \frac{\exp(\hat{s}_i \cdot s_j^\top / \tau)}{\sum_{k=1}^{B} \exp(\hat{s}_i \cdot s_k^\top / \tau)}$$

where the smoothed targets are:

$$\tilde{y}_{ij} = \begin{cases} 1 - \varepsilon & \text{if } i = j \\ \varepsilon / (B-1) & \text{if } i \neq j \end{cases}$$

with temperature `τ = 0.07` and smoothing `ε = 0.05`. Label smoothing prevents the model from driving logits to ±∞ on hard negatives early in training.

---

### Cosine Noise Schedule

Following Nichol & Dhariwal (2021), the variance schedule uses a cosine function with offset `s = 0.008`:

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \qquad f(t) = \cos^2\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$$

$$\beta_t = \text{clip}\!\left(1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}},\ 0,\ 0.999\right), \qquad \alpha_t = 1 - \beta_t$$

The offset `s` prevents `ᾱₜ` from approaching zero near `t=0`, which caused gradient explosions under a linear schedule. Compared to linear, cosine degrades the signal-to-noise ratio more gradually, giving the model more training signal at high noise levels.

---

### Forward Process

$$q(x_t \mid x_0) = \mathcal{N}\!\left(\sqrt{\bar{\alpha}_t}\, x_0,\ (1 - \bar{\alpha}_t)\, I\right)$$

Reparameterized as:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

All `x₀` are L2-normalized before noising. This bounds the input distribution to the unit hypersphere, giving stable MSE targets throughout training.

---

### Reverse Process & Training Objective

The decoder is trained to predict `ε` from `(xₜ, t, hs)`. The loss combines noise prediction and direct reconstruction:

$$\mathcal{L}_{\text{diff}} = \underbrace{\mathbb{E}_{t, \varepsilon}\left[\|\hat{\varepsilon} - \varepsilon\|^2\right]}_{\text{noise loss}} + 0.5 \cdot \underbrace{\mathbb{E}_{t, \varepsilon}\left[\|\hat{x}_0 - x_0\|^2\right]}_{\text{reconstruction loss}}$$

where the x₀ estimate is recovered analytically:

$$\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\, \hat{\varepsilon}}{\sqrt{\bar{\alpha}_t}}$$

The reconstruction term at half-weight provides additional gradient signal for learning the clean embedding geometry, complementing the noise-prediction objective.

---

### FiLM Conditioning

Each decoder layer applies Feature-wise Linear Modulation (Perez et al., 2018) to inject the timestep:

$$\text{FiLM}(x, t_\text{emb}) = x \odot (1 + W_s\, t_\text{emb}) + W_b\, t_\text{emb}$$

where `W_s, W_b ∈ ℝ^{d_model}` are learned per-layer. This is applied *after* the cross-attention sublayer and *before* the FFN. Compared to adding `t_emb` once at the input, FiLM allows each layer to independently modulate feature scales based on the current noise level — more expressive for a 10-layer network operating over T=1000 timesteps.

---

### Inference — DDIM Sampling

At inference, the predictor replaces the YEncoder as the condition source:

```
hs = TextPredictor(document)
```

DDIM (Song et al., 2021) is used for deterministic sampling with 250 steps (4× speedup over 1000-step DDPM):

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \hat{\varepsilon}_\theta(x_t, t, h_s)\right) + \sqrt{\text{posterior\_var}_t} \cdot z$$

Final token decoding via nearest-neighbor cosine similarity over the frozen vocabulary embedding matrix `W_e ∈ ℝ^{V × 384}`:

$$\hat{w}_i = \arg\max_{v \in \mathcal{V}}\ \frac{x_0^{(i)} \cdot W_e^{(v)}}{\|x_0^{(i)}\| \|W_e^{(v)}\|}$$

---

## Training

### Two-Stage Pipeline

```
Stage 1 — JEPA (8 epochs, XSum train split, 10k samples)
  Optimizer : AdamW, lr=3e-5, wd=1e-2
  Schedule  : linear warmup (10% steps) → cosine decay to 5% peak
  Loss      : InfoNCE + label smoothing (τ=0.07, ε=0.05)
  Precision : fp16 AMP, grad clip 1.0

Stage 2 — Diffusion (20 epochs, same split)
  Optimizer : AdamW, lr=2e-4, wd=1e-2
  Schedule  : linear warmup (10% steps) → cosine decay
  Loss      : MSE(noise) + 0.5 × MSE(recon)
  Precision : fp16 AMP, grad clip 1.0, grad accum ×2
  EMA       : decoder copy, decay 0.999 → 0.9999
```

### Key Stability Decisions

| Issue | Solution |
|---|---|
| EMA target collapse | Ramp decay from 0.999 → 0.9999 over 10k steps |
| Loss explosion at low t | Cosine schedule with s=0.008 offset |
| Overconfident InfoNCE | Label smoothing ε=0.05 |
| Gradient instability | Pre-norm transformer layers throughout |
| Base encoder interference | Freeze MiniLM in predictor; only train stacked heads |
| Embedding scale drift | L2-normalize all embeddings before loss computation |

---

## Evaluation

After Stage 1, JEPA quality is measured by cosine similarity between predicted and ground-truth embeddings on the test set (n=50):

```
mean cosine sim : ~0.65–0.75  (good alignment)
std             : ~0.08
```

Loss curves (train vs val) and gradient norms are plotted for both stages and saved to `loss_curves.png` and `embedding_sim.png`.

---

## Configuration

```python
# model dimensions
BASE_DIM   = 384    # MiniLM hidden size
PRED_DIM   = 768    # predictor transformer width
EMBED_DIM  = 1024   # JEPA latent dimension
DIFF_DIM   = 512    # diffusion decoder width
OUT_LEN    = 32     # output tokens

# diffusion
T          = 1000   # total timesteps
DDIM_STEPS = 250    # inference steps

# training
JEPA_LR    = 3e-5
DIFF_LR    = 2e-4
JEPA_EPOCHS = 8
DIFF_EPOCHS = 20
BATCH_SIZE  = 16
GRAD_ACCUM  = 2     # effective batch = 32
TEMPERATURE = 0.07
LABEL_SMOOTH = 0.05
EMA_DECAY_MIN = 0.999
EMA_DECAY_MAX = 0.9999
```

---

## Project Structure

```
jepa-diffusion-lm/
├── jepa_text_v4.ipynb     # full implementation
├── README.md
├── loss_curves.png        # training diagnostics (auto-generated)
└── embedding_sim.png      # JEPA retrieval quality (auto-generated)
```

---

## Dependencies

```
torch >= 2.0
transformers >= 4.30
datasets >= 2.12
sentencepiece
numpy
matplotlib
```

```bash
pip install torch transformers datasets sentencepiece matplotlib
```

GPU required. fp16 AMP activates automatically on CUDA. Full pipeline trains in ~2–3h on an A100.

---

## References

```
LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence.
  openreview.net/forum?id=BZ5a1r-kVsf

Assran et al. (2023). Self-Supervised Learning from Images with a
  Joint-Embedding Predictive Architecture. CVPR 2023.

Ho et al. (2020). Denoising Diffusion Probabilistic Models. NeurIPS 2020.

Nichol & Dhariwal (2021). Improved Denoising Diffusion Probabilistic
  Models. ICML 2021.

Song et al. (2021). Denoising Diffusion Implicit Models. ICLR 2021.

Perez et al. (2018). FiLM: Visual Reasoning with a General
  Conditioning Layer. AAAI 2018.

Narayan et al. (2018). Don't Give Me the Details, Just the Summary!
  Topic-Aware Convolutional Neural Networks for Extreme Summarization.
  EMNLP 2018. [XSum dataset]
```
