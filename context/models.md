# Models — Vision Transformer (ViT)

---

## Vision Transformer (ViT)

### Architecture

```
Input (3×224×224)
  → Patch Embedding (16×16 patches → 196 tokens)
  → [CLS] Token + Positional Encoding
  → Transformer Encoder × 12 layers
      Each layer:
        LayerNorm → Multi-Head Self-Attention (12 heads) → Residual
        LayerNorm → MLP (hidden_dim=3072) → Residual
  → [CLS] Output → Linear Classification Head → 7 classes
```

### Key Properties

- **Base Model:** `vit_base_patch16_224` from `timm`
- **Pretrained On:** ImageNet-21k (then ImageNet-1k fine-tune)
- **Parameters:** ~86 Million
- **Input Size:** 224×224 RGB
- **Patch Size:** 16×16 → 196 patches per image
- **Embedding Dim:** 768
- **Attention Heads:** 12
- **Transformer Layers:** 12

### Why ViT for Gesture Recognition?

- Captures **global spatial relationships** between all finger positions simultaneously
- Local receptive fields miss cross-finger dependencies; ViT's MHSA does not
- ImageNet pretraining provides rich feature representations with minimal custom data
- Attention maps can visualize which hand regions drive each classification

---

## Training Strategy: Two-Phase Fine-Tuning

### Phase 1 — Head-Only Training (epochs 1–5)
- Backbone is **frozen** (no gradient updates)
- Only the classification head is trained
- Fast convergence, avoids catastrophic forgetting
- Learning Rate: `1e-4` with AdamW

### Phase 2 — Full Fine-Tuning (epochs 6–30)
- Entire model is **unfrozen**
- Fine-tuned with `LR × 0.1` (i.e., `1e-5`) to preserve pretrained features
- CosineAnnealingLR scheduler

---

## Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Input Size | 224×224 |
| Batch Size | 16 |
| Learning Rate (Phase 1) | 1e-4 |
| Learning Rate (Phase 2) | 1e-5 |
| Weight Decay | 1e-4 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Label Smoothing | 0.1 |
| Dropout | 0.1 |
| Max Epochs | 30 |
| Freeze Epochs | 5 |
| Early Stopping Patience | 5 |

---

## Implementation

```python
# models/vit_model.py
model = build_vit_model(
    model_name="vit_base_patch16_224",
    num_classes=7,
    pretrained=True,
    dropout=0.1,
)
```
