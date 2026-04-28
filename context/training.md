# Training — Vision Transformer (ViT)

---

## Overview

The ViT model is trained on a custom hand gesture dataset using a two-phase
fine-tuning strategy. Phase 1 trains only the classification head while the
ImageNet-pretrained backbone is frozen. Phase 2 unfreezes the entire model for
full fine-tuning at a lower learning rate.

---

## Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Input Size | 224×224 |
| Batch Size | 16 |
| Learning Rate (Phase 1) | 1e-4 |
| Learning Rate (Phase 2) | 1e-5 (1/10th of Phase 1) |
| Weight Decay | 1e-4 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Label Smoothing | 0.1 |
| Dropout | 0.1 |
| Max Epochs | 30 |
| Freeze Epochs | 5 |
| Early Stopping Patience | 5 |

---

## Optimizer: AdamW

AdamW (Adam with decoupled weight decay) is the standard optimizer for transformer
fine-tuning. It prevents the weight decay from interfering with adaptive gradient
scaling, leading to better generalization.

---

## Scheduler: CosineAnnealingLR

```python
# Phase 1
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# Phase 2 (reset after unfreeze)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs - freeze_epochs)
```

Smoothly decays the learning rate from `lr` to near-zero, avoiding sharp drops
that can destabilize fine-tuning.

---

## Two-Phase Fine-Tuning

```python
# Phase 1: train head only
model.freeze_backbone()
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Phase 2: unfreeze at epoch 6
model.unfreeze_backbone()
optimizer = AdamW(model.parameters(), lr=1e-5)
```

### Why Two Phases?

| Reason | Explanation |
|--------|-------------|
| Prevents catastrophic forgetting | Backbone preserves ImageNet features during Phase 1 |
| Faster convergence | Head converges quickly before full model updates |
| Better final accuracy | Lower LR in Phase 2 fine-tunes without overwriting pretrained features |

---

## Training Script

```bash
python training/train.py --epochs 30
python training/train.py --epochs 20 --batch-size 8 --lr 5e-5
```

---

## Data Augmentation

| Augmentation | Parameter |
|--------------|-----------|
| Random resized crop | scale=(0.85, 1.0) |
| Random horizontal flip | p=0.5 |
| Random rotation | ±15° |
| Color jitter | brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1 |
| Gaussian blur | kernel=3, p=0.2 |
| Normalize | ImageNet mean/std |

---

## Expected Training Time

| Hardware | Approx. Time |
|----------|-------------|
| NVIDIA GPU (CUDA) | 15–30 minutes |
| CPU only | 1–2 hours |
