# Experiments — Vision Transformer (ViT) Gesture Recognition

> Track ablation studies and configuration experiments here.

---

## Experiment Log Template

| # | Change | Val Acc | Test Acc | FPS | Notes |
|---|--------|---------|----------|-----|-------|
| 1 | Baseline ViT (head only) | | | | freeze_epochs=5 |
| 2 | + Full fine-tune (Phase 2) | | | | LR×0.1 |
| 3 | + Data augmentation | | | | flip, rotation, jitter |
| 4 | + Label smoothing (0.1) | | | | reduces overconfidence |
| 5 | + Early stopping (patience=5) | | | | prevents overfit |
| 6 | + CosineAnnealingLR | | | | smooth LR decay |
| 7 | Final config | | | | best combination |

---

## Standard Config (VIT_CONFIG in config.py)

| Hyperparameter | Value |
|----------------|-------|
| model_name | vit_base_patch16_224 |
| input_size | 224 |
| batch_size | 16 |
| learning_rate | 1e-4 |
| weight_decay | 1e-4 |
| epochs | 30 |
| freeze_epochs | 5 |
| dropout | 0.1 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR |
| label_smoothing | 0.1 |
| pretrained | True |

---

## Common Variables to Explore

| Variable | Options to Try |
|----------|---------------|
| freeze_epochs | 3, 5, 10 |
| batch_size | 8, 16, 32 |
| learning_rate | 5e-5, 1e-4, 3e-4 |
| dropout | 0.0, 0.1, 0.3 |
| epochs | 20, 30, 50 |

---

## How to Run an Experiment

```bash
# Override any hyperparameter from the command line
python training/train.py --epochs 20 --batch-size 8 --lr 5e-5
```

Results are saved to:
- `results/vit_metrics.json` — training metrics
- `results/vit_eval_results.json` — test evaluation metrics
- `results/training_curves/vit_training_curves.png` — loss/accuracy plots
