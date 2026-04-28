# Results — Vision Transformer (ViT) Evaluation

> Fill in actual numbers after running `python training/evaluate.py`

---

## ViT Performance

| Metric | Value |
|--------|-------|
| Test Accuracy (%) | __ |
| Precision (macro %) | __ |
| Recall (macro %) | __ |
| F1-Score (macro %) | __ |
| Avg Latency (ms) | __ |
| FPS | __ |
| Model Size (MB) | ~327 |
| Parameters | ~86M |

---

## Per-Class F1-Score

| Gesture Class | ViT F1 (%) |
|---------------|-----------|
| open_palm | __ |
| index_point | __ |
| two_fingers_up | __ |
| fist | __ |
| pinch | __ |
| three_fingers_up | __ |
| three_fingers_down | __ |

---

## Confusion Matrix

Saved to: `results/confusion_matrices/vit_confusion_matrix.png`

- Rows = Actual class
- Columns = Predicted class
- Diagonal = Correct predictions

---

## Training Curves

Saved to: `results/training_curves/vit_training_curves.png`

- Left plot: Train vs Validation Loss
- Right plot: Train vs Validation Accuracy
- Green dashed line: Best epoch (lowest validation loss)

---

## Real-Time Feasibility

| Threshold | Requirement |
|-----------|------------|
| Minimum FPS for real-time cursor control | ≥ 15 FPS |
| Target latency | < 100 ms per frame |

For CPU-only deployment, use ONNX or INT8 quantization (see `optimization/`).

---

## Observations

*(Fill in after evaluation)*

- Which gesture class has the highest/lowest F1?
- Does the model confuse similar gestures (e.g., index_point vs two_fingers_up)?
- Does real-time FPS meet the ≥15 FPS threshold?
