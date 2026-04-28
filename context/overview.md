# Project Overview — Real-Time Cursor Control Using Hand Gestures with Vision Transformer

---

## What This Project Does

This is a **Final Year Project** that builds a real-time, webcam-based cursor control system
using hand gestures. A **Vision Transformer (ViT)** classifies hand gestures, which are then
mapped to cursor actions (move, click, scroll, drag) via PyAutoGUI.

---

## Project Goals

1. Build a real-time hand gesture recognition system using webcam input.
2. Train a **Vision Transformer (ViT)** classifier on custom gesture data.
3. Evaluate performance: Accuracy, F1-Score, FPS, Latency.
4. Deploy a working gesture-controlled cursor system.

---

## System Pipeline

```
Webcam → OpenCV Frame Capture
    → MediaPipe Hand Detection
    → ROI Extraction & Preprocessing (224×224, ImageNet normalization)
    → Vision Transformer Inference
    → Gesture Classification (7 classes)
    → Action Mapping (move / click / scroll / drag)
    → Cursor Control (PyAutoGUI)
```

---

## Key Outputs

| Output | Description |
|--------|-------------|
| Trained ViT Model | `checkpoints/best_vit_model.pth` |
| Evaluation Metrics | Accuracy, Precision, Recall, F1, FPS, Latency |
| Confusion Matrix | `results/confusion_matrices/vit_confusion_matrix.png` |
| Training Curves | `results/training_curves/vit_training_curves.png` |
| Research Report | `results/research_report.md` |

---

## Why Vision Transformer?

- **Global attention:** MHSA captures relationships between all finger positions simultaneously
- **Pretrained on ImageNet-21k:** Rich visual features out-of-the-box
- **Interpretability:** Attention maps reveal which image regions drive predictions
- **State-of-the-art accuracy** on vision benchmarks

---

## Gesture → Action Mapping

| Gesture | Cursor Action |
|---------|--------------|
| Open Palm | Move cursor |
| Index Point | Left click |
| Two Fingers Up | Right click |
| Fist | Neutral / Stop |
| Pinch | Drag |
| Three Fingers Up | Scroll up |
| Three Fingers Down | Scroll down |
