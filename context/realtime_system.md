# Real-Time System — ViT Gesture Control

---

## System Architecture

```
Webcam
  │
  ▼
OpenCV Frame Capture (640×480)
  │
  ▼
MediaPipe Hands — Hand Detection & Landmark Extraction
  │  (returns bounding box + 21 landmarks + index fingertip)
  ▼
ROI Extraction (hand bounding box + 20px padding)
  │
  ▼
Preprocessing: BGR→RGB → Resize(224×224) → ToTensor → Normalize(ImageNet)
  │
  ▼
Vision Transformer (ViT) Inference
  │  (returns class probabilities over 7 gesture classes)
  ▼
Gesture Debouncer — requires N consecutive confident frames
  │
  ▼
Action Mapper — gesture → cursor action
  │
  ▼
CursorSmoother (moving average of index fingertip position)
  │
  ▼
PyAutoGUI — move cursor / click / scroll / drag
```

---

## Component Details

### HandDetector (`realtime/hand_detector.py`)
- Wraps MediaPipe Hands
- Returns bounding box, 21 landmarks, and index fingertip position
- Extracts the hand ROI with configurable padding

### GestureDebouncer (`realtime/gesture_smoother.py`)
- Requires `debounce_frames=2` consecutive frames with the same gesture before triggering
- Enforces `click_cooldown_ms=200` to prevent rapid repeated clicks
- Filters predictions below `confidence_threshold=0.6`

### CursorSmoother (`realtime/gesture_smoother.py`)
- Moving average over last N index fingertip positions (`smoothing_window=5`)
- Reduces jitter from natural hand tremor
- Resets on hand-not-detected frames

### CursorController (`realtime/cursor_controller.py`)
- Maps normalized hand position → screen coordinates
- Executes: move, left_click, right_click, drag, scroll_up, scroll_down

---

## Real-Time Config (`config.py → REALTIME`)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| camera_id | 0 | Webcam index |
| camera_width | 640 | Frame width |
| camera_height | 480 | Frame height |
| smoothing_window | 5 | Moving average window for cursor |
| debounce_frames | 2 | Frames required to trigger an action |
| click_cooldown_ms | 200 | Min time between clicks |
| scroll_amount | 30 | Scroll units per trigger |
| confidence_threshold | 0.6 | Min prediction confidence |

---

## Performance Requirements

| Metric | Target |
|--------|--------|
| FPS | ≥ 15 FPS |
| End-to-end latency | < 100 ms per frame |

For CPU-only deployment, use ONNX export or INT8 quantization to meet the FPS target.

---

## Run Commands

```bash
# Standard real-time mode
python realtime/gesture_control.py

# Debug mode (no cursor movement)
python realtime/gesture_control.py --no-cursor

# Custom checkpoint
python realtime/gesture_control.py --checkpoint checkpoints/best_vit_model.pth

# Silent mode (no visualization)
python realtime/gesture_control.py --no-debug
```

---

## Emergency Stop

Move the mouse cursor to the **top-left corner** of the screen — PyAutoGUI's
built-in fail-safe will raise an exception and stop the system cleanly.
