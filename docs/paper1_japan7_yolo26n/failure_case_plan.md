# Failure Case Analysis Plan

## Goal

Collect and categorize detection failures to:
1. Validate the problem motivation (why D00/D10 are hard)
2. Provide visual evidence for the paper (before/after improvement)
3. Guide ablation design (which failure modes to target)

## Failure categories

| Category | Description | Expected % of failures |
| --- | --- | --- |
| F1: Thin crack miss | Crack <5px wide, missed entirely | 30–40% |
| F2: Low contrast miss | Crack blends into asphalt, confidence < threshold | 20–30% |
| F3: Background confusion | Road marking, shadow, patch → false positive or miss | 10–15% |
| F4: Scale mismatch | Crack too small/large for anchor scales | 10–15% |
| F5: Localization error | Detected but IoU < 0.5 (box too large/small/shifted) | 10–15% |
| F6: Class confusion | Correct box, wrong damage class | 5–10% |

## Collection protocol

For each failure category, collect 3–5 representative examples from the YOLO26n val set:

1. Run inference on val set:
   ```python
   from ultralytics import YOLO
   model = YOLO("runs/baseline/yolo26n_japan7_e100_img640_b32_seed42/weights/best.pt")
   results = model.val(data="configs/japan7_remote.yaml", save_json=True, conf=0.25, iou=0.6)
   ```

2. Save prediction JSON

3. Script to find failure examples:
   - FN (false negative): GT box with no matching prediction
   - FP (false positive): Prediction with no matching GT
   - Low IoU: GT-prediction pairs with IoU < 0.5
   - Class confusion: GT-prediction pairs with different class labels

4. Manually review and categorize

## Paper figures (Figure 5)

3×3 grid with 3 rows (D00, D10, D20) and 3 columns:

| | Ground Truth | YOLO26n | Ours-YOLO26n |
| --- | --- | --- | --- |
| D00 | GT boxes | Missed (red circle) | Detected (green box) |
| D10 | GT boxes | Missed / partial | Detected |
| D20 | GT boxes | Detected (baseline) | Detected (similar) |

## Analysis script

Create `scripts/analyze_failures.py`:
- Input: val predictions JSON + GT labels
- Output: failure_statistics.csv + sample image list
