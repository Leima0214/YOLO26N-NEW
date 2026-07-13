# D10 Annotation Audit

## Scope

On 2026-07-13, `scripts/sample_yolo_class_annotations.py` sampled 100 D10 boxes from each Japan7 split with seed 42. The exact sample identities are preserved in:

- `experiments/dataset_diagnostics/japan7/d10_annotation_audit_seed42_20260713/train_review_manifest.csv`
- `experiments/dataset_diagnostics/japan7/d10_annotation_audit_seed42_20260713/val_review_manifest.csv`

The remote review directories were:

- `runs/annotation_audit/paper1_d10_train_seed42_20260713_164737`
- `runs/annotation_audit/paper1_d10_val_seed42_20260713_164737`

Rendered crops and full-frame overlays are review aids under `runs`; they are not committed.

## Qualitative First Pass

The sampled train and validation boxes generally describe the same transverse-crack visual concept, with no obvious split-level class-definition mismatch. Recurring difficulty factors include weak contrast, road paint, shadows, blur, background texture, and variable split/merge scale for extended cracks. These factors can lower classification confidence even when localization is adequate.

This was an AI-assisted qualitative screen, not a blinded multi-reviewer annotation study. The manifest review columns remain available for a human adjudication pass. No annotation-error percentage should be reported in Paper 1 until those columns are completed under a written review rubric.

## Experimental Decision

The audit does not justify changing labels or stopping the current Japan7 protocol. Combined with the B0 assignment diagnostic, it rejects class-specific or unconditional upweighting: ambiguous positives must not all receive the same boost. B2 therefore uses a class-agnostic, bounded weight only when an assigned positive is both reasonably localized and low in correct-class confidence.

No source or derived dataset files were changed by this audit.
