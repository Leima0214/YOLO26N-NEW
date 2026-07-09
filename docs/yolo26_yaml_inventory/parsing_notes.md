# Parsing Notes

**Method**: `yaml.safe_load()` on each file, then static structural inspection.
**Total files**: 46
**Parsed OK**: 46
**Parse failures**: 0

All YAMLs parsed successfully.

## Detection scale mapping method

P2-P6 detection scales are inferred from the number of detection layer inputs.
This is a heuristic — actual scale labels come from backbone Conv stride comments.
Manual verification recommended for any YAML with >3 or <3 detection layers.

## Module classification

Standard modules: ['C2PSA', 'C3k2', 'Classify', 'Concat', 'Conv', 'Detect', 'OBB', 'Pose', 'SPPF', 'Segment', 'Upsample', 'nn.Upsample']

All other modules are flagged as custom and may require additional Python files in `ultralytics/nn/`.

## Unconfirmed items

- Whether custom modules are actually importable at runtime
- Whether custom modules change FLOPs/params in ways that affect fair comparison
- Whether any YAML has hidden dependencies not captured by static parsing