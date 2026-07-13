# Paper 1 Per-GT Assignment Diagnostic

- status: `PASS`
- split/images/GT: `val` / `2119` / `5000`
- weights SHA256: `8e811affeaca8bf4b4173f42405b6ffa431c54a6cb0bd9ef0ff29fe476e8692b`
- assignment overlap metric: `CIoU clamped to [0, 1], matching TaskAlignedAssigner`
- thresholds: IoU `0.3`, class score `0.05`

## one2many

| group | GT | zero positive | mean positives | median IoU | median class score |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 5000 | 0.0004 | 9.845 | 0.7742 | 0.0028 |
| D00 | 811 | 0.0012 | 9.873 | 0.7375 | 0.0023 |
| D44 | 877 | 0.0011 | 9.920 | 0.8642 | 0.0124 |
| D50 | 728 | 0.0000 | 9.449 | 0.8213 | 0.0032 |
| D20 | 1234 | 0.0000 | 9.955 | 0.7982 | 0.0032 |
| D40 | 425 | 0.0000 | 9.762 | 0.7608 | 0.0021 |
| D10 | 787 | 0.0000 | 9.956 | 0.5099 | 0.0019 |
| D43 | 138 | 0.0000 | 9.935 | 0.8030 | 0.0082 |
| AR:lt2 | 2666 | 0.0008 | 9.892 | 0.8073 | 0.0033 |
| AR:2to3 | 896 | 0.0000 | 9.769 | 0.7935 | 0.0027 |
| AR:3to5 | 906 | 0.0000 | 9.737 | 0.7150 | 0.0023 |
| AR:5to8 | 395 | 0.0000 | 9.904 | 0.5364 | 0.0021 |
| AR:ge8 | 137 | 0.0000 | 9.971 | 0.4081 | 0.0022 |

| overall outcome | rate |
| --- | ---: |
| both_adequate | 0.0282 |
| both_low | 0.0228 |
| iou_adequate_score_low | 0.9490 |
| iou_low_score_adequate | 0.0000 |

## one2one

| group | GT | zero positive | mean positives | median IoU | median class score |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 5000 | 0.0002 | 1.000 | 0.7659 | 0.0023 |
| D00 | 811 | 0.0012 | 0.999 | 0.7317 | 0.0022 |
| D44 | 877 | 0.0000 | 1.000 | 0.8606 | 0.0052 |
| D50 | 728 | 0.0000 | 1.000 | 0.8156 | 0.0016 |
| D20 | 1234 | 0.0000 | 1.000 | 0.7930 | 0.0025 |
| D40 | 425 | 0.0000 | 1.000 | 0.7407 | 0.0021 |
| D10 | 787 | 0.0000 | 1.000 | 0.4891 | 0.0018 |
| D43 | 138 | 0.0000 | 1.000 | 0.8061 | 0.0049 |
| AR:lt2 | 2666 | 0.0004 | 1.000 | 0.8024 | 0.0026 |
| AR:2to3 | 896 | 0.0000 | 1.000 | 0.7815 | 0.0022 |
| AR:3to5 | 906 | 0.0000 | 1.000 | 0.7110 | 0.0018 |
| AR:5to8 | 395 | 0.0000 | 1.000 | 0.5005 | 0.0019 |
| AR:ge8 | 137 | 0.0000 | 1.000 | 0.3860 | 0.0021 |

| overall outcome | rate |
| --- | ---: |
| both_adequate | 0.0032 |
| both_low | 0.0200 |
| iou_adequate_score_low | 0.9768 |
| iou_low_score_adequate | 0.0000 |

Use one2many as the primary early-training assignment diagnostic; one2one is a secondary E2E check.
