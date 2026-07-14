# Aijiaojioashou Dataset

## Overview

Aijiaojioashou is a 300-image YOLO detection dataset rebuilt from the public Microsoft COCO 2017 detection dataset for small-scale YOLO26n research experiments. The split is 6:2:2:

| split | images | boxes |
|---|---:|---:|
| train2017 | 180 | 180 |
| val2017 | 60 | 64 |
| test2017 | 60 | 88 |

The active research targets are three categories: car, dog, and chair. To preserve transfer from the pretrained COCO YOLO26n detection head, labels keep the original COCO80 class ids:

| active class | YOLO/COCO80 id | active id | COCO category id | Chinese |
|---|---:|---:|---:|---|
| car | 2 | 0 | 3 | automobile |
| dog | 16 | 1 | 18 | dog |
| chair | 56 | 2 | 62 | chair |

No other class ids are used in the label files.

## Source And License

The images are derived from official COCO 2017 detection train/val images and annotations:

- COCO website: https://cocodataset.org/
- COCO 2017 detection: https://cocodataset.org/#detection-2017
- COCO annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip

Images with NonCommercial or NoDerivs license ids were excluded. The retained source licenses are CC BY 2.0, CC BY-SA 2.0, Flickr Commons no known copyright restrictions, and United States Government Work. Per-image provenance, original COCO id, source URL, license id, license name, and license URL are listed in `metadata/records.csv`.

## Target Distribution

| class | boxes | small | medium | large |
|---|---:|---:|---:|---:|
| car | 114 | 10 | 14 | 90 |
| dog | 107 | 10 | 18 | 79 |
| chair | 111 | 11 | 19 | 81 |

Scale follows the COCO convention on the 640 x 640 output image: small is area < 32^2 px, medium is 32^2 to 96^2 px, and large is > 96^2 px.

## Image Conditions

All images are RGB JPEG files resized or cropped to 640 x 640. The dataset includes clear objects plus harder imaging cases:

| condition | count |
|---|---:|
| low light | 68 |
| shadow / high contrast | 36 |
| harsh light | 17 |
| edge or occlusion-like boxes | 2 |

The figures in `figures/` summarize split, class, scale, complex-condition distribution, and sample annotations using a clean journal-style layout.

## YOLO26n 100-Epoch Trial

Training script: `train_yolo26n_yaml_100e.py`

Model config: `yolo26n.yaml`

Pretrained checkpoint: `D:\Users\D\Desktop\ultralytics26-main5.26\yolo26n.pt`

Final 100-epoch validation result from `last.pt`:

| metric | value |
|---|---:|
| precision | 0.777489 |
| recall | 0.730048 |
| mAP50 | 0.749602 |
| mAP50-95 | 0.604754 |

The required metrics, precision, recall, and mAP50, are all within 70%-80%. The exact result payload is stored in `metadata/yolo26n_100e_metrics.json`.
