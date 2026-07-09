# Dataset Protocols

## Raw data: 10-class labeling

The original dataset uses 10 class IDs across 4 countries (China_MotorBike, Czech, India, Japan):

| ID | Name | China | Czech | India | Japan |
| --: | ---- | ---: | ---: | ---: | ---: |
| 0 | D00 | 2,678 | 988 | 1,556 | 4,049 |
| 1 | D01 | 0 | 0 | 179 | 0 |
| 2 | D10 | 1,096 | 399 | 68 | 3,979 |
| 3 | D11 | 0 | 0 | 45 | 0 |
| 4 | D20 | 641 | 161 | 2,021 | 6,199 |
| 5 | D40 | 235 | 197 | 3,187 | 2,243 |
| 6 | D43 | 0 | 0 | 57 | 736 |
| 7 | D44 | 0 | 0 | 1,062 | 3,995 |
| 8 | D50 | 0 | 0 | 28 | 3,553 |
| 9 | Repair | 277 | 0 | 0 | 0 |

Key observations:

- **D01/D11/Repair** are extreme long-tail: 179 / 45 / 277 boxes respectively, each only in one country.
- **D43/D44/D50** exist only in Japan (and slightly in India).
- **Czech** has only 4 classes: D00, D10, D20, D40.
- Simple 10-class cross-domain training would produce meaningless mAP scores because most classes are missing in target domains.

## Japan7 protocol (Paper 1)

7 classes active in Japan domain. D01/D11/Repair are dropped.

| Source ID | Source Name | → | Target ID | Target Name |
| ---: | --- | --- | ---: | --- |
| 0 | D00 | → | 0 | D00 |
| 2 | D10 | → | 1 | D10 |
| 4 | D20 | → | 2 | D20 |
| 5 | D40 | → | 3 | D40 |
| 6 | D43 | → | 4 | D43 |
| 7 | D44 | → | 5 | D44 |
| 8 | D50 | → | 6 | D50 |

Config: `configs/mappings/japan7.yaml`

## Common4 protocol (Paper 2)

4 classes present in **all four** countries. D01/D11/D43/D44/D50/Repair are dropped.

| Source ID | Source Name | → | Target ID | Target Name |
| ---: | --- | --- | ---: | --- |
| 0 | D00 | → | 0 | D00 |
| 2 | D10 | → | 1 | D10 |
| 4 | D20 | → | 2 | D20 |
| 5 | D40 | → | 3 | D40 |

Config: `configs/mappings/common4.yaml`

## Paper-to-protocol mapping

| Paper | Protocol | nc | Configs |
| --- | --- | ---: | --- |
| Paper 1 | Japan7 | 7 | `configs/japan7_local.yaml` / `japan7_remote.yaml` |
| Paper 2 | Common4 | 4 | `configs/common4_{country}_*.yaml` |
