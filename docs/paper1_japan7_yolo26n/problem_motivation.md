# Problem Motivation

## The crack detection challenge

Road crack detection differs from general object detection in several ways:

| Property | General objects | Road cracks |
| --- | --- | --- |
| Aspect ratio | ~1:1 to 3:1 | Up to 20:1 (thin cracks) |
| Contrast | High (object vs background) | Low (crack vs asphalt) |
| Scale variation | Moderate | Extreme (hairline vs pothole) |
| Texture | Distinct object texture | Blends into road surface |
| Occlusion | Rare | Common (shadows, markings) |

## Why YOLO26n?

YOLO26n is the lightest YOLO26 variant (2.376M params, 5.2G FLOPs). It is suitable for:

- Edge deployment (drones, road inspection vehicles)
- Real-time processing
- Resource-constrained environments

However, lightweight design sacrifices shallow detail representation, which is critical for thin crack detection.

## Evidence from baseline

| Evidence | Implication |
| --- | --- |
| YOLO26n D00 mAP50-95 = 0.183 | Thin pothole cracks are missed |
| YOLO26n D10 mAP50-95 = 0.148 | Longitudinal cracks severely under-detected |
| YOLO26s gains only +0.006 | Pure capacity scaling doesn't help |
| D20/D43/D50 are fine (0.346–0.541) | Large/obvious damage is well-detected |

## Research gap

Existing YOLO improvements (attention mechanisms, neck redesigns) primarily target general object detection. Few works address the specific challenge of **thin, low-contrast crack detection on lightweight backbones**.

## Proposed direction

Enhance YOLO26n's shallow feature representation through lightweight detail-preserving modules, without excessive parameter/FLOPs increase.

Candidate approaches:
1. P2 / shallow detail branch (high-resolution feature preservation)
2. Lightweight attention (channel/spatial, targeted at fine features)
3. Wavelet/frequency enhancement (decompose and enhance high-frequency crack details)
4. Multi-scale feature fusion (better integration of shallow + deep features)
