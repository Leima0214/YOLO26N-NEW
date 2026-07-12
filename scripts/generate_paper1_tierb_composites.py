"""Generate the Paper 1 Tier B three-module model YAMLs."""

from pathlib import Path

from generate_paper1_tiera_composites import ROOT, write_model


SPECS = [
    (
        "yolo26n-Paper1-TierB13-P2-SPDConv-LaplacianConv.yaml",
        {"p2": True, "detail": "spd", "head_detail": "lap"},
    ),
    (
        "yolo26n-Paper1-TierB14-P2-SPDConv-FDConv.yaml",
        {"p2": True, "detail": "spd", "head_detail": "fd"},
    ),
    (
        "yolo26n-Paper1-TierB15-P2-LaplacianConv-FDConv.yaml",
        {"p2": True, "detail": "lap", "head_detail": "fd"},
    ),
    (
        "yolo26n-Paper1-TierB16-P2-SPDConv-BiFPN.yaml",
        {"p2": True, "detail": "spd", "fusion": "bifpn"},
    ),
    (
        "yolo26n-Paper1-TierB17-P2-LaplacianConv-FFAFusion.yaml",
        {"p2": True, "detail": "lap", "fusion": "ffa"},
    ),
    (
        "yolo26n-Paper1-TierB18-P2-FDConv-FFAFusion.yaml",
        {"p2": True, "detail": "fd", "fusion": "ffa"},
    ),
    (
        "yolo26n-Paper1-TierB19-SPDConv-LaplacianConv-BiFPN.yaml",
        {"detail": "spd", "head_detail": "lap", "fusion": "bifpn"},
    ),
    (
        "yolo26n-Paper1-TierB20-SPDConv-FDConv-FFAFusion.yaml",
        {"detail": "spd", "head_detail": "fd", "fusion": "ffa"},
    ),
    (
        "yolo26n-Paper1-TierB21-P2-CARAFE-BiFPN.yaml",
        {"p2": True, "carafe": True, "fusion": "bifpn"},
    ),
    (
        "yolo26n-Paper1-TierB22-P2-CARAFE-FFAFusion.yaml",
        {"p2": True, "carafe": True, "fusion": "ffa"},
    ),
    (
        "yolo26n-Paper1-TierB23-SPDConv-CARAFE-BiFPN.yaml",
        {"detail": "spd", "carafe": True, "fusion": "bifpn"},
    ),
    (
        "yolo26n-Paper1-TierB24-LaplacianConv-CARAFE-FFAFusion.yaml",
        {"detail": "lap", "carafe": True, "fusion": "ffa"},
    ),
]


def main():
    if len(SPECS) != 12 or len({name for name, _ in SPECS}) != 12:
        raise ValueError("Tier B must define exactly 12 unique YAML names")
    for filename, options in SPECS:
        path = write_model(filename, options, Path(__file__).name)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
