"""Run P4 Single on the scene-disjoint Japan7-v2 seed42 split."""

from train_japan7_100e_common import run


if __name__ == "__main__":
    run(
        "ultralytics/cfg/models/26/yolo26n-OverLoCK-ProjectFit-P4-Gate1e3.yaml",
        "p4single_japan7_v2_scene_disjoint_100e_seed42",
        gate_layer=6,
        dataset="v2",
    )
