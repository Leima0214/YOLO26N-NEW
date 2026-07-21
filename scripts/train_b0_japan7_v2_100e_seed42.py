"""Run B0 on the scene-disjoint Japan7-v2 seed42 split."""

from train_japan7_100e_common import run


if __name__ == "__main__":
    run("ultralytics/cfg/models/26/yolo26.yaml", "b0_japan7_v2_scene_disjoint_100e_seed42", dataset="v2")
