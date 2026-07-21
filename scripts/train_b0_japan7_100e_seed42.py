"""Run the guarded B0 side of the Japan7 seed42 100e pair."""

from train_japan7_100e_common import run


if __name__ == "__main__":
    run("ultralytics/cfg/models/26/yolo26.yaml", "b0_pretrained_auto_japan7_100e_seed42")
