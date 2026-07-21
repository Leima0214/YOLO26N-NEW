"""Run the P4 Single side of the Japan7 exploratory old-split seed42 100e pair."""

from train_japan7_100e_common import run


if __name__ == "__main__":
    run(
        "ultralytics/cfg/models/26/yolo26n-OverLoCK-ProjectFit-P4-Gate1e3.yaml",
        "p4single_japan7_oldsplit_exploratory_100e_seed42",
        gate_layer=6,
    )
