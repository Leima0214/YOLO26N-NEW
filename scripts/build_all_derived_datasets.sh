#!/usr/bin/env bash
# build_all_derived_datasets.sh
# Run this on the remote GPU machine to generate all derived YOLO datasets.
# Assumes the source datasets exist at the paths below.

set -e

echo "=== Paper 1: Japan7 ==="
python scripts/build_remapped_yolo_dataset.py \
  --src /yolo26-probe/japan_yolo \
  --dst /yolo26-probe/derived/japan7 \
  --mapping configs/mappings/japan7.yaml \
  --mode symlink \
  --keep-empty

echo ""
echo "=== Paper 2: Common4 ==="

# Japan
python scripts/build_remapped_yolo_dataset.py \
  --src /yolo26-probe/japan_yolo \
  --dst /yolo26-probe/derived/common4/Japan \
  --mapping configs/mappings/common4.yaml \
  --mode symlink \
  --keep-empty

# Czech — uncomment when czech_yolo is uploaded
# python scripts/build_remapped_yolo_dataset.py \
#   --src /yolo26-probe/czech_yolo \
#   --dst /yolo26-probe/derived/common4/Czech \
#   --mapping configs/mappings/common4.yaml \
#   --mode symlink \
#   --keep-empty

# India — uncomment when india_yolo is uploaded
# python scripts/build_remapped_yolo_dataset.py \
#   --src /yolo26-probe/india_yolo \
#   --dst /yolo26-probe/derived/common4/India \
#   --mapping configs/mappings/common4.yaml \
#   --mode symlink \
#   --keep-empty

# China_MotorBike — uncomment when china_motorbike_yolo is uploaded
# python scripts/build_remapped_yolo_dataset.py \
#   --src /yolo26-probe/china_motorbike_yolo \
#   --dst /yolo26-probe/derived/common4/China_MotorBike \
#   --mapping configs/mappings/common4.yaml \
#   --mode symlink \
#   --keep-empty

echo ""
echo "=== Verify ==="
python scripts/check_dataset.py --data configs/japan7_remote.yaml

echo ""
echo "All derived datasets built. Next steps:"
echo "  python scripts/smoke_test_yolo26n.py --data configs/japan7_remote.yaml --device 0 --batch 8 --workers 4"
