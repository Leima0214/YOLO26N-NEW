#!/usr/bin/env bash
set -euo pipefail

cd /root/YOLO26N-NEW
log_dir=logs/exploratory_oldsplit
run_root=runs/paper1/exploratory_oldsplit
mkdir -p "$log_dir"

run_and_verify() {
  local entry=$1 name=$2 log=$3 eval_name=$4
  /opt/conda/bin/python "$entry" --allow-known-development-split 2>&1 | tee "$log_dir/$log"
  test "${PIPESTATUS[0]}" -eq 0
  test -s "$run_root/$name/weights/best.pt"
  test -s "$run_root/$name/results.csv"
  grep -Fq 'split_status: KNOWN_NEAR_DUPLICATE_DEVELOPMENT_SPLIT' "$run_root/$name/args.yaml"
  /opt/conda/bin/python scripts/validate_japan7_checkpoint.py \
    "$run_root/$name/weights/best.pt" \
    --name "$eval_name" 2>&1 | tee "$log_dir/${eval_name}.log"
  test "${PIPESTATUS[0]}" -eq 0
}

run_and_verify \
  scripts/train_b0_japan7_100e_seed42.py \
  b0_japan7_oldsplit_exploratory_100e_seed42 \
  b0_japan7_oldsplit_exploratory_100e_seed42.log \
  b0_japan7_oldsplit_exploratory_100e_seed42_reval

run_and_verify \
  scripts/train_p4single_japan7_100e_seed42.py \
  p4single_japan7_oldsplit_exploratory_100e_seed42 \
  p4single_japan7_oldsplit_exploratory_100e_seed42.log \
  p4single_japan7_oldsplit_exploratory_100e_seed42_reval
