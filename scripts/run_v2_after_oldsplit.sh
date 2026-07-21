#!/usr/bin/env bash
set -euo pipefail

cd /root/YOLO26N-NEW
while tmux has-session -t paper1_oldsplit_100e 2>/dev/null; do
  sleep 30
done

old_root=runs/paper1/exploratory_oldsplit
for name in b0_japan7_oldsplit_exploratory_100e_seed42 p4single_japan7_oldsplit_exploratory_100e_seed42; do
  test "$(($(wc -l < "$old_root/$name/results.csv") - 1))" -eq 100
  test -s "logs/exploratory_oldsplit/${name}_reval.log"
done
/opt/conda/bin/python scripts/analyze_oldsplit_100e_pair.py 2>&1 | tee logs/exploratory_oldsplit/pair_analysis.log
test "${PIPESTATUS[0]}" -eq 0

log_dir=logs/japan7_v2_scene_disjoint
run_root=runs/paper1/japan7_v2_scene_disjoint
mkdir -p "$log_dir"

run_and_verify() {
  local entry=$1 name=$2
  /opt/conda/bin/python "$entry" 2>&1 | tee "$log_dir/$name.log"
  test "${PIPESTATUS[0]}" -eq 0
  test -s "$run_root/$name/weights/best.pt"
  test -s "$run_root/$name/results.csv"
  grep -Fq 'split_status: PASS' "$run_root/$name/args.yaml"
  /opt/conda/bin/python scripts/validate_japan7_checkpoint.py \
    "$run_root/$name/weights/best.pt" \
    --data configs/japan7_v2_scene_disjoint/dataset.yaml \
    --name "${name}_reval" 2>&1 | tee "$log_dir/${name}_reval.log"
  test "${PIPESTATUS[0]}" -eq 0
}

run_and_verify scripts/train_b0_japan7_v2_100e_seed42.py b0_japan7_v2_scene_disjoint_100e_seed42

if /opt/conda/bin/python -c "import json; assert json.load(open('reports/oldsplit_exploratory_100e_pair_analysis.json'))['run_p4_v2']"; then
  run_and_verify scripts/train_p4single_japan7_v2_100e_seed42.py p4single_japan7_v2_scene_disjoint_100e_seed42
else
  printf 'P4 Japan7-v2 skipped by oldsplit decision rule.\n'
fi
