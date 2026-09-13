#!/usr/bin/env bash
set -euo pipefail

project=/home/libok/morai_project
python=/home/libok/vla_train_env/bin/python
data_root="$project/morai_dataset/processed/tcp_command_map_bag_avoid_combined_v001"
control_cache="$project/morai_dataset/processed/tcp_command_map_bag_avoid_controls_v001"
previous="$project/training_outputs/tcp_command_map_plus_bag_avoid_v1_b80/latest.pt"
output="$project/training_outputs/tcp_command_map_plus_bag_avoid_v1_b80_continue20"

mkdir -p "$output"
cd "$project"

exec "$python" -u -m tcp_morai_finetune.train_full_policy \
  --data-root "$data_root" \
  --split-manifest "$data_root/split.json" \
  --control-cache "$control_cache" \
  --init-checkpoint "$previous" \
  --output-dir "$output" \
  --epochs 20 \
  --batch-size 80 \
  --num-workers 8 \
  --sampling-profile hard-events \
  --selection-profile hard-events \
  --hard-event-fractions 0.20 0.25 0.20 0.15 0.20
