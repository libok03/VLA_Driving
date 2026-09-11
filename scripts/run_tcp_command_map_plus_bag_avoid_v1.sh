#!/usr/bin/env bash
set -euo pipefail

project=/home/libok/morai_project
python=/home/libok/vla_train_env/bin/python
ssd=/media/libok/b4bc958b-d82f-4611-96bd-67d55146f5bf/home/libok
teacher_bags="$ssd/tcp_teacher_bags"
teacher_data="$ssd/tcp_teacher_processed_v2_localization_command"
teacher_controls="$ssd/tcp_teacher_controls_v2_localization_command"
human_data="$project/morai_dataset/processed/v9_morai_all_reviewed_v002/all"
human_split="$project/multimodal_planner_v9/splits/morai_all_reviewed_v002.json"
human_controls="$project/morai_dataset/processed/tcp_morai_controls_v001"
combined_data="$project/morai_dataset/processed/tcp_command_map_bag_avoid_combined_v001"
combined_controls="$project/morai_dataset/processed/tcp_command_map_bag_avoid_controls_v001"
output="$project/training_outputs/tcp_command_map_plus_bag_avoid_v1_b80"

mkdir -p "$output"
cd "$project"

echo "pipeline_stage conversion"
"$python" -u scripts/build_tcp_teacher_dataset.py \
  --bag-dir "$teacher_bags" \
  --output "$teacher_data" \
  --control-cache "$teacher_controls"

echo "pipeline_stage combined_view"
"$python" -u scripts/build_tcp_combined_dataset_view.py \
  --human-root "$human_data" \
  --human-split "$human_split" \
  --human-controls "$human_controls" \
  --teacher-root "$teacher_data" \
  --teacher-split "$teacher_data/split.json" \
  --teacher-controls "$teacher_controls" \
  --output-root "$combined_data" \
  --output-controls "$combined_controls"

echo "pipeline_stage training"
exec "$python" -u -m tcp_morai_finetune.train_full_policy \
  --data-root "$combined_data" \
  --split-manifest "$combined_data/split.json" \
  --control-cache "$combined_controls" \
  --init-checkpoint "$project/training_outputs/tcp_teacher_full_policy_v6_command_fixed_b32_cont29/best.pt" \
  --output-dir "$output" \
  --epochs 10 \
  --batch-size 80 \
  --num-workers 8 \
  --sampling-profile hard-events \
  --selection-profile hard-events \
  --hard-event-fractions 0.20 0.25 0.20 0.15 0.20
