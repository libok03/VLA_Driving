#!/usr/bin/env bash
set -euo pipefail
root=/home/libok/morai_project
python=/home/libok/vla_train_env/bin/python
out="$root/training_outputs/tcp_morai_full_policy_v1"
mkdir -p "$out"
cd "$root"
if [[ "${1:-}" != --worker ]]; then
  setsid --fork "$0" --worker
  for _ in {1..50}; do [[ -s "$out/train.pid" ]] && break; sleep .1; done
  echo "PID $(<"$out/train.pid")"; echo "tail -f $out/train.log"; exit 0
fi
echo $$ > "$out/train.pid"
exec "$python" -u -m tcp_morai_finetune.train_full_policy \
  --data-root "$root/morai_dataset/processed/v9_morai_all_reviewed_v002/all" \
  --split-manifest "$root/multimodal_planner_v9/splits/morai_all_reviewed_v002.json" \
  --control-cache "$root/morai_dataset/processed/tcp_morai_controls_v001" \
  --init-checkpoint "$root/external_models/tcp_reproduction/tcp_state_dict_only.pt" \
  --output-dir "$out" --epochs 10 --batch-size 16 --num-workers 6 \
  > "$out/train.log" 2>&1
