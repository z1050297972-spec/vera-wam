#!/bin/bash
# PushT 92.5% reproduction attempt: server (recipe knobs) + headless driver,
# single GPU. Args: FRAME_INDICES N_REPEATS TAG [PORT]
#SBATCH --job-name=pusht_repro
#SBATCH --account=vision-sitzmann
#SBATCH --partition=vision-sitzmann-l40s
#SBATCH --qos=vision-sitzmann-emergency
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/data/scene-rep/u/sizheli/project/VERA-PUBLIC/examples/logs/sbatch_pusht_repro_%j.log
set -u
FRAME_INDICES=${1:?comma-separated state indices}
N_REPEATS=${2:?n repeats per state}
TAG=${3:?tag}
PORT=${4:-8830}
VISPORT=$((PORT+1))

VP=/data/scene-rep/u/sizheli/project/VERA-PUBLIC
PY=$VP/.venv_pusht_eval/bin/python
mkdir -p "$VP/examples/logs" "$VP/examples/results"

export PYTHONPATH=$VP
export WANDB_MODE=disabled PYTHONUNBUFFERED=1
# The recipe behind the 92.5% headline (docs/PUSHT_REPRODUCTION.md):
export VERA_PUSHT_TRACKER_BACKEND=megaflow
export VERA_PUSHT_N_ACTION_STEPS=3
export VERA_PUSHT_PLANNER_STEPS=70
# motion_plan_scale/action_scale/lam/action_chunk_horizon already default-match.

pkill -u "$USER" -f "start_vera_server.*--port $PORT" 2>/dev/null; sleep 3
(exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && { echo "PORT $PORT occupied — abort"; exit 1; }

echo "[repro] starting server (tracker=megaflow, exec=3, planner_steps=70) on $PORT"
( cd "$VP" && setsid nohup env PYTHONPATH="$VP" \
    "$PY" -m vera.server.start_vera_server --embodiment pusht --port $PORT --vis-port $VISPORT \
    --sample-steps 70 \
    >> "$VP/examples/logs/server_${TAG}.log" 2>&1 ) &

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && exec 3>&- 3<&- && return 0; return 1; }
for i in $(seq 1 60); do port_open $PORT && break; sleep 5; done
port_open $PORT || { echo "[repro] server failed to boot"; cat "$VP/examples/logs/server_${TAG}.log"; exit 1; }
echo "[repro] server up"

"$PY" "$VP/examples/pusht_reproduce_headless.py" --port $PORT \
    --frame-indices "$FRAME_INDICES" --n-repeats "$N_REPEATS" --seed 42 \
    --success-threshold 0.9 --horizon 200 \
    --out "$VP/examples/results/${TAG}.jsonl" \
    --output-dir "$VP/examples/outputs/${TAG}" \
    ${SAVE_VIDEOS:+--save-videos}
rc=$?
pkill -u "$USER" -f "start_vera_server.*--port $PORT" 2>/dev/null
echo "[repro] driver exited rc=$rc"
exit $rc
