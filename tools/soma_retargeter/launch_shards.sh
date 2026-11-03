#!/bin/bash
# Launch one detached SOMA converter per shard (see README "Full set").
#   launch_shards.sh <robot> <shard root> <export dir> <robots dir> <log dir> <gpu>...
# Shard i runs on the i-th GPU of the list, cycling. Batch size 64: the converter is
# bound by per-step launches, so a bigger batch raises throughput almost linearly.
set -euo pipefail
robot=$1 shards=$2 export=$3 robots=$4 logs=$5
shift 5
gpus=("$@")
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$logs"
cd $DATA/soma-retargeter
k=0
for s in "$shards"/s*; do
  name=$(basename "$s")
  cfg="$logs/batch_${robot}_${name}.json"
  sed -e "s#__IMPORT__#$s#" -e "s#__EXPORT__#$export#" -e "s#__ROBOTS__#$robots#" \
      -e "s#\"batch_size\": 10#\"batch_size\": 64#" "$here/batch_$robot.json" > "$cfg"
  g=${gpus[$((k % ${#gpus[@]}))]}
  k=$((k + 1))
  CUDA_VISIBLE_DEVICES=$g setsid nohup ~/.local/bin/uv run python app/bvh_to_csv_converter.py \
    --config "$cfg" > "$logs/${robot}_${name}.log" 2>&1 < /dev/null &
done
echo "launched $k converters for $robot"
