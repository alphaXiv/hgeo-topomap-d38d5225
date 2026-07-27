#!/usr/bin/env bash
set -euo pipefail

START_EPOCH=$(date +%s)
DATA_DIR=/tmp/openlane_v2_sample
CACHE_DIR=/tmp/hgeo_cache
RESULTS_DIR=/tmp/hgeo_results
SAMPLE_TAR=/tmp/OpenLane-V2_sample.tar

echo "RUN_START_UTC $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "COMMIT $(git rev-parse HEAD)"
echo "CONFIG $(tr -d '\n' < reproduction/config.json)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

python -m pip install --quiet --disable-pip-version-check \
  "transformers==4.53.2" "scipy==1.15.3" "pillow>=10.4"

if [[ ! -f "$SAMPLE_TAR" ]]; then
  python -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' \
    "https://drive.usercontent.google.com/download?id=1Ni-L6u1MGKJRAfUXm39PdBIxdk_ntdc6&export=download&confirm=t" \
    "$SAMPLE_TAR"
fi
mkdir -p "$DATA_DIR" "$CACHE_DIR" "$RESULTS_DIR"
tar -xf "$SAMPLE_TAR" -C "$DATA_DIR"

CUDA_VISIBLE_DEVICES=0 python -m reproduction.prepare \
  --data-root "$DATA_DIR" --cache "$CACHE_DIR"

pids=()
for seed in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$seed" python -m reproduction.train \
    --seed "$seed" --cache "$CACHE_DIR" --output "$RESULTS_DIR/seed_${seed}.json" \
    > "$RESULTS_DIR/seed_${seed}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
for seed in 0 1 2 3; do
  echo "SEED_LOG_BEGIN $seed"
  sed -n '1,240p' "$RESULTS_DIR/seed_${seed}.log"
  echo "SEED_LOG_END $seed"
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one seed process failed" >&2
  exit "$status"
fi

python -m reproduction.aggregate --results "$RESULTS_DIR"
END_EPOCH=$(date +%s)
echo "RUN_WALL_SECONDS $((END_EPOCH - START_EPOCH))"
echo "RUN_END_UTC $(date -u +%Y-%m-%dT%H:%M:%SZ)"
