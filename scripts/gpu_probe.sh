#!/bin/bash
# Sample GPU utilisation while a short training run executes, to answer whether
# one spacenav job saturates the card or leaves room for concurrent runs.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${SPACENAV_PY:-$ROOT/.venv-gpu/bin/python}"
OUT="${1:-$ROOT/data/gpuprobe}"
mkdir -p "$OUT"
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,power.draw \
  --format=csv,noheader,nounits -l 2 > "$OUT/util.csv" 2>&1 &
SMI=$!
SPACENAV_INSTRUMENTS=0 "$PY" -u "$ROOT/scripts/train.py" --run gpuprobe --updates 120 \
  --ckpt-every 1000 --log-every 20 > "$OUT/train.log" 2>&1
RC=$?
kill $SMI 2>/dev/null
echo "train rc=$RC"
grep -E "^u |^run " "$OUT/train.log" | grep -viE "cuda_exec" | tail -4
echo "--- utilisation while running (device 0 = 3090) ---"
awk -F', ' '$1==0{n++; u+=$3; m+=$4; if($3>mx)mx=$3; mem=$5; p+=$6}
  END{if(n)printf "samples %d  mean SM %.1f%%  peak SM %d%%  mean mem-bw %.1f%%  mem %s MiB  mean power %.0f W\n",n,u/n,mx,m/n,mem,p/n}' "$OUT/util.csv"
echo "--- SM utilisation histogram ---"
awk -F', ' '$1==0{b=int($3/10)*10; c[b]++} END{for(i=0;i<=100;i+=10) printf "%3d-%3d%% %s\n", i, i+9, c[i]?sprintf("%*s %d", c[i]>60?60:c[i], "", c[i]):""}' "$OUT/util.csv" | tr ' ' '#' | sed 's/#\([0-9]\)/ \1/' | sed 's/^\([0-9 -]*%\)#*/\1 /'
