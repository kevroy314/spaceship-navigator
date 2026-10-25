#!/bin/bash
# Does ONE spacenav training job saturate the 3090, or can the card carry two?
#
# Runs N training processes at once on the same card for a fixed number of
# updates and reports aggregate throughput against the known serial figure
# (~112k steps/s for one job).  If two jobs each sustain much more than half of
# that, a single job was leaving the card idle and the queue should run wide.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${SPACENAV_PY:-$ROOT/.venv-gpu/bin/python}"
N="${1:-2}"; UPDATES="${2:-120}"
OUT="$ROOT/data/gpuconc"; mkdir -p "$OUT"
# each process caps itself so N of them cannot overcommit the card
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# One job's measured working set is ~8.6 GB at batch 131,072 (gpu_probe).  Give
# each process a cap a little above that where it fits, and refuse to start if N
# processes cannot each have it -- a starved process OOMs instead of sharing.
NEED_GB=9.0
TOTAL_GB=24.0
PER=$(python3 -c "print(round(min(0.44, ($TOTAL_GB*0.92/$N)/$TOTAL_GB), 3))")
FITS=$(python3 -c "print(1 if ($TOTAL_GB*$PER) >= $NEED_GB else 0)")
if [ "$FITS" != "1" ]; then
  echo "SKIP N=$N: $N x ${NEED_GB}GB working set does not fit in ${TOTAL_GB}GB"
  echo "  (would give each process $(python3 -c "print(round($TOTAL_GB*$PER,1))")GB)"
  exit 0
fi
export XLA_PYTHON_CLIENT_MEM_FRACTION=$PER
echo "N=$N  updates=$UPDATES  per-process MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION"

nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw \
  --format=csv,noheader,nounits -l 2 > "$OUT/util_n$N.csv" 2>&1 &
SMI=$!
T0=$(date +%s)
for i in $(seq 1 "$N"); do
  SPACENAV_INSTRUMENTS=0 "$PY" -u "$ROOT/scripts/train.py" --run "conc${N}_$i" \
      --updates "$UPDATES" --ckpt-every 100000 --log-every 20 --seed "$i" \
      > "$OUT/conc${N}_$i.log" 2>&1 &
done
wait
T1=$(date +%s)
kill $SMI 2>/dev/null
WALL=$((T1-T0))
echo "wall ${WALL}s for $N concurrent runs of $UPDATES updates"
python3 - "$OUT" "$N" "$UPDATES" "$WALL" <<'EOF'
import sys,re,glob,os
out,N,upd,wall = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
# batch comes from the run header line
batch=None; per=[]
for f in sorted(glob.glob(os.path.join(out,f"conc{N}_*.log"))):
    txt=open(f,errors="ignore").read()
    m=re.search(r"batch ([\d,]+)",txt)
    if m: batch=int(m.group(1).replace(",",""))
    rows=re.findall(r"u\s+(\d+)\s+([\d.]+)M\s+sps\s+([\d,]+)",txt)
    if len(rows)>=2:
        (u0,s0,c0),(u1,s1,c1)=rows[-2],rows[-1]
        s0,s1=float(s0)*1e6,float(s1)*1e6; c0,c1=float(c0.replace(",","")),float(c1.replace(",",""))
        dt=(s1/c1)-(s0/c0)
        if dt>0: per.append((s1-s0)/dt)
print(f"  batch {batch}")
for i,p in enumerate(per,1): print(f"  process {i} marginal: {p:,.0f} steps/s")
if per:
    tot=sum(per)
    print(f"  AGGREGATE marginal: {tot:,.0f} steps/s across {N} process(es)")
    print(f"  one-process reference: 111,705 steps/s  ->  {tot/111705:.2f}x")
EOF
echo "--- utilisation, device 0 ---"
# Report the whole run AND the steady-state tail.  XLA compiles for well over a
# minute with the card idle, and averaging that in understates utilisation.
awk -F', ' '$1==0{n++;u+=$2;if($2>mx)mx=$2;if($3>mm)mm=$3;p+=$4}
  END{if(n)printf "  whole run : samples %d  mean SM %.1f%%  peak SM %d%%  peak mem %d MiB  mean power %.0f W\n",n,u/n,mx,mm,p/n}' "$OUT/util_n$N.csv"
SKIP=$(awk -F', ' '$1==0{n++} END{print int(n*0.45)}' "$OUT/util_n$N.csv")
awk -F', ' -v skip="$SKIP" '$1==0{i++; if(i<=skip) next; n++;u+=$2;if($2>mx)mx=$2;p+=$4}
  END{if(n)printf "  steady tail: samples %d  mean SM %.1f%%  peak SM %d%%  mean power %.0f W\n",n,u/n,mx,p/n}' "$OUT/util_n$N.csv"
awk -F', ' '$1==0{b=int($2/20)*20; c[b]++; n++} END{printf "  SM histogram:"; for(i=0;i<=100;i+=20) printf " [%d-%d%%] %d", i, i+19, c[i]+0; printf "  (n=%d)\n", n}' "$OUT/util_n$N.csv"
