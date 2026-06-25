#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/determ_sweep_$JOB_ID.log
#$ -l gpu=1
# Strict-determinism SWEEP across all 10 policies. MEDAL_STRICT_DETERMINISM=1 ->
# use_deterministic_algorithms(warn_only=False), so any op lacking a deterministic
# impl CRASHES + is named. origa = 3-class (exercises the multi-class chunked
# acquisition paths) + small pool (fast). Bounded iters so each policy completes
# train+eval+acquisition for all rounds quickly. rc: 0=clean, 2=crash(non-det op),
# 124=timeout(hung, inspect).
source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env
export CUDA_VISIBLE_DEVICES=0
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code
export MEDAL_STRICT_DETERMINISM=1
python -c "import torch;print('GPU',torch.cuda.get_device_name(0))"
echo "POLICY,rc,nondet_op"
for POL in P0 P1 P2 P3 P4 P5 P6 P7 P8 P9; do
  timeout 240 python -u -m medal_bench.runner.run_one \
    --policy $POL --dataset origa --seed 2000 --profile bench512_v5 \
    --out-dir /tmp/determ_sweep --device cuda:0 --foundation stub \
    --adaptive --min-iters 10 --max-iters 25 --force > /tmp/sweep_$POL.log 2>&1
  rc=$?
  op=$(grep -m1 'does not have a deterministic' /tmp/sweep_$POL.log | sed 's/.*RuntimeError: //; s/ does not have.*//')
  echo "RESULT $POL,rc=$rc,${op:-(none)}"
done
echo "SWEEP_DONE"
