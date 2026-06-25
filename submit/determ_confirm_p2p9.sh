#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/determ_confirm_$JOB_ID.log
#$ -l gpu=1
# Close the P2/P9 gap: they only TIMED OUT in the origa sweep (slow), never crashed.
# Re-run on glas2015 (133-pool, fast) so they COMPLETE under strict determinism -> a
# clean rc=0 confirms BALD + PAAL acquisition paths have no non-deterministic op.
source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env
export CUDA_VISIBLE_DEVICES=0
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code
export MEDAL_STRICT_DETERMINISM=1
python -c "import torch;print('GPU',torch.cuda.get_device_name(0))"
for POL in P2 P9; do
  timeout 300 python -u -m medal_bench.runner.run_one \
    --policy $POL --dataset glas2015 --seed 2000 --profile bench512_v5 \
    --out-dir /tmp/determ_confirm --device cuda:0 --foundation stub \
    --adaptive --min-iters 10 --max-iters 25 --force > /tmp/confirm_$POL.log 2>&1
  rc=$?
  op=$(grep -m1 'does not have a deterministic' /tmp/confirm_$POL.log | sed 's/.*RuntimeError: //;s/ does.*//')
  echo "RESULT $POL,rc=$rc,${op:-(none)}"
done
echo "CONFIRM_DONE"
