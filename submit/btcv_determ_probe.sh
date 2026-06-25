#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/btcv_determ_$JOB_ID.log
#$ -l gpu=1
# btcv strict-determinism CONFIRMATION after the cross_entropy fix.
# MEDAL_STRICT_DETERMINISM=1 -> use_deterministic_algorithms warn_only=False, so ANY op
# lacking a deterministic impl CRASHES + is named. Fast bounded rounds (<=30 iters) so a
# full pipeline (train+eval+acquire, all 7 rounds) completes quickly. P0 (train/eval path)
# + P9 (most complex acquisition: AccuracyPredictor + KMeans) cover the riskiest paths.
source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env
export CUDA_VISIBLE_DEVICES=0
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code
export MEDAL_STRICT_DETERMINISM=1
python -c "import torch;print('GPU',torch.cuda.get_device_name(0))"
for POL in P0 P9; do
  echo "===== STRICT probe: btcv_synapse $POL s2000 (bounded rounds) ====="
  python -u -m medal_bench.runner.run_one \
    --policy $POL --dataset btcv_synapse --seed 2000 --profile bench512_v5 \
    --out-dir /tmp/btcv_determ_probe --device cuda:0 --foundation stub \
    --adaptive --min-iters 10 --max-iters 30 --force
  echo "===== $POL EXIT CODE: $? (0 = strict-clean, all ops deterministic) ====="
done
echo "ALL_PROBES_DONE"
