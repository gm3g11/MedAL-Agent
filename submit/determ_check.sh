#$ -N determ
#$ -l gpu=1
#$ -o logs/determ_$JOB_ID.log
#$ -j y
#$ -cwd
source /groups/echambe2/gmeng/MedAL-Agent/medal-agent.env
export CUDA_VISIBLE_DEVICES=0
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code
/groups/echambe2/gmeng/conda_envs/medal-agent/bin/python -m medal_bench.runner.run_one \
  --policy P0 --dataset glas2015 --seed 2000 --profile bench512_v5 \
  --out-dir runs/v5_determ_check --force --device cuda:0 --save-predictions --defer-surface
