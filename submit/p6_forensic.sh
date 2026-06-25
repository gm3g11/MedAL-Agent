#$ -N p6foren
#$ -l gpu=1
#$ -o logs/p6foren_$JOB_ID.log
#$ -j y
#$ -cwd
export PYTHONPATH=/groups/echambe2/gmeng/MedAL-Agent/repo/code:$PYTHONPATH
cd /groups/echambe2/gmeng/MedAL-Agent/repo/code
/groups/echambe2/gmeng/conda_envs/medal-agent/bin/python submit/p6_forensic.py
