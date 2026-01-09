#!/bin/bash -l

#$ -P peaclab-mon

#$ -l h_rt=12:00:00

#$ -pe omp 16
#$ -l mem_per_core=12G

# Merge stderr and stdout to same file
#$ -e iterative_robust_prodigy_jobs/
#$ -o iterative_robust_prodigy_jobs/
#$ -j y

source /project/peaclab-mon/monitoring_venv.sh
chmod +x $HOME/projectx/AI4HPCAnalytics/src/unsupervised_anomaly_detection_telemetry/scripts/iterative_refine.py
$HOME/projectx/AI4HPCAnalytics/src/unsupervised_anomaly_detection_telemetry/scripts/iterative_refine.py $1 $2 $3