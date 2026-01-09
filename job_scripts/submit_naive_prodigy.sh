#!/bin/bash -l

#$ -P peaclab-mon

#$ -l h_rt=01:00:00

#$ -pe omp 12
#$ -l mem_per_core=12G

# Merge stderr and stdout to same file
#$ -e naive_prodigy_jobs/
#$ -o naive_prodigy_jobs/
#$ -j y

source /project/peaclab-mon/monitoring_venv.sh
chmod +x $HOME/projectx/AI4HPCAnalytics/src/unsupervised_anomaly_detection_telemetry/scripts/naive_prodigy.py
$HOME/projectx/AI4HPCAnalytics/src/unsupervised_anomaly_detection_telemetry/scripts/naive_prodigy.py $1 $2 $3