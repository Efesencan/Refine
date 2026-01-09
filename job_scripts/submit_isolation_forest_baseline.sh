#!/bin/bash -l

#$ -P peaclab-mon

#$ -l h_rt=01:00:00

#$ -pe omp 12
#$ -l mem_per_core=12G

# Merge stderr and stdout to same file
#$ -e isolation_forest_jobs/
#$ -o isolation_forest_jobs/
#$ -j y

source /project/peaclab-mon/monitoring_venv.sh
chmod +x $HOME/projectx/AI4HPCAnalytics/src/unsupervised_anomaly_detection_telemetry/scripts/isolation_forest.py
$HOME/projectx/AI4HPCAnalytics/src/unsupervised_anomaly_detection_telemetry/scripts/isolation_forest.py $1 $2 $3