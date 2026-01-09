#!/bin/bash -l

#$ -P peaclab-mon

#$ -l h_rt=01:00:00

#$ -pe omp 12
#$ -l mem_per_core=12G

# Merge stderr and stdout to same file
#$ -e local_outlier_factor_jobs/
#$ -o local_outlier_factor_jobs/
#$ -j y

source /project/peaclab-mon/monitoring_venv.sh
chmod +x $HOME/projectx/AI4HPCAnalytics/src/unsupervised_anomaly_detection_telemetry/scripts/local_outlier_factor.py
$HOME/projectx/AI4HPCAnalytics/src/unsupervised_anomaly_detection_telemetry/scripts/local_outlier_factor.py $1 $2 $3