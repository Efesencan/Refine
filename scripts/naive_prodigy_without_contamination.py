#!/usr/bin/env python
# coding: utf-8

import json
import logging
import os
import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
import seaborn as sns
import tensorflow as tf
from keras.callbacks import Callback, ModelCheckpoint
from scipy.spatial import distance, distance_matrix
from scipy.stats import wasserstein_distance
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2, f_classif
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras import backend as K, optimizers
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.losses import binary_crossentropy, kl_divergence, mse
from tensorflow.keras.models import Model, load_model
from tslearn.clustering import TimeSeriesKMeans
from tslearn.generators import random_walks
from tslearn.utils import to_time_series_dataset

from VAE import VAE
from callbacks import (
    PlotReconstructionErrorCallback,
    ReconstructionErrorCallback,
    PlotAverageSamplesErrorCallback,
)
from kneed import KneeLocator

# Local imports
os.chdir(Path("../../"))
print(os.getcwd())
sys.path.insert(0, "/usr3/graduate/esencan/projectx/AI4HPCAnalytics/src/")
from config import Configuration
from datasets import EclipseSampledDataset, VoltaSampledDataset
from utils import *

logging.basicConfig(
    format="%(asctime)s %(levelname)-7s %(message)s", stream=sys.stderr, level=logging.INFO
)

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def sample(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian."""
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def get_error_term(v1, v2, _rmse=True):
    if _rmse:
        return np.sqrt(np.mean((v1 - v2) ** 2, axis=1))
    # return MAE
    return np.mean(abs(v1 - v2), axis=1)


def create_vae_model(actual_train_data):
    original_dim = actual_train_data.shape[1]  # Dimension of input data
    intermediate_dim = int(original_dim / 2)
    latent_dim = int(original_dim / 3)

    # Instantiate the VAE class
    vae = VAE(original_dim=original_dim, intermediate_dim=intermediate_dim, latent_dim=latent_dim)

    # Compile the model within the class with the specified optimizer settings
    vae.compile(optimizer="adam", learning_rate=0.0001, clipvalue=0.5)
    vae.model.summary()  # Ensure the summary is for the correct model attribute

    return vae.model


# Compute the false positive rate for the positive class
def compute_fpr(report_dict):
    total_support = report_dict["macro avg"]["support"]
    positive_class = "1.0"
    metrics = report_dict[positive_class]

    precision = metrics["precision"]
    recall = metrics["recall"]
    support = metrics["support"]

    # True Positives
    tp = recall * support

    # False Positives
    fp = (tp / precision) - tp

    # True Negatives
    tn = total_support - support - fp

    # False Positive Rate
    fpr = fp / (fp + tn)

    return fpr


# Compute the false negative rate for the negative class
def compute_fnr(report_dict):
    total_support = report_dict["macro avg"]["support"]
    positive_class = "1.0"
    metrics = report_dict[positive_class]

    precision = metrics["precision"]
    recall = metrics["recall"]
    support = metrics["support"]

    # True Positives
    tp = recall * support

    # False Negatives
    fn = support - tp

    # False Negative Rate
    fnr = fn / (fn + tp)

    return fnr


def main():
    if len(sys.argv) < 4:
        print("Usage: python script.py <cv_index> <system> <train_anom_ratio>")
        sys.exit(1)

    result_df = pd.DataFrame(
        columns=[
            "train_anomaly_ratio",
            "test_anomaly_ratio",
            "dataset",
            "f1-score",
            "AUC",
            "cv_fold",
            "false_positive_rate",
            "false_negative_rate",
        ]
    )

    CV_INDEX = int(sys.argv[1])  # First argument
    SYSTEM = sys.argv[2]  # Second argument
    train_anom_ratio = float(sys.argv[3])  # Third argument

    save_dir = "/projectnb/peaclab-mon/sencan/changepoint_experiments/"
    model_save_file_name = "/projectnb/peaclab-mon/sencan/USAD/model.pth"
    user = "aksar"
    logging.warning(f"Are you sure that you are: {user}?")

    if SYSTEM == "volta":
        OUTPUT_DIR = f"/projectnb/peaclab-mon/{user}/active_learning_experiments"
    elif SYSTEM == "eclipse":
        OUTPUT_DIR = f"/projectnb/peaclab-mon/{user}/active_learning_experiments_final_hdfs"

    NUM_FEATURE = 2000
    FE_NAME = "tsfresh"
    EXP_NAME = "tsfresh_experiments"
    FS_NAME = "CHI"
    FEATURE_SELECTION = False
    SCALER = "None"
    MODEL_CONFIG = "changepoint_experiment_results"

    logging.warning("Results will be generated in {}, double check please!".format(MODEL_CONFIG))

    conf = Configuration(
        ipython=True,
        overrides={
            "output_dir": Path(OUTPUT_DIR),  # change
            "system": SYSTEM,
            "exp_name": EXP_NAME,
            "cv_fold": CV_INDEX,
            "model_config": MODEL_CONFIG,
        },
    )

    with open(str(conf["experiment_dir"]) + "/anom_dict.json") as f:
        ANOM_DICT = json.load(f)
    with open(str(conf["experiment_dir"]) + "/app_dict.json") as f:
        APP_DICT = json.load(f)

    APP_REVERSE_DICT = {}
    for app_name, app_encoding in APP_DICT.items():
        APP_REVERSE_DICT[app_encoding] = app_name

    ANOM_REVERSE_DICT = {}
    for anom_name, anom_encoding in ANOM_DICT.items():
        ANOM_REVERSE_DICT[anom_encoding] = anom_name

    if SYSTEM == "eclipse":
        eclipseDataset = EclipseSampledDataset(conf)
        train_data, train_label, test_data, test_label = eclipseDataset.load_dataset(
            cv_fold=CV_INDEX,
            scaler=SCALER,
            borghesi=False,
            mvts=True if FE_NAME == "mvts" else False,
            tsfresh=True if FE_NAME == "tsfresh" else False,
        )

    elif SYSTEM == "volta":
        voltaDataset = VoltaSampledDataset(conf)
        train_data, train_label, test_data, test_label = voltaDataset.load_dataset(
            cv_fold=CV_INDEX,
            scaler=SCALER,
            borghesi=False,
            mvts=True if FE_NAME == "mvts" else False,
            tsfresh=True if FE_NAME == "tsfresh" else False,
        )

    # assert list(train_data.index) == list(train_label.index)  # check the order of the labels
    # assert list(test_data.index) == list(test_label.index)  # check the order of the labels

    if FEATURE_SELECTION:
        selected_features = pd.read_csv(conf["experiment_dir"] / "selected_features.csv")
        train_data = train_data[list(selected_features["0"].values)]
        test_data = test_data[list(selected_features["0"].values)]

    train_label["anom_names"] = train_label.apply(lambda x: ANOM_REVERSE_DICT[x["anom"]], axis=1)
    train_label["app_names"] = train_label["app"].apply(lambda x: APP_REVERSE_DICT[x])
    test_label["anom_names"] = test_label.apply(lambda x: ANOM_REVERSE_DICT[x["anom"]], axis=1)
    test_label["app_names"] = test_label["app"].apply(lambda x: APP_REVERSE_DICT[x])

    all_data = pd.concat([train_data, test_data])
    all_data = all_data.dropna(axis=1, how="any")
    all_label = pd.concat([train_label, test_label])

    # indexing the same way
    all_data_indices = all_data.index.get_level_values("node_id").unique()
    all_label = all_label.loc[all_data_indices]

    train_data = all_data.loc[train_label.index]
    test_data = all_data.loc[test_label.index]

    logging.info("Train data shape %s", train_data.shape)
    logging.info("Train label shape %s", train_label.shape)
    logging.info("Test data shape %s", test_data.shape)
    logging.info("Test label shape %s", test_label.shape)

    logging.info("Train data label dist: \n%s", train_label["anom"].value_counts())
    logging.info("Test data label dist: \n%s", test_label["anom"].value_counts())

    SCALER = "MinMax"  # previously it was standard scaler

    if SCALER == "MinMax":
        minmax_scaler = MinMaxScaler().fit(train_data)
        train_data = pd.DataFrame(
            minmax_scaler.transform(train_data), columns=train_data.columns, index=train_data.index
        )
        test_data = pd.DataFrame(
            minmax_scaler.transform(test_data), columns=test_data.columns, index=test_data.index
        )

    elif SCALER == "Standard":
        # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
        scaler = StandardScaler().fit(train_data)
        train_data = pd.DataFrame(
            scaler.transform(train_data), columns=train_data.columns, index=train_data.index
        )
        test_data = pd.DataFrame(
            scaler.transform(test_data), columns=test_data.columns, index=test_data.index
        )

    logging.info(train_data.shape)
    logging.info(test_data.shape)

    FS_NAME = "CHI"  # Implement new feature selection strategies below
    # FS_NAME = 'Variance'
    # FS_NAME = 'F'
    if FS_NAME == "CHI":
        selector = SelectKBest(chi2, k=NUM_FEATURE)
        selector.fit(train_data, train_label["anom"])
        train_data = train_data[train_data.columns[selector.get_support(indices=True)]]
        selected_columns = train_data.columns
        test_data = test_data[test_data.columns & selected_columns]

    elif FS_NAME == "Variance":
        selector = VarianceThreshold(threshold=0.1)
        selector.fit_transform(train_data)
        train_data = train_data[train_data.columns[selector.get_support(indices=True)]]
        selected_columns = train_data.columns
        test_data = test_data[test_data.columns & selected_columns]

    else:
        selector = SelectKBest(f_classif, k=NUM_FEATURE)
        selector.fit(train_data, train_label["anom"])
        train_data = train_data[train_data.columns[selector.get_support(indices=True)]]
        selected_columns = train_data.columns
        test_data = test_data[test_data.columns & selected_columns]

    ## Determine Train Test Split
    ### use the feature extracted version of the code
    ### filter out the anomalous samples from the training dataset
    healthy = "None" if SYSTEM == "eclipse" else "none"

    healthy_data_node_ids = train_label[train_label["anom_names"] == healthy].index
    healthy_train_data = train_data[
        train_data.index.get_level_values("node_id").isin(healthy_data_node_ids)
    ]
    node_id_order = healthy_train_data.index.get_level_values("node_id").unique()
    healthy_label = train_label.reindex(node_id_order)

    testing_data = test_data.copy()
    testing_label = test_label.copy()

    # Train - Validation Split
    train_validation_split = (
        True  # if this is false, then you get the validation set from the test data
    )
    train_with_only_healthy_data = True

    anom_ratio = train_anom_ratio
    if anom_ratio == 1:  # this means it is %1
        new_train_data = train_data
        new_train_label = train_label
    else:
        if not train_with_only_healthy_data:
            # anomalous data
            anom_data_node_ids = train_label[train_label["anom_names"] != healthy].index
            anomalous_train_data = train_data[
                train_data.index.get_level_values("node_id").isin(anom_data_node_ids)
            ]
            anomalous_train_label = train_label[
                train_label.index.get_level_values("node_id").isin(anom_data_node_ids)
            ]

            healthy_node_ids = train_label[train_label["anom_names"] == healthy].index
            healthy_train_label = train_label[
                train_label.index.get_level_values("node_id").isin(healthy_node_ids)
            ]

            anomalous_train_data_label, anomalous_test_data_label = train_test_split(
                anomalous_train_label,
                test_size=anom_ratio,
                stratify=anomalous_train_label[["anom"]],
                random_state=0,
            )  # for the volta dataset (it as also ,input_column_name)
            updated_train_label = pd.concat([healthy_train_label, anomalous_test_data_label])
            new_train_data = train_data[
                train_data.index.get_level_values("node_id").isin(updated_train_label.index)
            ]

            node_id_order = new_train_data.index.get_level_values("node_id").unique()
            new_train_label = updated_train_label.reindex(node_id_order)
        else:
            # you should remove anomalies from the trianing set
            healthy_data_node_ids = train_label[train_label["anom_names"] == "none"].index
            healthy_train_data = train_data[
                train_data.index.get_level_values("node_id").isin(healthy_data_node_ids)
            ]
            healthy_train_label = train_label[
                train_label.index.get_level_values("node_id").isin(healthy_data_node_ids)
            ]

    if train_with_only_healthy_data:
        anomaly_ratio = 0

    print("Anomaly ratio:", anomaly_ratio)

    epoch_count = 400  # Number of epochs for initial training
    vae_model = create_vae_model(healthy_train_data)
    results = vae_model.fit(
        healthy_train_data,
        healthy_train_data,
        shuffle=True,
        epochs=epoch_count,  # instead if fix 30 epoch_count
        batch_size=256,
    )

    # predictions on test dataset
    X_train_pred = vae_model.predict(healthy_train_data)  # previously it was healthy_train_data
    mae_vector = get_error_term(X_train_pred, healthy_train_data, _rmse=False)
    error_thresh = np.quantile(mae_vector, 0.99)
    X_pred = vae_model.predict(test_data)  # previously it was actual_test_data
    test_mae_vector = get_error_term(
        X_pred, test_data, _rmse=False
    )  # previously it was actual_test_data
    anomalies = test_mae_vector > error_thresh
    y_test = [
        0.0 if row["anom"] == 0 else 1.0 for index, row in test_label.iterrows()
    ]  # it was actual_test_from_test_data_label
    classification_results = classification_report(y_test, anomalies, output_dict=True)
    # Calculate the FPR for the positive class
    fpr_positive_class = compute_fpr(classification_results)
    print(f"False Positive Rate for class 1.0: {fpr_positive_class:.4f}")

    # Calculate the FNR for the negative class
    fnr_negative_class = compute_fnr(classification_results)
    print(f"False Negative Rate for class 0: {fnr_negative_class:.4f}")
    # Compute AUC score
    # Handle NaN and infinity values in test_mae_vector
    test_mae_vector = np.nan_to_num(
        test_mae_vector,
        nan=np.nanmean(test_mae_vector),
        posinf=np.nanmean(test_mae_vector),
        neginf=np.nanmean(test_mae_vector),
    )
    anomalies = (test_mae_vector > error_thresh).astype(float)
    # Compute AUC score
    auc_score = roc_auc_score(y_test, test_mae_vector)
    print(f"AUC Score: {auc_score}")
    new_row = {
        "train_anomaly_ratio": anomaly_ratio,
        "test_anomaly_ratio": 0.93 if SYSTEM == "eclipse" else 0.10,
        "dataset": SYSTEM,
        "f1-score": classification_results["macro avg"]["f1-score"],
        "cv_fold": CV_INDEX,
        "false_positive_rate": fpr_positive_class,
        "false_negative_rate": fnr_negative_class,
        "AUC": auc_score,
    }
    result_df = result_df.append(new_row, ignore_index=True)
    # Constructing the filename with variable values
    # Define the directory path
    directory_path = (
        "/projectnb/peaclab-mon/sencan/robust_prodigy/naive_prodigy_without_contamination_results"
    )

    # Constructing the filename with variable values
    filename = f"{directory_path}/naive_prodigy_{str(anomaly_ratio)}%_cv_{CV_INDEX}_{SYSTEM}.csv"

    # Save the dataframe to CSV
    result_df.to_csv(filename)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    main()