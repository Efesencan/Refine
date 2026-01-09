import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from tensorflow.keras.callbacks import Callback


class ReconstructionErrorCallback(Callback):
    def __init__(self, input_data, labels, n_epochs=1):
        super().__init__()
        self.input_data = input_data
        self.labels = labels
        self.n_epochs = n_epochs

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.n_epochs == 0:
            reconstructed = self.model.predict(self.input_data)
            reconstruction_errors = np.mean(np.square(self.input_data - reconstructed), axis=1)

            threshold = np.percentile(reconstruction_errors, 99)  # example threshold
            anomalies = reconstruction_errors > threshold
            anomalous_samples = self.input_data[anomalies]
            anomalous_labels = self.labels[anomalies]

            true_anomalies = np.sum(anomalous_labels)  # Assuming anomalous labels are 1
            total_isolated = len(anomalous_samples)

            print("Accuracy:", true_anomalies / total_isolated)
            print(f"\nReconstruction Error at Epoch {epoch + 1}: {np.mean(reconstruction_errors)}")


class PlotReconstructionErrorCallback(Callback):
    def __init__(self, input_data, labels, n_epochs=20):
        super().__init__()
        self.input_data = input_data
        self.labels = labels
        self.n_epochs = n_epochs
        self.max_anomalous_samples = 0
        self.epoch_with_max_anomalous = 0
        self.anomalous_counts = []

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.n_epochs == 0:
            reconstructed = self.model.predict(self.input_data)
            reconstruction_errors = np.mean(np.square(self.input_data - reconstructed), axis=1)

            # Identify and count anomalous samples
            threshold = np.percentile(reconstruction_errors, 95)
            anomalous_count = np.sum(reconstruction_errors > threshold)

            self.anomalous_counts.append(anomalous_count)
            print("anomalous count", anomalous_count)
            # Check if current epoch has the maximum number of anomalous samples
            if anomalous_count > self.max_anomalous_samples:
                self.max_anomalous_samples = anomalous_count
                self.epoch_with_max_anomalous = epoch

            # Separate the reconstruction errors based on labels
            healthy_errors = reconstruction_errors[self.labels == 0]
            anomalous_errors = reconstruction_errors[self.labels == 1]

            # Calculate and print statistics
            healthy_mean, healthy_median, healthy_std = (
                np.mean(healthy_errors),
                np.median(healthy_errors),
                np.std(healthy_errors),
            )
            anomalous_mean, anomalous_median, anomalous_std = (
                np.mean(anomalous_errors),
                np.median(anomalous_errors),
                np.std(anomalous_errors),
            )

            print(f"\nEpoch {epoch + 1} Statistics:")
            print(
                f"Healthy Samples: Mean={healthy_mean}, Median={healthy_median}, Std={healthy_std}"
            )
            print(
                f"Anomalous Samples: Mean={anomalous_mean}, Median={anomalous_median},"
                f" Std={anomalous_std}"
            )

            # Plotting the distribution of reconstruction errors
            plt.figure(figsize=(10, 6))
            plt.hist(
                healthy_errors,
                bins=50,
                alpha=0.6,
                label=f"Healthy (Mean: {healthy_mean:.4f}, Std: {healthy_std:.4f})",
            )
            plt.hist(
                anomalous_errors,
                bins=50,
                alpha=0.6,
                label=f"Anomalous (Mean: {anomalous_mean:.4f}, Std: {anomalous_std:.4f})",
            )
            plt.title(f"Reconstruction Error Distribution at Epoch {epoch + 1}", fontsize=24)
            plt.xlabel("Reconstruction Error", fontsize=20)
            plt.ylabel("Number of Samples", fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=16)
            plt.show()
            # Additional logic to isolate anomalies can be added here if needed


class PlotAverageErrorCallback(Callback):
    def __init__(self, input_data, labels, n_epochs=20):
        super().__init__()
        self.input_data = input_data
        self.labels = labels
        self.n_epochs = n_epochs
        self.avg_healthy_errors = []
        self.avg_anomalous_errors = []

    def on_epoch_end(self, epoch, logs=None):
        reconstructed = self.model.predict(self.input_data)
        reconstruction_errors = np.mean(np.square(self.input_data - reconstructed), axis=1)

        # Separate the reconstruction errors based on labels
        healthy_errors = reconstruction_errors[self.labels == 0]
        anomalous_errors = reconstruction_errors[self.labels == 1]

        # Calculate and store average errors
        self.avg_healthy_errors.append(np.mean(healthy_errors))
        self.avg_anomalous_errors.append(np.mean(anomalous_errors))

        # Periodically plot the distribution of reconstruction errors
        if (epoch + 1) % self.n_epochs == 0:
            plt.figure(figsize=(10, 6))
            plt.hist(
                healthy_errors,
                bins=50,
                alpha=0.6,
                label=(
                    f"Healthy (Mean: {np.mean(healthy_errors):.4f}, Std:"
                    f" {np.std(healthy_errors):.4f})"
                ),
            )
            plt.hist(
                anomalous_errors,
                bins=50,
                alpha=0.6,
                label=(
                    f"Anomalous (Mean: {np.mean(anomalous_errors):.4f}, Std:"
                    f" {np.std(anomalous_errors):.4f})"
                ),
            )
            plt.title(f"Reconstruction Error Distribution at Epoch {epoch + 1}")
            plt.xlabel("Reconstruction Error")
            plt.ylabel("Number of Samples")
            plt.legend()
            plt.show()

    def on_train_end(self, logs=None):
        # Plot average reconstruction errors over epochs
        plt.figure(figsize=(12, 6))
        plt.plot(self.avg_healthy_errors, label="Average Healthy Error", color="blue")
        plt.plot(self.avg_anomalous_errors, label="Average Anomalous Error", color="red")
        plt.xlabel("Epoch", fontsize=22)
        plt.ylabel("Average Reconstruction Error", fontsize=22)
        plt.title("Average Reconstruction Error Over Epochs", fontsize=24)
        plt.legend(fontsize=20)
        plt.grid(True)
        plt.show()
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)


class PlotAverageSamplesErrorCallback(Callback):
    def __init__(self, input_data, labels, n_epochs=20):
        super().__init__()
        self.input_data = input_data
        self.n_epochs = n_epochs
        self.avg_reconstruction_errors = []

    def on_epoch_end(self, epoch, logs=None):
        reconstructed = self.model.predict(self.input_data)
        reconstruction_errors = np.mean(np.square(self.input_data - reconstructed), axis=1)

        # Calculate and store the average reconstruction error
        self.avg_reconstruction_errors.append(np.mean(reconstruction_errors))

        # Periodically plot the distribution of reconstruction errors
        if (epoch + 1) % self.n_epochs == 0:
            plt.figure(figsize=(10, 6))
            plt.hist(
                reconstruction_errors,
                bins=50,
                alpha=0.6,
                label=f"All Samples (Mean: {np.mean(reconstruction_errors):.4f})",
            )
            plt.title(f"Reconstruction Error Distribution at Epoch {epoch + 1}")
            plt.xlabel("Reconstruction Error")
            plt.ylabel("Number of Samples")
            plt.legend()
            plt.show()

    def on_train_end(self, logs=None):
        # Plot average reconstruction errors over epochs
        plt.figure(figsize=(12, 6))
        plt.plot(self.avg_reconstruction_errors, label="Average Reconstruction Error", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Average Reconstruction Error")
        plt.title("Average Reconstruction Error Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()


class DistributionShiftCallback(Callback):
    def __init__(self, input_data, labels, n_epochs=20):
        super().__init__()
        self.input_data = input_data
        self.labels = labels
        self.n_epochs = n_epochs
        self.error_history = []  # Store error stats for each epoch
        self.wasserstein_distances_healthy = []
        self.wasserstein_distances_anomalous = []

    def on_epoch_end(self, epoch, logs=None):
        reconstructed = self.model.predict(self.input_data)
        reconstruction_errors = np.mean(np.square(self.input_data - reconstructed), axis=1)

        # Separate the reconstruction errors based on labels
        healthy_errors = reconstruction_errors[self.labels == 0]
        anomalous_errors = reconstruction_errors[self.labels == 1]

        # Store the statistics
        error_stats = {
            "epoch": epoch + 1,
            "healthy_errors": healthy_errors,
            "anomalous_errors": anomalous_errors,
        }
        self.error_history.append(error_stats)

        # Calculate Wasserstein distances and store them from the second epoch onwards
        if epoch >= 1:
            prev_epoch_stats = self.error_history[-2]
            distance_healthy = wasserstein_distance(
                prev_epoch_stats["healthy_errors"], healthy_errors
            )
            distance_anomalous = wasserstein_distance(
                prev_epoch_stats["anomalous_errors"], anomalous_errors
            )
            self.wasserstein_distances_healthy.append(distance_healthy)
            self.wasserstein_distances_anomalous.append(distance_anomalous)

        # Plotting at specified intervals
        if (epoch + 1) % self.n_epochs == 0:
            plt.figure(figsize=(10, 6))
            plt.hist(healthy_errors, bins=50, alpha=0.6, label="Healthy")
            plt.hist(anomalous_errors, bins=50, alpha=0.6, label="Anomalous")
            plt.title(f"Reconstruction Error Distribution at Epoch {epoch + 1}")
            plt.xlabel("Reconstruction Error")
            plt.ylabel("Number of Samples")
            plt.legend()
            plt.show()

    def on_train_end(self, logs=None):
        # Find the epoch with the maximum difference in Wasserstein distances
        if self.wasserstein_distances_healthy and self.wasserstein_distances_anomalous:
            max_distance_diff_epoch = (
                np.argmax(
                    np.abs(
                        np.array(self.wasserstein_distances_healthy)
                        - np.array(self.wasserstein_distances_anomalous)
                    )
                )
                + 1
            )
            max_distance_diff = np.abs(
                np.array(self.wasserstein_distances_healthy)
                - np.array(self.wasserstein_distances_anomalous)
            )[max_distance_diff_epoch - 1]
            print(
                "Maximum difference in Wasserstein distances is at Epoch"
                f" {max_distance_diff_epoch}: {max_distance_diff}"
            )