import numpy as np
import ot
import matplotlib.pyplot as plt

# Define mean and covariance matrices
mean_pos = [-25.235371459074646, 43.20407188493103]
sigma1 = np.array([[0.5**2, 0.1 * 0.5 * 0.3], 
                   [0.1 * 0.5 * 0.3, 0.3**2]])
sigma2 = np.array([[1.5**2, 0.1 * 1.5 * 0.3], 
                   [0.1 * 1.5 * 0.3, 0.3**2]])
sigma3 = np.array([[2.5**2, 0.1 * 2.5 * 0.3], 
                   [0.1 * 2.5 * 0.3, 0.3**2]])

# Simulated sensor noise data (2D)
np.random.seed(0)
noisy_pos1 = np.random.multivariate_normal(mean=mean_pos, cov=sigma1)  # (2,)
noisy_pos2 = np.random.multivariate_normal(mean=mean_pos, cov=sigma2)  # (2,)
noisy_pos3 = np.random.multivariate_normal(mean=mean_pos, cov=sigma3)  # (2,)

# Generate multiple samples around each noisy position
xy_samples1 = np.random.multivariate_normal(mean=noisy_pos1, cov=sigma1, size=10)  # (10, 2)
xy_samples2 = np.random.multivariate_normal(mean=noisy_pos2, cov=sigma2, size=10)  # (10, 2)
xy_samples3 = np.random.multivariate_normal(mean=noisy_pos3, cov=sigma3, size=10)  # (10, 2)

Xs = [xy_samples1, xy_samples2, xy_samples3]
bs = [0.4, 0.4, 0.2]  # Sensor-level weights
measures_weights = [np.ones(10) / 10 * bs[0], np.ones(10) / 10 * bs[1], np.ones(10) / 10 * bs[2]]  # Per-sample weights

k = 15
a = np.ones(k) / k

# Initialize barycenter support with some spread
all_points = np.vstack(Xs)  # (30, 2)
mean = np.mean(all_points, axis=0)  # (2,)
std = np.std(all_points, axis=0)  # (2,)
Ys_init = mean + np.random.normal(0, 0.1 * std, size=(k, 2))  # (15, 2)

# Try different entropic regularization values
reg_values = [1, 5, 100]
barycenters = []

for regg in reg_values:
    X_bary = ot.bregman.free_support_sinkhorn_barycenter(
        measures_locations=Xs,
        measures_weights=measures_weights,  # List of (10,) arrays
        X_init=Ys_init.copy(),
        b=a,
        weights=bs,  # Sensor-level weights
        reg=regg,
        numItermax=10
    )
    print(f"Barycenter points for reg={regg}:\n", X_bary)
    barycenters.append(X_bary)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = [f"reg = {r}" for r in reg_values]

for i, ax in enumerate(axes):
    ax.scatter(xy_samples1[:, 0], xy_samples1[:, 1], alpha=0.6, label="Sensor 1", color='blue')
    ax.scatter(xy_samples2[:, 0], xy_samples2[:, 1], alpha=0.6, label="Sensor 2", color='green')
    ax.scatter(xy_samples3[:, 0], xy_samples3[:, 1], alpha=0.6, label="Sensor 3", color='orange')
    ax.scatter(*barycenters[i].T, c='red', marker='x', s=80, label="Barycenter")
    ax.set_title(titles[i])
    ax.legend()
    ax.grid(True)
    ax.axis("equal")

plt.tight_layout()
plt.show()