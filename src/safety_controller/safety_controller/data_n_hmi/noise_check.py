# Re-import necessary libraries after code execution environment reset
import json
import numpy as np
import matplotlib.pyplot as plt

# Reload the JSON data from the uploaded file
file_path = "/home/peiyu/mpc-cbf/CDC/data_n_hmi/output data/['6']/rt_data_['6']_1_['MPC-CBF-QP']_[10].json"
with open(file_path, "r") as f:
    data = json.load(f)

# Extract true positions and noisy positions
true_positions = np.array([entry["true_vehicle"][:2] for entry in data])
noisy_positions = np.array([entry["noisy_enu"][:2] for entry in data])

# Compute Euclidean errors
errors = np.linalg.norm(noisy_positions - true_positions, axis=1)
mean_error = np.mean(errors)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(errors, label="Euclidean Error per Time Step")
plt.axhline(mean_error, color='red', linestyle='--', label=f"Mean Error: {mean_error:.3f} m")
plt.title("Noise-induced Position Error Over Time")
plt.xlabel("Time Step")
plt.ylabel("Error (meters)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

mean_error
