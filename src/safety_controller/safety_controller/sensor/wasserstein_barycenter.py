import numpy as np
import ot
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

class WB_distribution:

    def __init__(self):

        super().__init__()


    def get_freesupport_wb(self, Xs, bs, regg=1.0, k=10):
        """
        Compute Wasserstein barycenter over (x, y) for each obstacle at the first time step (t=0).

        Args:
            Xs: list of 3 arrays (lidar, camera, v2x), each shape (N, M, S, D)
            bs: list of 3 floats — sensor-level weights (e.g., [0.33, 0.34, 0.33])
            regg: entropic regularization for Sinkhorn (default: 5.0)
            k: number of support points in barycenter (default: 15)

        Returns:
            barycenters: array of shape (M, k, 2) → barycenter (x, y) support points for t=0
        """
        N, M, S, D = Xs[0].shape  # Xs[0] is the first sensor's data
        a = np.ones(k) / k        # uniform weights for barycenter support, shape (k,)
        barycenters = np.zeros((M, k, 2))  # output array for (x, y) barycenters at t=0
        for obs in range(M):
            # Extract (S, 2) samples for each sensor at [t=0, obs], taking only (x, y)
            sensor_samples = [sensor_data[0, obs, :, :2] for sensor_data in Xs]  # list of (S, 2) arrays

            # Compute weights for each sensor's samples, summing to bs[i]
            b_weights = [np.ones(S) * (bs[i] / S) for i in range(len(Xs))]  # list of (S,) arrays

            # Initialize barycenter support points
            all_points = np.vstack(sensor_samples)  # shape (3*S, 2)
            mean = np.mean(all_points, axis=0)  # (2,)
            std = np.std(all_points, axis=0)  # (2,)
            Ys_init = mean + np.random.normal(0, 0.1 * std, size=(k, 2))  # (15, 2)

            # Compute Sinkhorn barycenter
            X_bary = ot.bregman.free_support_sinkhorn_barycenter(
                measures_locations=sensor_samples,  # list of (S, 2) arrays
                measures_weights=b_weights,         # list of (S,) weight arrays
                X_init=Ys_init,                     # initial support, shape (k, 2)
                b=a,                                # barycenter weights, shape (k,)
                weights=bs,                         # sensor weights from input
                reg=regg,                           # regularization parameter
                numItermax=10                       # max iterations
            )
            barycenters[obs] = X_bary  # assign result, shape (k, 2)

        return barycenters

    def get_fixed_wb(self, Xs, bs, regg=10.0, k=5):
        """
        Compute fixed-support Wasserstein barycenter over (x, y) for each obstacle at t=0.

        Args:
            Xs: list of 3 arrays (lidar, camera, v2x), each shape (N, M, S, D)
            bs: list of 3 floats — sensor-level weights (e.g., [0.33, 0.34, 0.33])
            regg: entropic regularization for Sinkhorn (default: 10.0)
            k: number of support points in barycenter (default: 5)

        Returns:
            barycenters: array of shape (M, k, 2) → barycenter (x, y) support points for t=0
        """
        N, Q, S, D = Xs[0].shape
        barycenters = np.zeros((Q, k, 2))

        # Define fixed support points (e.g., a grid around the mean of all samples at t=0)
        all_samples = np.vstack([sensor_data[0, :, :, :2].reshape(-1, 2) for sensor_data in Xs])  # (3*M*S, 2)
        mean = np.mean(all_samples, axis=0)  # (2,)
        std = np.std(all_samples, axis=0)    # (2,)
        fixed_support = np.array([mean + std * np.array([x, y]) 
                                for x in [-1, 0, 1] for y in [-1, 0, 1]])[:k]  # e.g., k points

        # Compute the cost matrix M between fixed support points (k, k)
        M = ot.dist(fixed_support, fixed_support)  # Pairwise Euclidean distances, shape (k, k)

        for obs in range(Q):
            # Extract samples for this obstacle
            sensor_samples = [sensor_data[0, obs, :, :2] for sensor_data in Xs]  # list of (S, 2)
            
            # Compute histograms (weights) over the fixed support for each sensor
            histograms = []
            for samples in sensor_samples:
                M_samples_to_support = ot.dist(samples, fixed_support)  # Cost matrix (S, k)
                hist = ot.bregman.sinkhorn(
                    a=np.ones(S) / S,  # Uniform weights for samples
                    b=np.ones(k) / k,  # Uniform weights for support
                    M=M_samples_to_support,
                    reg=regg
                )  # Shape (S, k)
                histograms.append(hist.sum(axis=0))  # Sum over samples to get (k,)

            # Stack histograms into A, shape (k, 3)
            A = np.vstack(histograms).T  # (k, num_sensors), e.g., (k, 3)

            # Compute fixed-support barycenter weights
            bary_weights = ot.bregman.barycenter(
                A=A,
                M=M,
                reg=regg,
                weights=bs,
                numItermax=20
            )  # Shape (k,)

            # Barycenter is the fixed support weighted by bary_weights
            barycenters[obs] = fixed_support  # Locations are fixed; weights could be used if needed

        return barycenters  # Note: only locations returned; weights discarded here

    