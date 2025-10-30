from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import rice, laplace, gamma

class Sensor(ABC):
    def __init__(self):
        self.states = None

    @abstractmethod
    def update_obs_pos(self, *args, **kwargs):
        """Update the scenario states."""
        pass

    @abstractmethod
    def samples_from_dis(self, *args, **kwargs):
        """Samples from noisy data"""
        pass

    @abstractmethod
    def set_distribution(self, *args, **kwargs):
        """Initialize sensor distribution."""
        pass

    # @abstractmethod
    # def update_distribution(self, *args, **kwargs):
    #     """Update sensor distribution."""
    #     pass

    @staticmethod
    def create_sensors(all_para):

        sensor_list = {}

        sensor_map = {
            "lidar": LIDAR,
            "camera": CAMERA,
            "v2x": V2X
        }

        sensors = all_para.sensors
        for sensor_type, sensor_info in sensors.items():
            sensor_list[sensor_type] = sensor_map[sensor_type](sensor_type, sensor_info)

        return sensor_list
    
class LIDAR(Sensor):

    def __init__(self, sensor_type, sensor_info):

        super().__init__()
        self.type = sensor_type.lower()
        self.weight = sensor_info.get("weight", None)
        self.mean_x = sensor_info.get("mean_x", None)
        self.mean_y = sensor_info.get("mean_y", None)
        std_x = sensor_info.get("std_x", None)
        std_y = sensor_info.get("std_y", None)
        correlation = sensor_info.get("correlation", None)
        self.sigma = self.set_distribution(std_x, std_y, correlation)

        self.nu = np.array(sensor_info.get("nu", None))
        self.rice_sigma = np.array(sensor_info.get("sigma", None))
        self.b_lidar = self.nu / self.rice_sigma
        print('Done LIDAR initialize, weight')

    # Guaussian distribution to generate one noisy data
    def update_obs_pos(self, num_obs, obs_pos):

        N = obs_pos.shape[0]  # number of time steps

        lidar_pos = obs_pos.copy()

        for i in range(num_obs):
            mean_posx = obs_pos[:, i * 4] + self.mean_x
            mean_posy = obs_pos[:, i * 4 + 1] +self.mean_y

            # Loop through each time step
            for t in range(N):
                mean_pos = np.array([mean_posx[t], mean_posy[t]])  # shape (2,)
                noisy_pos = np.random.multivariate_normal(mean=mean_pos, cov=self.sigma)

                lidar_pos[t, i * 4]     = noisy_pos[0]  # x
                lidar_pos[t, i * 4 + 1] = noisy_pos[1]  # y

        return lidar_pos


    def samples_from_dis(self, num_obs, obs_pos, num_samples):

        # [N, obstacle_id, distribution_samples, state(x,y,psi,v)]
        N = obs_pos.shape[0]  # number of time steps

        lidar_noisy_measurement = self.update_obs_pos(num_obs, obs_pos) # shape (N, num_obs*4)
        noisy_pred = lidar_noisy_measurement.reshape(N, num_obs, 4)  # → [N, M, 4]

        # Output: expanded samples per timestep, obstacle, particle
        lidar_samples = np.zeros((N, num_obs, num_samples, 4))

        for t in range(N):
            for m in range(num_obs):
                mean = noisy_pred[t, m]  # shape: [4]

                # Sample only [x, y]
                mean_xy = mean[:2]                  # [x, y]
                xy_samples = np.random.multivariate_normal(mean=mean_xy, cov=self.sigma, size=num_samples)  # shape [Z, 2]

                # Repeat [psi, v] as fixed values
                psi_v = mean[2:]                    # shape [2]
                psi_v_repeat = np.tile(psi_v, (num_samples, 1))  # shape [Z, 2]

                # Concatenate to get full [x, y, psi, v]
                lidar_samples[t, m] = np.hstack((xy_samples, psi_v_repeat))  # shape [Z, 4]

        return lidar_samples
    

    def update_rice_obs_pos(self, num_obs, obs_pos):

        N = obs_pos.shape[0]  # number of time steps

        lidar_pos = obs_pos.copy()

        for i in range(num_obs):
            true_posx = obs_pos[:, i * 4] 
            true_posy = obs_pos[:, i * 4 + 1] 

            # Loop through each time step
            for t in range(N):
                samples = 20
                x_lidar = true_posx[t] + rice.rvs(self.b_lidar[0], scale=self.rice_sigma[0], size=samples) - rice.mean(self.b_lidar[0], scale=self.rice_sigma[0])
                y_lidar = true_posy[t] + rice.rvs(self.b_lidar[1], scale=self.rice_sigma[1], size=samples) - rice.mean(self.b_lidar[1], scale=self.rice_sigma[1])

                mean_x_lidar = np.mean(x_lidar)
                mean_y_lidar = np.mean(y_lidar)

                lidar_pos[t, i * 4]     = mean_x_lidar  # x
                lidar_pos[t, i * 4 + 1] = mean_y_lidar  # y

        return lidar_pos

    def samples_from_ricedis(self, num_obs, obs_pos, num_samples):

        # [N, obstacle_id, distribution_samples, state(x,y,psi,v)]
        N = obs_pos.shape[0]  # number of time steps

        lidar_noisy_measurement = self.update_rice_obs_pos(num_obs, obs_pos) # shape (N, num_obs*4)
        noisy_pred = lidar_noisy_measurement.reshape(N, num_obs, 4)  # → [N, M, 4]

        # Output: expanded samples per timestep, obstacle, particle
        lidar_samples = np.zeros((N, num_obs, num_samples, 4))

        for t in range(N):
            for m in range(num_obs):
                mean = noisy_pred[t, m]  # shape: [4]

                # Sample only [x, y]
                mean_xy = mean[:2]                  

                # Generate zero-mean Rician noise
                noise_x = rice.rvs(self.b_lidar[0], scale=self.rice_sigma[0], size=num_samples) - rice.mean(self.b_lidar[0], scale=self.rice_sigma[0])
                noise_y = rice.rvs(self.b_lidar[0], scale=self.rice_sigma[0], size=num_samples) - rice.mean(self.b_lidar[0], scale=self.rice_sigma[0])

                # Add noise to current mean_xy
                xy_samples = mean_xy + np.stack([noise_x, noise_y], axis=-1)  # shape [num_samples, 2]

                # Repeat [psi, v] as fixed values
                psi_v = mean[2:]                    # shape [2]
                psi_v_repeat = np.tile(psi_v, (num_samples, 1))  # shape [Z, 2]

                # Concatenate to get full [x, y, psi, v]
                lidar_samples[t, m] = np.hstack((xy_samples, psi_v_repeat))  # shape [Z, 4]

        return lidar_samples

    def set_distribution(self, std_x, std_y, correlation):

        covariance_matrix = np.array([[std_x**2, correlation * std_x * std_y], 
                        [correlation * std_x * std_y, std_y**2]])
        
        return covariance_matrix



class CAMERA(Sensor):

    def __init__(self, sensor_type, sensor_info):

        super().__init__()
        self.type = sensor_type.lower()
        self.weight = sensor_info.get("weight", None)
        self.mean_x = sensor_info.get("mean_x", None)
        self.mean_y = sensor_info.get("mean_y", None)
        std_x = sensor_info.get("std_x", None)
        std_y = sensor_info.get("std_y", None)
        correlation = sensor_info.get("correlation", None)
        self.sigma = self.set_distribution(std_x, std_y, correlation)

        self.nu = np.array(sensor_info.get("nu", None))
        self.rice_sigma = np.array(sensor_info.get("sigma", None))
        self.b_camera = self.nu / self.rice_sigma

        self.scale_laplace = np.array(sensor_info.get("scale_laplace", None))

        print('Done CAMERA initialize, weight')

    def update_obs_pos(self, num_obs, obs_pos):

        N = obs_pos.shape[0]  

        camera_pos = obs_pos.copy()

        for i in range(num_obs):
            mean_posx = obs_pos[:, i * 4] + self.mean_x
            mean_posy = obs_pos[:, i * 4 + 1] +self.mean_y

            # Loop through each time step
            for t in range(N):
                mean_pos = np.array([mean_posx[t], mean_posy[t]])  # shape (2,)
                noisy_pos = np.random.multivariate_normal(mean=mean_pos, cov=self.sigma)

                camera_pos[t, i * 4]     = noisy_pos[0]  # x
                camera_pos[t, i * 4 + 1] = noisy_pos[1]  # y

        return camera_pos  

    def samples_from_dis(self, num_obs, obs_pos, num_samples):

        # [N, obstacle_id, distribution_samples, state(x,y,psi,v)]
        N = obs_pos.shape[0]  # number of time steps

        camera_noisy_measurement = self.update_obs_pos(num_obs, obs_pos) # shape (N, num_obs*4)
        noisy_pred = camera_noisy_measurement.reshape(N, num_obs, 4)  # → [N, M, 4]

        # Output: expanded samples per timestep, obstacle, particle
        camera_samples = np.zeros((N, num_obs, num_samples, 4))

        for t in range(N):
            for m in range(num_obs):
                mean = noisy_pred[t, m]  # shape: [4]
                
                # Sample only [x, y]
                mean_xy = mean[:2]                  # [x, y]
                xy_samples = np.random.multivariate_normal(mean=mean_xy, cov=self.sigma, size=num_samples)  # shape [Z, 2]

                # Repeat [psi, v] as fixed values
                psi_v = mean[2:]                    # shape [2]
                psi_v_repeat = np.tile(psi_v, (num_samples, 1))  # shape [Z, 2]

                # Concatenate to get full [x, y, psi, v]
                camera_samples[t, m] = np.hstack((xy_samples, psi_v_repeat))  # shape [Z, 4]

        return camera_samples


    def set_distribution(self, std_x, std_y, correlation):

        covariance_matrix = np.array([[std_x**2, correlation * std_x * std_y], 
                        [correlation * std_x * std_y, std_y**2]])
        
        return covariance_matrix

  

    def update_rice_obs_pos(self, num_obs, obs_pos):

        N = obs_pos.shape[0]  # number of time steps

        camera_pos = obs_pos.copy()

        for i in range(num_obs):
            true_posx = obs_pos[:, i * 4] 
            true_posy = obs_pos[:, i * 4 + 1] 

            # Loop through each time step
            for t in range(N):
                samples = 20
                x_camera = true_posx[t] + rice.rvs(self.b_camera[0], scale=self.rice_sigma[0], size=samples) - rice.mean(self.b_camera[0], scale=self.rice_sigma[0])
                y_camera = true_posy[t] + rice.rvs(self.b_camera[1], scale=self.rice_sigma[1], size=samples) - rice.mean(self.b_camera[1], scale=self.rice_sigma[1])

                mean_x_camera = np.mean(x_camera)
                mean_y_camera = np.mean(y_camera)

                camera_pos[t, i * 4]     = mean_x_camera  # x
                camera_pos[t, i * 4 + 1] = mean_y_camera  # y

        return camera_pos

    def samples_from_ricedis(self, num_obs, obs_pos, num_samples):

        # [N, obstacle_id, distribution_samples, state(x,y,psi,v)]
        N = obs_pos.shape[0]  # number of time steps

        camera_noisy_measurement = self.update_rice_obs_pos(num_obs, obs_pos) # shape (N, num_obs*4)
        noisy_pred = camera_noisy_measurement.reshape(N, num_obs, 4)  # → [N, M, 4]

        # Output: expanded samples per timestep, obstacle, particle
        camera_samples = np.zeros((N, num_obs, num_samples, 4))

        for t in range(N):
            for m in range(num_obs):
                mean = noisy_pred[t, m]  # shape: [4]

                # Sample only [x, y]
                mean_xy = mean[:2]                  

                # Generate zero-mean Rician noise
                noise_x = rice.rvs(self.b_camera[0], scale=self.rice_sigma[0], size=num_samples) - rice.mean(self.b_camera[0], scale=self.rice_sigma[0])
                noise_y = rice.rvs(self.b_camera[0], scale=self.rice_sigma[0], size=num_samples) - rice.mean(self.b_camera[0], scale=self.rice_sigma[0])

                # Add noise to current mean_xy
                xy_samples = mean_xy + np.stack([noise_x, noise_y], axis=-1)  # shape [num_samples, 2]

                # Repeat [psi, v] as fixed values
                psi_v = mean[2:]                    # shape [2]
                psi_v_repeat = np.tile(psi_v, (num_samples, 1))  # shape [Z, 2]

                # Concatenate to get full [x, y, psi, v]
                camera_samples[t, m] = np.hstack((xy_samples, psi_v_repeat))  # shape [Z, 4]

        return camera_samples


    def update_lapalce_obs_pos(self, num_obs, obs_pos):

        N = obs_pos.shape[0]  # number of time steps

        camera_pos = obs_pos.copy()

        for i in range(num_obs):
            true_posx = obs_pos[:, i * 4] 
            true_posy = obs_pos[:, i * 4 + 1] 

            # Loop through each time step
            for t in range(N):
                samples = 20

                x_camera = laplace.rvs(loc=true_posx[t], scale=self.scale_laplace[0], size=samples)
                y_camera = laplace.rvs(loc=true_posy[t], scale=self.scale_laplace[1], size=samples)

                mean_x_camera = np.mean(x_camera)
                mean_y_camera = np.mean(y_camera)

                camera_pos[t, i * 4]     = mean_x_camera  # x
                camera_pos[t, i * 4 + 1] = mean_y_camera  # y

        return camera_pos

    def samples_from_laplacedis(self, num_obs, obs_pos, num_samples):

        # [N, obstacle_id, distribution_samples, state(x,y,psi,v)]
        N = obs_pos.shape[0]  # number of time steps

        camera_noisy_measurement = self.update_lapalce_obs_pos(num_obs, obs_pos) # shape (N, num_obs*4)
        noisy_pred = camera_noisy_measurement.reshape(N, num_obs, 4)  # → [N, M, 4]

        # Output: expanded samples per timestep, obstacle, particle
        camera_samples = np.zeros((N, num_obs, num_samples, 4))

        for t in range(N):
            for m in range(num_obs):
                mean = noisy_pred[t, m]  # shape: [4]

                # Sample only [x, y]
                mean_xy = mean[:2]                  

                noise_x = laplace.rvs(loc=mean_xy[0], scale=self.scale_laplace[0], size=num_samples)
                noise_y = laplace.rvs(loc=mean_xy[1], scale=self.scale_laplace[1], size=num_samples)

                # mean_x_camera = np.mean(noise_x)
                # mean_y_camera = np.mean(noise_y)

                # Add noise to current mean_xy
                xy_samples =  np.stack([noise_x, noise_y], axis=1)

                # Repeat [psi, v] as fixed values
                psi_v = mean[2:]                    # shape [2]
                psi_v_repeat = np.tile(psi_v, (num_samples, 1))  # shape [Z, 2]
                # Concatenate to get full [x, y, psi, v]
                camera_samples[t, m] = np.hstack((xy_samples, psi_v_repeat))  # shape [Z, 4]

        return camera_samples


class V2X(Sensor):

    def __init__(self, sensor_type, sensor_info):

        super().__init__()
        self.type = sensor_type.lower()
        self.weight = sensor_info.get("weight", None)
        self.mean_x = sensor_info.get("mean_x", None)
        self.mean_y = sensor_info.get("mean_y", None)
        std_x = sensor_info.get("std_x", None)
        std_y = sensor_info.get("std_y", None)
        correlation = sensor_info.get("correlation", None)
        self.sigma = self.set_distribution(std_x, std_y, correlation)

        self.nu = np.array(sensor_info.get("nu", None))
        self.rice_sigma = np.array(sensor_info.get("sigma", None))
        self.b_v2x = self.nu / self.rice_sigma

        self.shape_gamma = np.array(sensor_info.get("shape_gamma", None))
        self.scale_gamma = np.array(sensor_info.get("scale_gamma", None))

        print('Done V2X initialize')

    def update_obs_pos(self, num_obs, obs_pos):

        N = obs_pos.shape[0]  # number of time steps

        v2x_pos = obs_pos.copy()

        for i in range(num_obs):
            mean_posx = obs_pos[:, i * 4] + self.mean_x
            mean_posy = obs_pos[:, i * 4 + 1] +self.mean_y

            # Loop through each time step
            for t in range(N):
                mean_pos = np.array([mean_posx[t], mean_posy[t]])  # shape (2,)
                noisy_pos = np.random.multivariate_normal(mean=mean_pos, cov=self.sigma)

                v2x_pos[t, i * 4]     = noisy_pos[0]  # x
                v2x_pos[t, i * 4 + 1] = noisy_pos[1]  # y

        return v2x_pos

    def samples_from_dis(self, num_obs, obs_pos, num_samples):

        # [N, obstacle_id, distribution_samples, state(x,y,psi,v)]
        N = obs_pos.shape[0]  # number of time steps

        v2x_noisy_measurement = self.update_obs_pos(num_obs, obs_pos) # shape (N, num_obs*4)
        noisy_pred = v2x_noisy_measurement.reshape(N, num_obs, 4)  # → [N, M, 4]

        # Output: expanded samples per timestep, obstacle, particle
        v2x_samples = np.zeros((N, num_obs, num_samples, 4))

        for t in range(N):
            for m in range(num_obs):
                mean = noisy_pred[t, m]  # shape: [4]
                
                # Sample only [x, y]
                mean_xy = mean[:2]                  # [x, y]
                xy_samples = np.random.multivariate_normal(mean=mean_xy, cov=self.sigma, size=num_samples)  # shape [Z, 2]

                # Repeat [psi, v] as fixed values
                psi_v = mean[2:]                    # shape [2]
                psi_v_repeat = np.tile(psi_v, (num_samples, 1))  # shape [Z, 2]

                # Concatenate to get full [x, y, psi, v]
                v2x_samples[t, m] = np.hstack((xy_samples, psi_v_repeat))  # shape [Z, 4]

        return v2x_samples


    def set_distribution(self, std_x, std_y, correlation):

        covariance_matrix = np.array([[std_x**2, correlation * std_x * std_y], 
                        [correlation * std_x * std_y, std_y**2]])
        
        return covariance_matrix
    

    def update_rice_obs_pos(self, num_obs, obs_pos):

        N = obs_pos.shape[0]  # number of time steps

        v2x_pos = obs_pos.copy()

        for i in range(num_obs):
            true_posx = obs_pos[:, i * 4] 
            true_posy = obs_pos[:, i * 4 + 1] 

            # Loop through each time step
            for t in range(N):
                samples = 20
                x_v2x = true_posx[t] + rice.rvs(self.b_v2x[0], scale=self.rice_sigma[0], size=samples) - rice.mean(self.b_v2x[0], scale=self.rice_sigma[0])
                y_v2x = true_posy[t] + rice.rvs(self.b_v2x[1], scale=self.rice_sigma[1], size=samples) - rice.mean(self.b_v2x[1], scale=self.rice_sigma[1])

                mean_x_v2x = np.mean(x_v2x)
                mean_y_v2x = np.mean(y_v2x)

                v2x_pos[t, i * 4]     = mean_x_v2x  # x
                v2x_pos[t, i * 4 + 1] = mean_y_v2x  # y

        return v2x_pos

    def samples_from_ricedis(self, num_obs, obs_pos, num_samples):

        # [N, obstacle_id, distribution_samples, state(x,y,psi,v)]
        N = obs_pos.shape[0]  # number of time steps

        v2x_noisy_measurement = self.update_rice_obs_pos(num_obs, obs_pos) # shape (N, num_obs*4)
        noisy_pred = v2x_noisy_measurement.reshape(N, num_obs, 4)  # → [N, M, 4]

        # Output: expanded samples per timestep, obstacle, particle
        v2x_samples = np.zeros((N, num_obs, num_samples, 4))

        for t in range(N):
            for m in range(num_obs):
                mean = noisy_pred[t, m]  # shape: [4]

                # Sample only [x, y]
                mean_xy = mean[:2]                  

                # Generate zero-mean Rician noise
                noise_x = rice.rvs(self.b_v2x[0], scale=self.rice_sigma[0], size=num_samples) - rice.mean(self.b_v2x[0], scale=self.rice_sigma[0])
                noise_y = rice.rvs(self.b_v2x[0], scale=self.rice_sigma[0], size=num_samples) - rice.mean(self.b_v2x[0], scale=self.rice_sigma[0])

                # Add noise to current mean_xy
                xy_samples = mean_xy + np.stack([noise_x, noise_y], axis=-1)  # shape [num_samples, 2]

                # Repeat [psi, v] as fixed values
                psi_v = mean[2:]                    # shape [2]
                psi_v_repeat = np.tile(psi_v, (num_samples, 1))  # shape [Z, 2]

                # Concatenate to get full [x, y, psi, v]
                v2x_samples[t, m] = np.hstack((xy_samples, psi_v_repeat))  # shape [Z, 4]

        return v2x_samples


    def update_gamma_obs_pos(self, num_obs, obs_pos):

        N = obs_pos.shape[0]  # number of time steps

        v2x_pos = obs_pos.copy()

        for i in range(num_obs):
            true_posx = obs_pos[:, i * 4] 
            true_posy = obs_pos[:, i * 4 + 1] 

            # Loop through each time step
            for t in range(N):
                samples = 20

                x_v2x = true_posx[t] + gamma.rvs(self.shape_gamma[0], scale=self.scale_gamma[0], size=samples) - gamma.mean(self.shape_gamma[0], scale=self.scale_gamma[0])
                y_v2x = true_posy[t] + gamma.rvs(self.shape_gamma[1], scale=self.scale_gamma[1], size=samples) - gamma.mean(self.shape_gamma[1], scale=self.scale_gamma[1])
                
                mean_x_v2x = np.mean(x_v2x)
                mean_y_v2x = np.mean(y_v2x)

                v2x_pos[t, i * 4]     = mean_x_v2x  # x
                v2x_pos[t, i * 4 + 1] = mean_y_v2x  # y

        return v2x_pos

    def samples_from_gammadis(self, num_obs, obs_pos, num_samples):

        # [N, obstacle_id, distribution_samples, state(x,y,psi,v)]
        N = obs_pos.shape[0]  # number of time steps

        v2x_noisy_measurement = self.update_gamma_obs_pos(num_obs, obs_pos) # shape (N, num_obs*4)
        noisy_pred = v2x_noisy_measurement.reshape(N, num_obs, 4)  # → [N, M, 4]

        # Output: expanded samples per timestep, obstacle, particle
        v2x_samples = np.zeros((N, num_obs, num_samples, 4))

        for t in range(N):
            for m in range(num_obs):
                mean = noisy_pred[t, m]  # shape: [4]

                # Sample only [x, y]
                mean_xy = mean[:2] -1
 
                noise_x =  gamma.rvs(self.shape_gamma[0], scale=self.scale_gamma[0], size=num_samples) - gamma.mean(self.shape_gamma[0], scale=self.scale_gamma[0])
                noise_y =  gamma.rvs(self.shape_gamma[1], scale=self.scale_gamma[1], size=num_samples) - gamma.mean(self.shape_gamma[1], scale=self.scale_gamma[1])
                
                # Add noise to current mean_xy
                xy_samples = mean_xy + np.stack([noise_x, noise_y], axis=-1)  # shape [num_samples, 2]

                # Repeat [psi, v] as fixed values
                psi_v = mean[2:]                    # shape [2]
                psi_v_repeat = np.tile(psi_v, (num_samples, 1))  # shape [Z, 2]

                # Concatenate to get full [x, y, psi, v]
                v2x_samples[t, m] = np.hstack((xy_samples, psi_v_repeat))  # shape [Z, 4]

        return v2x_samples