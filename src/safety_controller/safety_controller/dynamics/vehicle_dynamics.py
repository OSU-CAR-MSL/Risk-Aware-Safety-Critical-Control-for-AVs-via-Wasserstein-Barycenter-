import numpy as np
from abc import ABC, abstractmethod

class VehicleDynamics(ABC):  # Inherit from ABC
    def __init__(self, test_type, L_F, L_R):
        self.L_F = L_F
        self.L_R = L_R
        self.dynamics_type = test_type

    @abstractmethod
    def compute_dynamics(self, states, inputs, sampling_time):
        """Abstract method for computing dynamics. To be implemented by subclasses."""
        pass

    
    @staticmethod
    def create_dynamics(all_para):
        """
        Factory method to create a dynamics object based on the specified type.

        Args:
            dynamics_type (str): Type of dynamics ("Euler", "RK2", "RK4").
            L_F (float): Distance from CoG to front axle.
            L_R (float): Distance from CoG to rear axle.

        Returns:
            VehicleDynamics: An instance of the selected dynamics class.
        """

        L_F = all_para.vehicle_params.get("L_F")
        L_R = all_para.vehicle_params.get("L_R")
        dynamics_type = all_para.dynamics_params

        dynamics_map = {
            "Euler": EulerDynamics,
            "RK2": RK2Dynamics,
            "RK4": RK4Dynamics,
        }
        for type in dynamics_type:
            test_type = type

        print('dynamics_type', test_type)
        if test_type not in dynamics_map:
            raise ValueError(f"Unknown dynamics type: {test_type}")

        return dynamics_map[test_type](test_type, L_F, L_R)

class EulerDynamics(VehicleDynamics):

    def __init__(self, test_type, L_F, L_R):
        # Call the parent constructor
        super().__init__(test_type, L_F, L_R)

    def compute_dynamics(self, states, inputs, sampling_time):
        x, y, psi, v = states
        acc, steering = inputs

        beta = np.arctan(self.L_R / (self.L_F + self.L_R) * np.tan(steering))

        # Euler integration
        xn = x + sampling_time * (v * np.cos(psi + beta))
        yn = y + sampling_time * (v * np.sin(psi + beta))
        psin = psi + sampling_time * ((v / self.L_R) * np.sin(beta))
        vn = v + sampling_time * acc

        return [xn, yn, psin, vn]

class RK2Dynamics(VehicleDynamics):

    def __init__(self, test_type, L_F, L_R):
        # Call the parent constructor
        super().__init__(test_type, L_F, L_R)

    def compute_dynamics(self, states, inputs, sampling_time):
        x, y, psi, v = states
        acc, steering = inputs

        beta = np.arctan(self.L_R / (self.L_F + self.L_R) * np.tan(steering))

        # k1
        k1_x = v * np.cos(psi + beta)
        k1_y = v * np.sin(psi + beta)
        k1_psi = (v / self.L_R) * np.sin(beta)
        k1_v = acc

        # k2
        x_mid = x + sampling_time * k1_x
        y_mid = y + sampling_time * k1_y
        psi_mid = psi + sampling_time * k1_psi
        v_mid = v + sampling_time * k1_v

        k2_x = v_mid * np.cos(psi_mid + beta)
        k2_y = v_mid * np.sin(psi_mid + beta)
        k2_psi = (v_mid / self.L_R) * np.sin(beta)
        k2_v = acc

        # RK2 integration
        xn = x + (sampling_time / 2) * (k1_x + k2_x)
        yn = y + (sampling_time / 2) * (k1_y + k2_y)
        psin = psi + (sampling_time / 2) * (k1_psi + k2_psi)
        vn = v + (sampling_time / 2) * (k1_v + k2_v)

        return [xn, yn, psin, vn]

class RK4Dynamics(VehicleDynamics):

    def __init__(self, test_type, L_F, L_R):
        # Call the parent constructor
        super().__init__(test_type, L_F, L_R)
    
    def compute_dynamics(self, states, inputs, sampling_time):
        x, y, psi, v = states
        acc, steering = inputs

        beta = np.arctan(self.L_R / (self.L_F + self.L_R) * np.tan(steering))

        # k1
        k1_x = v * np.cos(psi + beta)
        k1_y = v * np.sin(psi + beta)
        k1_psi = (v / self.L_R) * np.sin(beta)
        k1_v = acc

        # k2
        x_k2 = x + (sampling_time / 2) * k1_x
        y_k2 = y + (sampling_time / 2) * k1_y
        psi_k2 = psi + (sampling_time / 2) * k1_psi
        v_k2 = v + (sampling_time / 2) * k1_v

        k2_x = v_k2 * np.cos(psi_k2 + beta)
        k2_y = v_k2 * np.sin(psi_k2 + beta)
        k2_psi = (v_k2 / self.L_R) * np.sin(beta)
        k2_v = acc

        # k3
        x_k3 = x + (sampling_time / 2) * k2_x
        y_k3 = y + (sampling_time / 2) * k2_y
        psi_k3 = psi + (sampling_time / 2) * k2_psi
        v_k3 = v + (sampling_time / 2) * k2_v

        k3_x = v_k3 * np.cos(psi_k3 + beta)
        k3_y = v_k3 * np.sin(psi_k3 + beta)
        k3_psi = (v_k3 / self.L_R) * np.sin(beta)
        k3_v = acc

        # k4
        x_k4 = x + sampling_time * k3_x
        y_k4 = y + sampling_time * k3_y
        psi_k4 = psi + sampling_time * k3_psi
        v_k4 = v + sampling_time * k3_v

        k4_x = v_k4 * np.cos(psi_k4 + beta)
        k4_y = v_k4 * np.sin(psi_k4 + beta)
        k4_psi = (v_k4 / self.L_R) * np.sin(beta)
        k4_v = acc

        # RK4 integration
        xn = x + (sampling_time / 6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        yn = y + (sampling_time / 6) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
        psin = psi + (sampling_time / 6) * (k1_psi + 2 * k2_psi + 2 * k3_psi + k4_psi)
        vn = v + (sampling_time / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        return [xn, yn, psin, vn]
