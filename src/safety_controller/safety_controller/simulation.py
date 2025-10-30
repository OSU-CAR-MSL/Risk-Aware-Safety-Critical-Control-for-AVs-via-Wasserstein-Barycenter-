import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
import time
import numpy as np
import math
import multiprocessing
import yaml
from rclpy.executors import MultiThreadedExecutor

from ament_index_python.packages import get_package_share_directory
import os

# package-qualified imports so module resolves when installed
from safety_controller.utilities import trajectory as tr
from safety_controller.utilities import info_publisher
from safety_controller import vehicle_control
from safety_controller import controller as controller
from safety_controller.utilities.configprocessor import ConfigProcessor
from safety_controller.utilities import functions
from safety_controller.scenario import scenarios
from safety_controller.dynamics import vehicle_dynamics
from safety_controller.sensor import sensors, wasserstein_barycenter


current_lat = multiprocessing.Value("d", 0)  # 39.99789379865)
current_lon = multiprocessing.Value("d", 0)  # -83.03321094485)
current_yaw = multiprocessing.Value("d", 0.0)  # origin N:0 (degree)
current_velocityCAN = multiprocessing.Value("d", 0)
controller_steer = multiprocessing.Value("i", 0)
controller_acc = multiprocessing.Value("d", 0)
controller_brake = multiprocessing.Value("d", 0.0)


class SafeController:

    def __init__(self, configfile, run_time):

        self.config = ConfigProcessor(configfile)

        # for store mpc previous solution
        self.u_prev = [0, 0]
        self.acc_prev = 0
        self.steer_prev = 0

        # for initial yaw store
        self.count = 0
        self.trigger_cnt = 0

        # for initial execute time
        self.previous_etime = 0
        self.L_F = self.config.vehicle_params.get("L_F")
        self.L_R = self.config.vehicle_params.get("L_R")
        self.LHD = self.config.mpc_params.get("LHD")

        # for gps noise
        gps_config = self.config.vehicle_params["GPS"]
        self.gps_samples = gps_config["samples"]
        self.gps_mean = gps_config["mean"]
        self.gps_std = gps_config["std_dev"]

        # read trajectory and finalized local coordinate waypoints
        self.traj = tr.Trajectory(self.config.trajectory_params)
        scen = scenarios.Scenario.create_scenarios(self.traj, self.config)
        self.dynamics = vehicle_dynamics.VehicleDynamics.create_dynamics(self.config)
        for controller_type in self.config.controller_params:
            mpc = controller.get_controller(
                controller_type, self.config, scen.set_obs_init()
            )

        sensors_list = sensors.Sensor.create_sensors(
            self.config
        )  # Pass sensor_type individually
        wb_obj = wasserstein_barycenter.WB_distribution()
        self.ctrl = vehicle_control.VehicleControl(
            self.config,
            mpc,
            scen,
            current_lat,
            current_lon,
            current_yaw,
            current_velocityCAN,
            controller_steer,
            controller_acc,
            controller_brake,
            run_time,
        )
        self.ctrl._update_sensors(sensors_list)
        self.ctrl._update_wb(wb_obj)

    def indoorsimu(self):

        # indoor testing

        wps_lat = self.traj.waypoints_lla[0]
        wps_lon = self.traj.waypoints_lla[1]
        wps_alt = self.traj.waypoints_lla[2]
        wps_yaw = self.traj.waypoints_lla[9]

        run_flag = True

        while run_flag:
            # for i in range(0, len(wps_lat), 5):

            start_time = time.time()  # Record the start time
            current_lat.value = wps_lat[0]
            current_lon.value = wps_lon[0]
            current_yaw.value = wps_yaw[0]

            if (
                current_lat.value != 0.0
                and current_lon.value != 0.0
                and current_yaw.value != 0.0
            ):

                # initialize all states
                if self.count == 0:
                    # current_lat.value = wps_lat[0]
                    # current_lon.value = wps_lon[0]
                    # current_yaw.value = wps_yaw[0]
                    cur_lat = current_lat.value
                    cur_lon = current_lon.value
                    cur_yaw = current_yaw.value

                    x0, y0 = functions.lla_to_enu(
                        cur_lat, cur_lon, 229, wps_lat[0], wps_lon[0], wps_alt[0]
                    )
                    x0 = float(x0)
                    y0 = float(y0)
                    tmp = functions.N2E(cur_yaw, 0)
                    psi0 = functions.pi_2_pi(tmp)  # pi2pi
                    v0 = 0

                    # initialize ENU states matrix
                    states = [x0, y0, psi0, v0]
                    self.true_vehicle_states = states
                    self.vehicle_noisy_samples = np.zeros((10, 4))
                    self.count = self.count + 1

            if (
                current_lat.value != 0.0
                and current_lon.value != 0.0
                and current_yaw.value != 0.0
            ):

                steering, acc, brake, _acc_prev, _steer_prev = self.ctrl.run_controller(
                    self.acc_prev,
                    self.steer_prev,
                    cur_lat,
                    cur_lon,
                    cur_yaw,
                    self.previous_etime,
                    self.true_vehicle_states,
                    self.vehicle_noisy_samples,
                )
                inputs = [_acc_prev, _steer_prev]

                states = self.dynamics.compute_dynamics(
                    states, inputs, self.previous_etime
                )
                current_velocityCAN.value = states[3]

                self.true_vehicle_states = states

                # add GPS noise

                # Generate noise for x and y
                noise_x = np.random.normal(
                    self.gps_mean, self.gps_std, self.gps_samples
                )
                noise_y = np.random.normal(
                    self.gps_mean, self.gps_std, self.gps_samples
                )

                # Apply noise to x and y
                noisy_x = states[0] + noise_x
                noisy_y = states[1] + noise_y

                # Repeat psi and velocity
                psi_array = np.full(self.gps_samples, states[2])
                v_array = np.full(self.gps_samples, states[3])

                # Stack to form (10, 4) sample set
                self.vehicle_noisy_samples = np.stack(
                    (noisy_x, noisy_y, psi_array, v_array), axis=1
                )
                veh_meas_mean = np.mean(self.vehicle_noisy_samples, axis=0)
                states[0] = veh_meas_mean[0]
                states[1] = veh_meas_mean[1]

                llastates = functions.enu_to_lla(
                    states[0], states[1], wps_lat[0], wps_lon[0], wps_alt[0]
                )

                current_lat.value = llastates[0]
                current_lon.value = llastates[1]

                cur_lat = current_lat.value
                cur_lon = current_lon.value
                cur_yaw = np.degrees(functions.E2N(states[2], 1))

                # store for next calculation
                self.acc_prev = _acc_prev
                self.steer_prev = _steer_prev

            final_x, final_y = functions.lla_to_enu(
                wps_lat[len(wps_lat) - 1],
                wps_lon[len(wps_lat) - 1],
                229,
                wps_lat[0],
                wps_lon[0],
                wps_alt[0],
            )
            dis_cur_end = np.sqrt(
                (states[0] - final_x) ** 2 + (states[1] - final_y) ** 2
            )
            print("dis_cur_end", dis_cur_end)

            if dis_cur_end < self.LHD:
                # print('dis_cur_end', dis_cur_end)
                run_flag = False
            end_time = time.time()  # Record the end time
            execution_time = end_time - start_time  # Calculate the elapsed time
            self.previous_etime = execution_time
            print("Execute time:", execution_time)


class SimulatioNode(Node):
    def __init__(self):
        super().__init__("safety_controller")
        # Node initialization code here
        configfilepath = os.path.join(
            get_package_share_directory("safety_controller"),
            "utilities",
            "config_IFAC.yaml",
        )

        with open(configfilepath, "r") as file:
            self.configfile = yaml.safe_load(file)

    def run_simulation(self):

        scenario_list = self.configfile["scenarios"]
        controller_list = self.configfile["controller"]
        velocity_list = self.configfile["trajectory"]["V_REF"]
        dynamics_list = self.configfile["dynamics"]
        print("run_time", scenario_list)
        for scenario in scenario_list:
            run_time = scenario.get("run_time", 1)
            for controller_type in controller_list:
                for velocity in velocity_list:
                    for dynamics in dynamics_list:
                        for i in range(run_time):
                            print(
                                f"Running {i+1} times simulation with V={velocity}, Controller={controller_type}, Scenario={scenario}, Dynamics={dynamics}"
                            )

                            # Modify configuration dynamically
                            self.configfile["trajectory"]["V_REF"] = [velocity]
                            self.configfile["controller"] = [controller_type]
                            self.configfile["scenarios"] = [scenario]
                            self.configfile["dynamics"] = [dynamics]

                            sub = SafeController(self.configfile, i + 1)

                            infopub = info_publisher.InformationPublisher(
                                self.configfile["vehicle"].get("VEHICLE_ID", 2),
                                self,
                            )
                            sub.ctrl._update_publisher(infopub)
                            sub.indoorsimu()
                            time.sleep(1.0)


def main(args=None):

    rclpy.init(args=args)

    simulation_node = SimulatioNode()

    executor = MultiThreadedExecutor()
    executor.add_node(simulation_node)
    try:
        simulation_node.run_simulation()
    except KeyboardInterrupt:
        simulation_node.get_logger().warning("Simulation interrupted by user (Ctrl+C)")
        if rclpy.ok():
            simulation_node.destroy_node()
            rclpy.shutdown()
        return

    simulation_node.get_logger().info("Simulation Completed. Pressing Ctrl+C to exit.")
    executor.spin()


if __name__ == "__main__":
    main()
