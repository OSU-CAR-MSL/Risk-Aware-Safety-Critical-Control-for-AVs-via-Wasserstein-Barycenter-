#!/usr/bin/env python3
"""ROS2 rclpy node placeholder for sensor + WB processing.
Subscribes to PointCloud2 topics and publishes a Float64MultiArray placeholder.
"""

import os
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import PointCloud2
from ament_index_python.packages import get_package_share_directory
from common.utilities.configprocessor import ConfigProcessor
from common.utilities import trajectory as tr
from common.scenario import scenarios
from noised_sensors.sensor import sensors, wasserstein_barycenter
from safety_msgs.msg import ObstacleStateList, ObstacleState
import numpy as np


class SensorWBNode(Node):
    def __init__(self):
        super().__init__("noised_sensor_wb_node")
        # Subscribers
        self.create_subscription(
            Float64MultiArray,
            "/true_obstacle_state",
            self.TrueObstacleStateCallback,
            qos_profile=QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
            ),
        )

        # # Publisher
        # self.wb_pub = self.create_publisher(Float64MultiArray, "/safety/wb", 10)
        # # Timer to publish placeholder at 10 Hz
        # self.create_timer(0.1, self.timer_callback)

        # Load default config from the common package's utilities
        config_path = os.path.join(
            get_package_share_directory("common"), "utilities", "config_IFAC.yaml"
        )
        with open(config_path, "r") as f:
            config_file = yaml.safe_load(f)

        # Initialize ConfigProcessor with loaded config
        self.config = ConfigProcessor(config_file)
        self.get_logger().info(
            f"ConfigProcessor initialized with config from: {config_path}"
        )

        self.traj = tr.Trajectory(self.config.trajectory_params)
        scen = scenarios.Scenario.create_scenarios(self.traj, self.config)

        self.sensors_list = sensors.Sensor.create_sensors(
            self.config
        )  # Pass sensor_type individually
        self.wb_obj = wasserstein_barycenter.WB_distribution()

        # self.obs = sce

        self.sample_nums = self.config["GPS"]["samples"]

    def TrueObstacleStateCallback(self, msg: ObstacleStateList):
        self.numobs = msg.numofobs
        if self.numobs != 1:
            self.get_logger().error(
                f"Expected 1 obstacle, but got {self.numobs} obstacles."
            )
            return
        true_obs_pos = [
            [
                msg.obstacle_state_list[0].x,
                msg.obstacle_state_list[0].y,
                msg.obstacle_state_list[0].yaw,
                msg.obstacle_state_list[0].v,
            ]
            for _ in range(self.config["mpc para"]["N"] + 1)
        ]
        obs_newstates = true_obs_pos.copy()

        lidar_data = self.sensors_list["lidar"].samples_from_ricedis(
            self.numobs, obs_newstates, self.sample_nums
        )
        lidar_mean = np.mean(lidar_data[0, 0, :, :], axis=0)

        camera_data = self.sensors_list["camera"].samples_from_laplacedis(
            self.numobs, obs_newstates, self.sample_nums
        )
        camera_mean = np.mean(camera_data[0, 0, :, :], axis=0)

        v2x_data = self.sensors_list["v2x"].samples_from_gammadis(
            self.numobs, obs_newstates, self.sample_nums
        )
        v2x_mean = np.mean(v2x_data[0, 0, :, :], axis=0)

        x_sensor = [lidar_data, camera_data, v2x_data]
        b_sensor = [
            self.sensors_list["lidar"].weight,
            self.sensors_list["camera"].weight,
            self.sensors_list["v2x"].weight,
        ]

        # WB obstacle position for CVaR
        opti_meas = self.wb_obj.get_freesupport_wb(x_sensor, b_sensor)
        opti_meas = np.squeeze(opti_meas, axis=0)  # shape = (10, 2)

        # comment out this when CVaR implement
        # opti_meas = np.stack([lidar_mean, camera_mean, v2x_mean], axis=0)
        ##############################
        # mean obstacle position for QP
        obs_stacked = np.stack([lidar_mean, camera_mean, v2x_mean], axis=0)
        mean_obs_pos = np.mean(obs_stacked, axis=0)
        obs_newstates[:, :2] = mean_obs_pos[:2]


def main(args=None):
    rclpy.init(args=args)
    node = SensorWBNode()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == "__main__":
    main()
