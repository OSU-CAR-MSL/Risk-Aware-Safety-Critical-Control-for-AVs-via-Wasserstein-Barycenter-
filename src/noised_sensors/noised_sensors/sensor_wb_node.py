#!/usr/bin/env python3
"""ROS2 rclpy node placeholder for sensor + WB processing.
Subscribes to PointCloud2 topics and publishes a Float64MultiArray placeholder.
"""

import os
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import PointCloud2
from ament_index_python.packages import get_package_share_directory
from common.utilities.configprocessor import ConfigProcessor
from common.utilities import trajectory as tr
from common.scenario import scenarios
from noised_sensors.sensor import sensors, wasserstein_barycenter
from safety_msgs.msg import ObstacleStateList
import numpy as np


class SensorWBNode(Node):
    def __init__(self):
        super().__init__("noised_sensor_wb_node")
        # Subscribers
        self.create_subscription(
            ObstacleStateList,
            "/true_obstacle_state",
            self.TrueObstacleStateCallback,
            qos_profile=QoSProfile(
                depth=10,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
            ),
        )
        self.mean_noised_obstacle_state_pub = self.create_publisher(
            ObstacleStateList, "/mean_noised_obstacle_state", 10
        )
        self.opti_meas_pub = self.create_publisher(
            Float64MultiArray, "/opti_meas_noised_obstacle_state", 10
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
        self.sample_nums = self.config.config["vehicle"]["GPS"]["samples"]

    def TrueObstacleStateCallback(self, msg: ObstacleStateList):
        print(f"Received TrueObstacleState: {msg}")
        self.numobs = msg.numofobs
        self.num_of_time_step = msg.num_of_time_step
        if self.numobs != 1:
            self.get_logger().error(
                f"Expected 1 obstacle, but got {self.numobs} obstacles."
            )
            return

        true_obs_pos = np.asarray(msg.obstacle_data, dtype=float).reshape(
            [self.num_of_time_step, self.numobs * 4]
        )
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

        # Publish mean noised obstacle state
        mean_noised_msg = ObstacleStateList()
        mean_noised_msg.header.stamp = self.get_clock().now().to_msg()
        mean_noised_msg.numofobs = self.numobs
        mean_noised_msg.num_of_time_step = self.num_of_time_step
        mean_noised_msg.obstacle_data = obs_newstates.flatten().tolist()
        self.mean_noised_obstacle_state_pub.publish(mean_noised_msg)

        # Publish opti_meas noised obstacle state
        opti_meas_msg = Float64MultiArray()
        opti_meas_msg.data = opti_meas.flatten().tolist()
        self.opti_meas_pub.publish(opti_meas_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SensorWBNode()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == "__main__":
    main()
