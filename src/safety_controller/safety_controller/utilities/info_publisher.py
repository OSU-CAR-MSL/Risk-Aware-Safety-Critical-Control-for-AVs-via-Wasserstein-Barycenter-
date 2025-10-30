import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
import json
import numpy as np


class InformationPublisher:

    def __init__(self, vehicle_id, node: Node):

        self.node = node

        # QoS: trajectory/control data use reliable best-effort with small history
        qos_transient = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        qos_volatile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # Publisher 1 for trajectory data
        self.trajectory_publisher = self.node.create_publisher(
            Float64MultiArray, "trajectory_data", qos_transient
        )

        self.config_publisher = self.node.create_publisher(
            String, "config_data", qos_transient
        )

        namespace = f"/vehicle_{vehicle_id}"
        self.rtdict_publisher = self.node.create_publisher(
            String, f"{namespace}/rtdict_data", qos_volatile
        )

        # Publisher 3 for control data
        self.mpc_publisher = self.node.create_publisher(
            String, f"{namespace}/mpc_data", qos_volatile
        )

    def dict_msg(self, data):
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
            elif isinstance(value, list):
                data[key] = [
                    v.tolist() if isinstance(v, np.ndarray) else v for v in value
                ]

        # Serialize the dictionary to a JSON string
        msg = String()
        # msg.data = json.dumps(data)
        msg.data = json.dumps(data, default=str)  # Quick fix
        return msg
