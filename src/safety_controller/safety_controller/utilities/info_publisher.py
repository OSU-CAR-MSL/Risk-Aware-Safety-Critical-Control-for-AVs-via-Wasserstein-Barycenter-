import rclpy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
import json
import numpy as np	

class InformationPublisher:
        
    def __init__(self):

        self.node = rclpy.create_node('waypoints_mpc_follower')

        # Publisher 1 for trajectory data
        self.trajectory_publisher = self.node.create_publisher(
            Float64MultiArray, 'trajectory_data', 10
        )

        self.config_publisher = self.node.create_publisher(
            String, 'config_data', 10
        )

        self.rtdict_publisher = self.node.create_publisher(
            String, 'rtdict_data', 10
        )

        # Publisher 3 for control data
        self.mpc_publisher = self.node.create_publisher(
            String, 'mpc_data', 10
        )


    def dict_msg(self, data):
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
            elif isinstance(value, list):
                data[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]


        # Serialize the dictionary to a JSON string
        msg = String()
        # msg.data = json.dumps(data)
        msg.data = json.dumps(data, default=str)  # Quick fix
        return msg
