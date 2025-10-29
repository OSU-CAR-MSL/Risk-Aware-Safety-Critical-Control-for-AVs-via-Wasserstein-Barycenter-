from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="cooperative_gcs",
                executable="User_Interface",
                name="user_interface_node",
                output="screen",
                emulate_tty=True,
            )
        ]
    )
