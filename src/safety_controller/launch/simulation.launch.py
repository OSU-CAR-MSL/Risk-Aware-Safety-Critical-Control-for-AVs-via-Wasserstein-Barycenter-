from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    node = Node(
        package="safety_controller",
        executable="simulation",
        name="simulation_node",
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([node])
