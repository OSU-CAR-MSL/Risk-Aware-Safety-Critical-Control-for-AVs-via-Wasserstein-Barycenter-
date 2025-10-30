from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    share_dir = get_package_share_directory("safety_controller")
    # Run from share/safety_controller so relative paths like 'utilities/config_IFAC.yaml' and 'trajectory_data/*.csv' work
    cwd = os.path.join(share_dir)

    node = Node(
        package="safety_controller",
        executable="simulation",
        name="simulation_node",
        output="screen",
        emulate_tty=True,
        cwd=cwd,
        # Export chosen config to the node process as an environment variable
        # env={"SAFETY_CONFIG": config_arg},
    )

    return LaunchDescription([node])
