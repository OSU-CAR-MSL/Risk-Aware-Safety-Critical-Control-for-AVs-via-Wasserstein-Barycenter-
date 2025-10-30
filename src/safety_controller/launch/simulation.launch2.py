from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    share_dir = get_package_share_directory("safety_controller")
    # Run from share/safety_controller so relative paths like 'utilities/config_IFAC.yaml' and 'trajectory data/*.csv' work
    cwd = os.path.join(share_dir)

    # default config: use config_IFAC.yaml (packaged default). If missing, the node will raise an error.
    default_config = os.path.join(share_dir, "utilities", "config_IFAC.yaml")

    declare_cfg = DeclareLaunchArgument(
        "config_file",
        default_value=default_config,
        description="Path to YAML config file to use (overrides packaged defaults)",
    )

    config_arg = LaunchConfiguration("config_file")

    # Build a PYTHONPATH that includes the install site's site-packages plus
    # the current process's PYTHONPATH so the launched node can import installed
    # packages even when the ROS launch environment doesn't propagate it.
    install_site = os.path.join(share_dir, "..", "lib", "python3.10", "site-packages")
    current_pp = os.environ.get("PYTHONPATH", "")

    # Build the environment for the node by copying current env and updating
    # only the variables we need. This preserves HOME, XDG, and other system
    # variables required by rclpy and underlying C libraries.
    node_env = os.environ.copy()
    node_env.update({
        "SAFETY_CONFIG": config_arg,
        "PYTHONPATH": install_site + (":" + current_pp if current_pp else ""),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
    })

    node = Node(
        package="safety_controller",
        executable="simulation",
        name="simulation_node",
        output="screen",
        emulate_tty=True,
        cwd=cwd,
        # pass the prepared environment dict
        env=node_env,
    )

    return LaunchDescription([declare_cfg, node])
