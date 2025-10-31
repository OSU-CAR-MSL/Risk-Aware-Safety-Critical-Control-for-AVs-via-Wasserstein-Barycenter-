from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='noised_sensors',
            executable='sensor_wb_node',
            name='sensor_wb_node',
            output='screen',
        ),
    ])
