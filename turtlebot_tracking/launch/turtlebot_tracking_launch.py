from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='turtlebot_tracking',
            executable='yolo_tracker',
            namespace='bot2',
            output='screen'
        )
    ])
