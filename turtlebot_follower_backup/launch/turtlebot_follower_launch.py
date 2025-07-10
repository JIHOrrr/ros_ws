from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='turtlebot_follower',
            executable='yolo_lidar_follower',
            namespace='bot2',
            output='screen'
        )
    ])
