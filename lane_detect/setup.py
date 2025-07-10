from setuptools import setup

package_name = 'lane_detect'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='woong2',
    maintainer_email='woong2@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "image_processor_node = lane_detect.image_processor_node : main",
            "lane_detector_node = lane_detect.lane_detector_node : main",
            "track_moving_node = lane_detect.track_moving_node : main",
            "lane_detector_node2 = lane_detect.lane_detector_node2 : main",
            "hsv_color_detector_node = lane_detect.hsv_color_detector : main",
            "slicingwindow_detector_node = lane_detect.slicingwindow_detector_node : main",
            "control_lane = lane_detect.control_lane : main",
            "robot_controller = lane_detect.robot_controller : main",
            "cmd_vel_selector = lane_detect.cmd_vel_selector : main",
        ],
    },
)
