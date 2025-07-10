from setuptools import setup

package_name = 'turtlebot_tracking'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/turtlebot_tracking']),
        ('share/turtlebot_tracking', ['package.xml']),
        ('share/turtlebot_tracking/launch', ['launch/turtlebot_tracking_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jiho',
    maintainer_email='jiho@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_tracker = turtlebot_tracking.yolo_deepsort_tracker:main',
        ],
    },
)
