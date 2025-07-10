from setuptools import setup, find_packages

package_name = 'turtlebot_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),  # 현재 디렉토리 기준으로 패키지 탐색
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/turtlebot_follower_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jiho',
    maintainer_email='jiho@todo.todo',
    description='YOLO + Lidar로 터틀봇 추종',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_lidar_follower = turtlebot_follower.yolo_lidar_follower:main',
        ],
    },
)
