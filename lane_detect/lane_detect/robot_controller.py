#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist

import numpy as np
import cv2
from cv_bridge import CvBridge


class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.get_logger().info("RobotController Node Started (ROS 2 Foxy)")

        self.bridge = CvBridge()
        self.latest_image = None

        # Publishers & Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.image_callback,
            10
        )

        # Line Segment Detector
        self.LSD = cv2.createLineSegmentDetector(0)

        # 반복 제어 루프 (10Hz)
        self.timer = self.create_timer(0.1, self.control_loop)

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imshow("Original Image", self.latest_image)
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            self.latest_image = None

    def control_loop(self):
        if self.latest_image is None:
            return

        hsv = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2HSV)

        # ROI (하단 영역 추출)
        crop_L = hsv[0:240, 0:160]
        crop_R = hsv[0:240, 160:320]

        # 색상 마스크 (개선된 HSV 범위 적용)
        # 왼쪽 ROI - 주로 노란색 차선
        L_mask = cv2.inRange(crop_L, (10, 50, 80), (30, 255, 255))  # 노란색
        L2_mask = cv2.inRange(crop_L, (0, 0, 200), (255, 255, 255))  # 흰색

        # 오른쪽 ROI - 주로 흰색 차선
        R_mask = cv2.inRange(crop_R, (0, 0, 200), (255, 255, 255))  # 흰색
        R2_mask = cv2.inRange(crop_R, (10, 50, 80), (30, 255, 255))  # 노란색


        # 직선 검출
        yello_line = self.LSD.detect(L_mask)
        yello_line2 = self.LSD.detect(L2_mask)
        white_line = self.LSD.detect(R_mask)
        white_line2 = self.LSD.detect(R2_mask)

        # 주행 제어
        twist = Twist()
        if yello_line[0] is None and yello_line2[0] is None:
            twist.linear.x = 0.05
            twist.angular.z = 0.3  # 왼쪽 차선 없음
        elif white_line[0] is None and white_line2[0] is None:
            twist.linear.x = 0.05
            twist.angular.z = -0.3  # 오른쪽 차선 없음
        else:
            twist.linear.x = 0.05
            twist.angular.z = 0.0

        self.cmd_vel_pub.publish(twist)

        # 디버깅 이미지 보기
        cv2.imshow("Left ROI", crop_L)
        cv2.imshow("Right ROI", crop_R)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
