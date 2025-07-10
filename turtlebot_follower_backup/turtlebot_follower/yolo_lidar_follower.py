import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import math

def iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return interArea / float(box1Area + box2Area - interArea + 1e-6)

class TurtlebotFollower(Node):
    def __init__(self):
        super().__init__('turtlebot_follower')
        self.bridge = CvBridge()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model_turtlebot = YOLO('/home/jiho/robot_ws/src/runs/detect/train4/weights/best.pt').to(device)

        self.get_logger().info(f"📦 YOLO 모델 로딩 완료 ({device.upper()})")

        self.cmd_pub = self.create_publisher(Twist, '/bot2/cmd_vel', 10)
        self.create_subscription(CompressedImage, '/bot2/image/compressed', self.image_callback, 10)
        self.create_subscription(LaserScan, '/bot2/scan', self.lidar_callback, qos_profile_sensor_data)

        self.detected = False
        self.distance = float('inf')
        self.offset_x = 0.0  # 객체 중심 offset

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        result_tb = self.model_turtlebot.predict(
            source=frame, imgsz=640, conf=0.25, verbose=False
        )[0]

        self.detected = False
        self.offset_x = 0.0  # 초기화

        target_offset = None
        min_center_diff = float('inf')

        for box in result_tb.boxes:
            cls_id = int(box.cls[0])
            label = self.model_turtlebot.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # 중심 좌표 계산
            center_x = (x1 + x2) / 2
            frame_center_x = frame.shape[1] / 2
            offset = (center_x - frame_center_x) / frame_center_x  # -1 ~ 1 정규화 값

            # 제어 대상은 turtlebot만
            if label == 'turtlebot':
                center_diff = abs(offset)
                if center_diff < min_center_diff:
                    min_center_diff = center_diff
                    target_offset = offset
                    self.detected = True

            # 박스 색상 설정
            color = (0, 255, 0) if label == 'turtlebot' else (0, 0, 255)  # 초록: turtlebot, 빨강: turtlebot_front

            # 박스 및 텍스트 출력 (모든 박스)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 추적 대상 offset 저장
        if self.detected:
            self.offset_x = target_offset

        # 이미지 출력
        resized = cv2.resize(frame, (800, 600))
        cv2.imshow("YOLO Detection", resized)
        cv2.waitKey(1)

        self.control_robot()

    def lidar_callback(self, msg):
        angle_deg = 10
        half_range = int(angle_deg / (msg.angle_increment * 180.0 / math.pi))
        front_ranges = msg.ranges[-half_range:] + msg.ranges[:half_range]
        valid = [r for r in front_ranges if msg.range_min < r < msg.range_max and not math.isinf(r) and not math.isnan(r)]
        self.distance = min(valid) if valid else float('inf')
        self.control_robot()

    def control_robot(self):
        twist = Twist()
        if self.detected:
            if self.distance < 0.3:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.get_logger().info(f"🔴 감지됨 + 거리({self.distance:.2f}m) 가까움 → 정지")
            else:
                twist.linear.x = 0.05
                twist.angular.z = -0.7 * self.offset_x  # 중심 보정용 회전
                self.get_logger().info(f"🟢 감지됨 + 거리({self.distance:.2f}m) 충분 → 전진 + 회전({twist.angular.z:.2f})")
        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
