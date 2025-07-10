import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import math
from deep_sort_realtime.deepsort_tracker import DeepSort

class TurtlebotTracker(Node):
    def __init__(self):
        super().__init__('turtlebot_tracker')
        self.bridge = CvBridge()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO('/home/jiho/robot_ws/src/runs/detect/train4/weights/best.pt').to(device)
        self.get_logger().info(f"📦 YOLO 모델 로딩 완료 ({device.upper()})")

        self.tracker = DeepSort(max_age=30)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_of', 10)
        self.active_pub = self.create_publisher(Bool, '/object_tracking/active', 10)

        self.create_subscription(CompressedImage, '/camera/image/compressed', self.image_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)

        self.distance = float('inf')
        self.target_id = None
        self.missing_frames = 0
        self.max_missing_frames = 10

        # EMA 관련 변수
        self.offset_x_ema = 0.0
        self.ema_alpha = 0.3

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        results = self.model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)[0]

        detections = []
        current_tracks = []
        frame_center_x = frame.shape[1] / 2

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            if label not in ['turtlebot', 'turtlebot_front']:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            bbox = [x1, y1, x2 - x1, y2 - y1]
            detections.append((bbox, conf, label))

        tracks = self.tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            center_x = (l + r) / 2
            offset = (center_x - frame_center_x) / frame_center_x
            label = track.get_det_class() if hasattr(track, 'get_det_class') else 'turtlebot'

            current_tracks.append({
                'id': track_id,
                'bbox': (l, t, r, b),
                'offset': offset,
                'label': label
            })

            color = (0, 255, 0)
            text_y = max(15, t - 10)
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, f'ID:{track_id} {label}', (l, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        target_found = False
        if self.target_id is not None:
            for trk in current_tracks:
                if trk['id'] == self.target_id:
                    target_found = True
                    self.offset_x = trk['offset']
                    break

            if target_found:
                self.missing_frames = 0
            else:
                self.missing_frames += 1
                if self.missing_frames > self.max_missing_frames:
                    self.get_logger().info(f"❌ ID {self.target_id} 추적 대상 소실, 대상 해제")
                    self.target_id = None
                    self.offset_x = 0.0
                    self.missing_frames = 0

        if self.target_id is None and current_tracks:
            turtlebot_candidates = [trk for trk in current_tracks if trk['label'] == 'turtlebot']
            if turtlebot_candidates:
                best_target = min(turtlebot_candidates, key=lambda x: abs(x['offset']))
                self.target_id = best_target['id']
                self.offset_x = best_target['offset']
                self.get_logger().info(f"🎯 추적 대상 변경: ID {self.target_id} (중앙에 가장 가까운 turtlebot)")

        resized = cv2.resize(frame, (800, 600))
        cv2.imshow("YOLO + Deep SORT", resized)
        cv2.waitKey(1)

        self.control_robot()

    def lidar_callback(self, msg):
        angle_deg = 10
        half_range = int(angle_deg / (msg.angle_increment * 180.0 / math.pi))
        front_ranges = msg.ranges[-half_range:] + msg.ranges[:half_range]
        valid = [r for r in front_ranges if msg.range_min < r < msg.range_max and not math.isinf(r) and not math.isnan(r)]
        self.distance = min(valid) if valid else float('inf')

    def control_robot(self):
        twist = Twist()

        if self.target_id is not None:
            self.active_pub.publish(Bool(data=True))

            raw_offset = self.offset_x

            # EMA 적용
            self.offset_x_ema = self.ema_alpha * raw_offset + (1 - self.ema_alpha) * self.offset_x_ema
            smooth_offset = self.offset_x_ema

            # Dead zone 설정
            dead_zone = 0.05
            if abs(smooth_offset) < dead_zone:
                smooth_offset = 0.0

            if self.distance < 0.3:
                twist.linear.x = 0.0
                twist.angular.z = -0.7 * smooth_offset
                self.get_logger().info(f"🔴 가까움({self.distance:.2f}m), 회전만 ({twist.angular.z:.2f})")
            else:
                twist.linear.x = 0.05
                twist.angular.z = -1.2 * smooth_offset
                self.get_logger().info(f"🟢 추적 중, 거리: {self.distance:.2f}m, 전진+회전({twist.angular.z:.2f})")
        else:
            self.active_pub.publish(Bool(data=False))
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.get_logger().info("⏹️ 추적 대상 없음, 정지")

        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
