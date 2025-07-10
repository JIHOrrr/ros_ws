import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import csv
import numpy as np
from ultralytics import YOLO
import torch
import math
import time
from deep_sort_realtime.deepsort_tracker import DeepSort

KNOWN_WIDTH = 0.3
FOCAL_LENGTH = 615
CORRECTION_FACTOR = 0.14

class TurtlebotTracker(Node):
    def __init__(self):
        super().__init__('turtlebot_tracker')
        self.bridge = CvBridge()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO('/home/jiho/robot_ws/src/runs/detect/train7/weights/best.pt').to(device)
        self.coco_model = YOLO('yolov8n.pt').to(device)
        self.get_logger().info(f"📦 YOLO 모델 로딩 완료 ({device.upper()})")

        self.frame_count = 0
        self.detection_stats = []
        self.fps_log = []

        self.stats_output_path = '/home/jiho/robot_ws/src/turtlebot_tracking/detection_stats.csv'
        self.fps_log_path = '/home/jiho/robot_ws/src/turtlebot_tracking/fps_log.csv'

        self.tracker = DeepSort(max_age=30)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(CompressedImage, '/camera/image/compressed', self.image_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)

        self.distance = float('inf')
        self.estimated_distance = float('inf')
        self.target_id = None
        self.missing_frames = 0
        self.max_missing_frames = 10
        self.offset_x = 0.0

        self.calibration_coef = 3.2325
        self.calibration_intercept = -0.5708
        self.poly_coeffs = [3.24003463, 0.41937413, -0.08074617]

    def apply_polynomial_correction(self, raw_distance):
        a, b, c = self.poly_coeffs
        return a * raw_distance**2 + b * raw_distance + c

    def estimate_distance_from_bbox(self, bbox):
        x1, _, x2, _ = bbox
        pixel_width = x2 - x1
        if pixel_width <= 0 or pixel_width < 20 or pixel_width > 500:
            self.get_logger().warn(f"⚠️ bbox width={pixel_width:.1f}px: 거리 추정 불가")
            return float('inf')

        raw_distance = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width
        corrected_raw = raw_distance * CORRECTION_FACTOR

        if corrected_raw < 0.2 or corrected_raw > 1.5:
            self.get_logger().warn(f"⚠️ 카메라 추정 거리 {corrected_raw:.2f}m 신뢰 불가 (raw={raw_distance:.2f})")
            return float('inf')

        corrected_poly = self.apply_polynomial_correction(corrected_raw)
        return corrected_poly

    def image_callback(self, msg):
        frame_start_time = time.time()

        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # YOLOv8 예측 시간 측정
        yolo_start = time.time()
        results = self.model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)[0]
        yolo_end = time.time()
        yolo_time = yolo_end - yolo_start

        # COCO 예측 시간 측정
        coco_start = time.time()
        coco_results = self.coco_model.predict(source=frame, imgsz=640, conf=0.3, verbose=False)[0]
        coco_end = time.time()
        coco_time = coco_end - coco_start

        detections = []
        current_tracks = []
        frame_center_x = frame.shape[1] / 2
        self.estimated_distance = float('inf')

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            if label not in ['turtlebot', 'turtlebot_front']:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            bbox = [x1, y1, x2 - x1, y2 - y1]
            detections.append((bbox, conf, label))

        # DeepSORT 추적 시간 측정
        deep_start = time.time()
        tracks = self.tracker.update_tracks(detections, frame=frame)
        deep_end = time.time()
        deep_time = deep_end - deep_start

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

            color = (0, 255, 0) if label == 'turtlebot' else (0, 0, 255)
            text_y = max(15, t - 10)
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, f'ID:{track_id} {label}', (l, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for box in coco_results.boxes:
            cls_id = int(box.cls[0])
            label = self.coco_model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 1)
            cv2.putText(frame, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)

        target_found = False
        if self.target_id is not None:
            for trk in current_tracks:
                if trk['id'] == self.target_id:
                    target_found = True
                    self.offset_x = trk['offset']
                    self.estimated_distance = self.estimate_distance_from_bbox(trk['bbox'])
                    break

            if target_found:
                self.missing_frames = 0
            else:
                self.missing_frames += 1
                if self.missing_frames > self.max_missing_frames:
                    self.get_logger().info(f"[Frame {self.frame_count}] ❌ ID {self.target_id} 추적 대상 소실 → 해제")
                    self.target_id = None
                    self.offset_x = 0.0
                    self.estimated_distance = float('inf')
                    self.missing_frames = 0

        if self.target_id is None and current_tracks:
            turtlebot_candidates = [trk for trk in current_tracks if trk['label'] == 'turtlebot']
            if turtlebot_candidates:
                best_target = min(turtlebot_candidates, key=lambda x: abs(x['offset']))
                self.target_id = best_target['id']
                self.offset_x = best_target['offset']
                self.estimated_distance = self.estimate_distance_from_bbox(best_target['bbox'])
                self.get_logger().info(f"[Frame {self.frame_count}] 🎯 추적 대상 변경 → ID {self.target_id} | 거리: {self.estimated_distance:.2f}m (CAMERA)")

        resized = cv2.resize(frame, (800, 600))
        cv2.imshow("YOLO + Deep SORT + COCO", resized)
        cv2.waitKey(1)

        self.control_robot()

        self.frame_count += 1
        detection_result = {
            'frame': self.frame_count,
            'track_id': self.target_id if self.target_id is not None else -1,
            'label': '',
            'is_true_detection': 0
        }

        if self.target_id is not None:
            for trk in current_tracks:
                if trk['id'] == self.target_id:
                    detection_result['label'] = trk['label']
                    if trk['label'] in ['turtlebot', 'turtlebot_front']:
                        detection_result['is_true_detection'] = 1
                    break

        self.detection_stats.append(detection_result)

        # 프레임 처리 시간 측정 및 로그 저장
        frame_end_time = time.time()
        total_time = frame_end_time - frame_start_time

        self.fps_log.append({
            'frame': self.frame_count,
            'yolo_time': yolo_time,
            'coco_time': coco_time,
            'deep_time': deep_time,
            'total_time': total_time
        })

    def lidar_callback(self, msg):
        angle_deg = 10
        half_range = int(angle_deg / (msg.angle_increment * 180.0 / math.pi))
        front_ranges = msg.ranges[-half_range:] + msg.ranges[:half_range]
        valid = [r for r in front_ranges if msg.range_min < r < msg.range_max and not math.isinf(r) and not math.isnan(r)]
        self.distance = min(valid) if valid else float('inf')

    def control_robot(self):
        twist = Twist()
        distance = self.distance
        distance_source = "LiDAR"

        if not (0.05 < distance < 5.0):
            est = self.estimated_distance
            if 0.2 < est < 1.5:
                distance = est
                distance_source = "CAMERA"
            else:
                self.get_logger().warn(
                    f"[Frame {self.frame_count}] ❌ 거리 인식 실패 → 정지 (LiDAR+카메라 모두 신뢰 불가)"
                )
                distance = float('inf')

        if self.target_id is not None:
            offset_x = self.offset_x
            if distance < 0.3:
                twist.linear.x = 0.0
                twist.angular.z = -0.7 * offset_x
                self.get_logger().info(
                    f"[Frame {self.frame_count}] 🔴 ID {self.target_id} 가까움 | 거리: {distance:.2f}m ({distance_source}) → 회전만({twist.angular.z:.2f})"
                )
            else:
                twist.linear.x = 0.1
                twist.angular.z = -1.2 * offset_x
                self.get_logger().info(
                    f"[Frame {self.frame_count}] 🟢 ID {self.target_id} 추적 중 | 거리: {distance:.2f}m ({distance_source}) → 전진+회전({twist.angular.z:.2f})"
                )
        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.get_logger().info(
                f"[Frame {self.frame_count}] 🟢 ID {self.target_id} 추적 중 | 거리: {distance:.2f}m ({distance_source}) → 전진+회전({twist.angular.z:.2f})"
            )

        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    with open(node.stats_output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['frame', 'track_id', 'label', 'is_true_detection'])
        writer.writeheader()
        writer.writerows(node.detection_stats)

    with open(node.fps_log_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['frame', 'yolo_time', 'coco_time', 'deep_time', 'total_time'])
        writer.writeheader()
        writer.writerows(node.fps_log)

    print(f"\n📊 detection_stats.csv 저장 완료: {node.stats_output_path}")
    print(f"📈 fps_log.csv 저장 완료: {node.fps_log_path}")

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
