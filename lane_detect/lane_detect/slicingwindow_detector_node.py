import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64, UInt8
from cv_bridge import CvBridge
import cv2
import numpy as np

class SlicingWindowDetector(Node):
    def __init__(self):
        super().__init__('slicingwindow_detector_node')
        self.get_logger().info('SlicingWindow Detector Node Initialized')

        self.bridge = CvBridge()

        self.subscription_edges = self.create_subscription(
            CompressedImage, '/processed_image/edges', self.edges_callback, 10
        )
        self.subscription_camera = self.create_subscription(
            CompressedImage, '/camera/image/compressed', self.camera_callback, 10
        )

        self.pub_lane = self.create_publisher(Float64, '/detect/lane', 1)
        self.pub_lane_state = self.create_publisher(UInt8, '/detect/lane_state', 1)
        self.pub_image_output = self.create_publisher(CompressedImage, '/detect/image_output/compressed', 1)
        self.pub_yellow_line_reliability = self.create_publisher(UInt8, '/detect/yellow_line_reliability', 1)
        self.pub_white_line_reliability = self.create_publisher(UInt8, '/detect/white_line_reliability', 1)

        self.latest_edge_image = None
        self.latest_camera_image = None

        # 차선 폭을 픽셀 단위로 가정 (환경에 따라 조정 필요)
        self.window_width = 320.0 # 기본값 : 1000
        self.window_height = 240.0 # 기본값 : 600 
        self.lane_width_pixels = 150

    def camera_callback(self, msg):
        try:
            self.latest_camera_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert camera image: {e}")

    def edges_callback(self, msg):
        try:
            self.latest_edge_image = self.bridge.compressed_imgmsg_to_cv2(msg, "mono8")
            self.process_lane()
        except Exception as e:
            self.get_logger().error(f"Failed to convert edge image: {e}")

    def compute_reliability(self, pixel_count, threshold=3000):
        return min(int((pixel_count / threshold) * 100), 100)

    def process_lane(self):
        if self.latest_edge_image is None:
            return

        binary_img = self.latest_edge_image
        height, width = binary_img.shape
        ploty = np.linspace(0, height - 1, height)

        # 이미지 컬러 변환 (시각화용)
        color_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

        # 히스토그램 기반 윈도우 시작 위치 계산
        histogram = np.sum(binary_img[int(height / 2):, :], axis=0)
        midpoint = width // 2
        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint

        # 슬라이딩 윈도우 파라미터
        nwindows = 10
        window_height = height // nwindows
        # 320x240에 맞게 조정
        margin = 30 
        minpix = 30

        nonzero = binary_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = left_base
        rightx_current = right_base

        left_lane_inds = []
        right_lane_inds = []

        # 슬라이딩 윈도우
        for window in range(nwindows):
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
            ).nonzero()[0]

            good_right_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
            ).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # 인덱스 결합
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # 신뢰도 계산 및 퍼블리시
        yellow_reliability = self.compute_reliability(len(left_lane_inds))
        white_reliability = self.compute_reliability(len(right_lane_inds))

        msg_yellow = UInt8()
        msg_yellow.data = yellow_reliability
        self.pub_yellow_line_reliability.publish(msg_yellow)

        msg_white = UInt8()
        msg_white.data = white_reliability
        self.pub_white_line_reliability.publish(msg_white)

        # 시각화 텍스트 추가
        cv2.putText(color_img, f"Left Line Reliability: {yellow_reliability}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(color_img, f"Right Line Reliability: {white_reliability}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 차선 피팅 및 중심선 계산
        left_fitx = right_fitx = None
        lane_state = UInt8()
        lane_state.data = 0

        if len(left_lane_inds) > 3000:
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            left_fit = np.polyfit(lefty, leftx, 2)
            left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            lane_state.data = 1
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
            cv2.polylines(color_img, pts_left, False, (0, 255, 255), 5)

        if len(right_lane_inds) > 3000:
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            right_fit = np.polyfit(righty, rightx, 2)
            right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
            lane_state.data = 3 if lane_state.data == 0 else 2
            pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))], dtype=np.int32)
            cv2.polylines(color_img, pts_right, False, (0, 255, 0), 5)

        # 중심선 계산
        centerx = None
        if lane_state.data == 2:  # 양쪽 차선 모두 검출
            centerx = (left_fitx + right_fitx) / 2
        elif lane_state.data == 1:  # 왼쪽 차선만 검출
            left_fit = np.polyfit(lefty, leftx, 2)
            left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            # 곡률 계산 (간단히 상수로 가정 대신 동적 계산 필요)
            curvature = abs(left_fit[0])  # 2차 계수로 곡률 근사
            dynamic_lane_width = self.lane_width_pixels * (1 + curvature * 100)  # 곡률에 따라 조정
            right_fitx = left_fitx + dynamic_lane_width
            centerx = (left_fitx + right_fitx) / 2
        elif lane_state.data == 3:  # 오른쪽Ok 차선만 검출
            right_fit = np.polyfit(righty, rightx, 2)
            right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
            curvature = abs(right_fit[0])
            dynamic_lane_width = self.lane_width_pixels * (1 + curvature * 100)
            left_fitx = right_fitx - dynamic_lane_width
            centerx = (left_fitx + right_fitx) / 2
        else:  # 양쪽 모두 미검출
            centerx = None  # 또는 이전 centerx 유지 가능

        if centerx is not None:
            center_pts = np.array([np.transpose(np.vstack([centerx, ploty]))], dtype=np.int32)
            cv2.polylines(color_img, center_pts, False, (255, 0, 0), 4)

            # 이미지 하단의 centerx를 발행 (예: y=350)
            msg_center = Float64()
            msg_center.data = float(centerx[350]) if centerx.shape[0] > 350 else float(centerx[-1])
            self.pub_lane.publish(msg_center)

        self.pub_lane_state.publish(lane_state)

        # 이미지 퍼블리시 및 시각화
        try:
            msg_img = self.bridge.cv2_to_compressed_imgmsg(color_img, "jpeg")
            self.pub_image_output.publish(msg_img)
        except Exception as e:
            self.get_logger().error(f"Failed to publish result image: {e}")

        try:
            cv2.imshow("Lane Detection on BEV Edge Image", color_img)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Failed to show image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SlicingWindowDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
