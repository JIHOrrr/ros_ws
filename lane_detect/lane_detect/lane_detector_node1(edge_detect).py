import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

# lane_detector_node에서 발행하는 파란색 주행선 계수를 구독하는 부분은
# 이 파일에 직접적으로 영향을 주지 않지만, 전체 시스템의 일부이므로 유지
from std_msgs.msg import Float32MultiArray # track_moving_node로 계수를 발행하기 위함

class LaneDetectorNode(Node):
    def __init__(self):
        super().__init__('lane_detector_node')
        self.get_logger().info('Lane Detector Node has been started, now processing and visualizing lane lines using Hough + Linear Regression + Kalman Filter.')

        # image_processor_node에서 발행하는 엣지 이미지 토픽을 구독합니다.
        self.subscription = self.create_subscription(
            CompressedImage,
            '/processed_image/edges',
            self.image_callback,
            10
        )
        self.subscription

        # 원본 카메라 이미지를 구독합니다. 차선 검출 결과를 이 이미지 위에 그릴 것입니다.
        self.original_image_subscription = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed', # 원본 TurtleBot3 camera topic
            self.original_image_callback,
            10
        )
        self.original_image_subscription

        # 최종 시각화된 차선 이미지를 발행합니다. (새로운 퍼블리셔 추가)
        self.publisher = self.create_publisher(CompressedImage, '/lane_detection/output_image', 10)
        
        # ⚡️ 파란색 주행선 계수를 track_moving_node로 발행하기 위한 퍼블리셔 추가
        self.mid_lane_coeffs_publisher = self.create_publisher(Float32MultiArray, '/lane_detection/mid_lane_coeffs', 10)


        self.bridge = CvBridge()

        # 최신 원본 이미지를 저장할 변수
        self.latest_original_image = None

        # 차선 검출 결과를 표시할 OpenCV 창 생성
        cv2.namedWindow('Lane Detection Output', cv2.WINDOW_AUTOSIZE)

        # --------------------------------------------------
        # 칼만 필터 초기화 (상태 벡터를 3차원으로 변경: [A, B, C] for x = Ay^2 + By + C)
        # --------------------------------------------------
        # 좌측 차선 칼만 필터
        self.kf_left = cv2.KalmanFilter(3, 3) # 3차원 상태, 3차원 측정
        self.kf_left.transitionMatrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
        self.kf_left.measurementMatrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
        self.kf_left.processNoiseCov = np.array([[1e-6, 0, 0], [0, 1e-6, 0], [0, 0, 1e-6]], np.float32) # 프로세스 노이즈
        self.kf_left.measurementNoiseCov = np.array([[1e-1, 0, 0], [0, 1e-1, 0], [0, 0, 1e-1]], np.float32) # 측정 노이즈
        self.kf_left.errorCovPost = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32) * 100 # 초기 오차 공분산
        self.kf_left.statePost = np.array([[0.], [0.], [0.]], np.float32) # 초기 상태 (A, B, C)

        # 우측 차선 칼만 필터
        self.kf_right = cv2.KalmanFilter(3, 3) # 3차원 상태, 3차원 측정
        self.kf_right.transitionMatrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
        self.kf_right.measurementMatrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
        self.kf_right.processNoiseCov = np.array([[1e-6, 0, 0], [0, 1e-6, 0], [0, 0, 1e-6]], np.float32)
        self.kf_right.measurementNoiseCov = np.array([[1e-1, 0, 0], [0, 1e-1, 0], [0, 0, 1e-1]], np.float32)
        self.kf_right.errorCovPost = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32) * 100
        self.kf_right.statePost = np.array([[0.], [0.], [0.]], np.float32)

        # ⚡️ 차선 감지를 위한 최소 포인트 임계값 설정
        self.min_lane_points_threshold = 20 # np.polyfit을 위한 최소한의 (x,y) 쌍 개수

    def original_image_callback(self, msg):
        """
        원본 카메라 이미지를 수신하여 저장하는 콜백 함수.
        """
        try:
            self.latest_original_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting original image: {e}")

    def image_callback(self, msg):
        """
        /processed_image/edges 토픽에서 엣지 이미지를 수신하고 차선 검출 및 시각화를 수행합니다.
        """
        if self.latest_original_image is None:
            self.get_logger().warn("Waiting for original image to be received before processing edges...")
            return

        try:
            # 엣지 이미지를 OpenCV 이미지로 변환
            temp_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            
            if len(temp_image.shape) == 3: # 3채널인 경우 (BGR)
                edges = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
            elif len(temp_image.shape) == 2: # 1채널인 경우 (이미 그레이스케일)
                edges = temp_image
            else:
                self.get_logger().error(f"Unexpected number of channels ({len(temp_image.shape)}) in image received from CvBridge for edges.")
                return

        except Exception as e:
            self.get_logger().error(f"Error converting edge image: {e}")
            return

        # 원본 이미지 복사본에 차선 그리기
        output_image = np.copy(self.latest_original_image)
        
        height, width = edges.shape

        # 허프 변환을 통한 직선 성분 추출
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=40,    # 최소 투표 수
            minLineLength=40, # 최소 길이
            maxLineGap=15    # 최대 간격
        )

        # 선형 회귀를 위한 포인트 데이터 수집
        left_lane_points_x = []
        left_lane_points_y = []
        right_lane_points_x = []
        right_lane_points_y = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                
                # 기울기 계산 (수직선 방지)
                if (x2 - x1) == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                
                # 기울기 임계값
                slope_threshold = 0.3 
                # BEV 이미지에서는 차선이 거의 수직이므로, 너무 수평인 선만 제외
                if abs(slope) < slope_threshold: 
                    continue
                
                # 기울기와 위치를 기반으로 좌우 차선 분류
                if slope < 0: # 음수 기울기 (좌측 차선)
                    if x1 < width / 2 and x2 < width / 2: # 이미지 좌반부 확인
                        left_lane_points_x.extend([x1, x2])
                        left_lane_points_y.extend([y1, y2])
                elif slope > 0: # 양수 기울기 (우측 차선)
                    if x1 > width / 2 and x2 > width / 2: # 이미지 우반부 확인
                        right_lane_points_x.extend([x1, x2])
                        right_lane_points_y.extend([y1, y2])

        # --------------------------------------------------
        # 칼만 필터 예측 단계
        # --------------------------------------------------
        # 예측된 상태를 기본값으로 사용
        prediction_left = self.kf_left.predict()
        predicted_coeffs_left = prediction_left.flatten() # [A, B, C]
        
        prediction_right = self.kf_right.predict()
        predicted_coeffs_right = prediction_right.flatten() # [A, B, C]

        # --------------------------------------------------
        # 선형 회귀 (2차 다항식) 및 칼만 필터 업데이트 단계
        # ⚡️ 최소 포인트 임계값 검사 로직 추가
        # --------------------------------------------------
        filtered_coeffs_left = predicted_coeffs_left
        # 좌측 차선 포인트가 충분할 경우에만 측정값으로 보정
        if len(left_lane_points_x) >= self.min_lane_points_threshold:
            left_fit = np.polyfit(left_lane_points_y, left_lane_points_x, 2) 
            measurement_left = np.array([[left_fit[0]], [left_fit[1]], [left_fit[2]]], np.float32)
            self.kf_left.correct(measurement_left)
            filtered_coeffs_left = self.kf_left.statePost.flatten()
        else:
            # 포인트가 부족하면 예측값만 사용하고 경고 로그
            self.get_logger().warn("Not enough points for left lane detection. Relying on Kalman Filter prediction.")

        filtered_coeffs_right = predicted_coeffs_right
        # 우측 차선 포인트가 충분할 경우에만 측정값으로 보정
        if len(right_lane_points_x) >= self.min_lane_points_threshold:
            right_fit = np.polyfit(right_lane_points_y, right_lane_points_x, 2)
            measurement_right = np.array([[right_fit[0]], [right_fit[1]], [right_fit[2]]], np.float32)
            self.kf_right.correct(measurement_right)
            filtered_coeffs_right = self.kf_right.statePost.flatten()
        else:
            # 포인트가 부족하면 예측값만 사용하고 경고 로그
            self.get_logger().warn("Not enough points for right lane detection. Relying on Kalman Filter prediction.")


        # --------------------------------------------------
        # 곡선 좌표 생성 헬퍼 함수
        # --------------------------------------------------
        def generate_curved_line_points(image_height, coeffs, image_width):
            if coeffs is None or len(coeffs) != 3:
                return None
            
            # 이미지 하단부터 상단까지 50개 포인트 생성 (y 값)
            plot_y = np.linspace(image_height, 0, num=50).astype(int) 
            
            # x = Ay^2 + By + C 계산
            plot_x = coeffs[0] * plot_y**2 + coeffs[1] * plot_y + coeffs[2]
            
            # x 좌표가 이미지 범위를 벗어나지 않도록 클리핑 (넓은 범위 클리핑)
            plot_x = np.clip(plot_x, -image_width * 2, image_width * 2).astype(int) 

            # (x, y) 쌍으로 반환하고 polylines를 위한 형태로 reshape
            points = np.column_stack((plot_x, plot_y))
            return points.reshape((-1, 1, 2)) 

        left_line_points = generate_curved_line_points(height, filtered_coeffs_left, width)
        right_line_points = generate_curved_line_points(height, filtered_coeffs_right, width)


        # --------------------------------------------------
        # 차선 내부를 초록색으로 채우기
        # --------------------------------------------------
        # ⚡️ 두 차선 포인트가 모두 유효할 때만 채우기 및 파란선 그리기
        mid_lane_coeffs = None
        if left_line_points is not None and right_line_points is not None:
            # 다각형의 꼭짓점 정의: 왼쪽 차선 포인트 (아래에서 위로) + 오른쪽 차선 포인트 (위에서 아래로)
            all_points = np.vstack((left_line_points, np.flipud(right_line_points)))
            
            # 투명한 오버레이 생성
            overlay = np.zeros_like(output_image, dtype=np.uint8)
            cv2.fillPoly(overlay, [all_points], (0, 255, 0)) # 초록색 (BGR)으로 채우기
            
            # 원본 이미지와 오버레이 블렌딩 (투명도 40%)
            alpha = 0.4 
            output_image = cv2.addWeighted(output_image, 1, overlay, alpha, 0)

            # --------------------------------------------------
            # 가상 주행선 (파란색) 표시 및 계수 발행
            # --------------------------------------------------
            mid_line_points = []
            # 파란선 계수 (단순 평균)
            mid_A = (filtered_coeffs_left[0] + filtered_coeffs_right[0]) / 2.0
            mid_B = (filtered_coeffs_left[1] + filtered_coeffs_right[1]) / 2.0
            mid_C = (filtered_coeffs_left[2] + filtered_coeffs_right[2]) / 2.0
            mid_lane_coeffs = [mid_A, mid_B, mid_C]

            for i in range(left_line_points.shape[0]):
                lx, ly = left_line_points[i][0]
                rx, ry = right_line_points[i][0]
                mid_x = (lx + rx) // 2
                mid_y = (ly + ry) // 2
                mid_line_points.append([mid_x, mid_y])
            
            mid_line_points = np.array(mid_line_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(output_image, [mid_line_points], False, (255, 0, 0), 5) # 파란색 (BGR) 곡선
        else:
            self.get_logger().warn("One or both lanes not reliably detected. Skipping fill and mid-lane drawing.")

        # --------------------------------------------------
        # 검출된 차선 그리기 (빨간색) - 채우기 후에 그리면 선이 더 잘 보임
        # --------------------------------------------------
        if left_line_points is not None:
            cv2.polylines(output_image, [left_line_points], False, (0, 0, 255), 10) # 빨간색 (BGR) 곡선
        
        if right_line_points is not None:
            cv2.polylines(output_image, [right_line_points], False, (0, 0, 255), 10) # 빨간색 (BGR) 곡선

        # 최종 출력 이미지 표시
        cv2.imshow('Lane Detection Output', output_image)
        cv2.waitKey(1)

        # ⚡️ 파란색 주행선 계수 발행 (mid_lane_coeffs가 유효할 때만)
        if mid_lane_coeffs is not None:
            coeffs_msg = Float32MultiArray()
            coeffs_msg.data = mid_lane_coeffs
            self.mid_lane_coeffs_publisher.publish(coeffs_msg)
        else:
            # 유효한 파란선 계수가 없을 경우, 0으로 채워진 메시지를 발행하거나 발행하지 않을 수 있음
            # track_moving_node의 "No lane coefficients received" 경고를 활용하기 위해 메시지 발행 안 함
            pass


        # 최종 시각화된 이미지 ROS 토픽 발행
        try:
            processed_msg = self.bridge.cv2_to_compressed_imgmsg(output_image, "jpeg")
            self.publisher.publish(processed_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing output image: {e}")


def main(args=None):
    rclpy.init(args=args)
    lane_detector_node = LaneDetectorNode()
    try:
        rclpy.spin(lane_detector_node)
    except KeyboardInterrupt:
        pass
    finally:
        lane_detector_node.destroy_node()
        cv2.destroyAllWindows() # OpenCV 창 닫기
        rclpy.shutdown()

if __name__ == '__main__':
    main()