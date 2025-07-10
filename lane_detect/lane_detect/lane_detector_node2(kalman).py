import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np

class LaneDetectorNode(Node):
    def __init__(self):
        super().__init__('lane_detector_node2')
        self.get_logger().info('Slicing Window Lane Detector Node has been started, now processing and visualizing lane lines using Sliding Window + Kalman Filter.')

        # 1. image_processor_node에서 발행하는 전처리된 BEV 엣지 이미지 구독
        self.subscription = self.create_subscription(
            CompressedImage,
            '/processed_image/edges', # BEV 엣지 이미지 토픽
            self.image_callback,
            10
        )
        self.subscription

        # 1. 원본 카메라 이미지 구독 (차선 검출 결과를 이 이미지 위에 그릴 것임)
        self.original_image_subscription = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed', # 원본 TurtleBot3 카메라 토픽
            self.original_image_callback,
            10
        )
        self.original_image_subscription
        

        # 최종 시각화된 차선 이미지를 발행합니다.
        self.publisher = self.create_publisher(CompressedImage, '/lane_detection/output_image', 10)

        # 파란색 주행선 계수를 발행합니다. (A, B, C)
        self.mid_lane_coeffs_publisher = self.create_publisher(Float32MultiArray, '/lane_detection/center_line_polynomial', 10)

        self.bridge = CvBridge()

        # 최신 원본 이미지를 저장할 변수
        self.latest_original_image = None

        # 차선 검출 결과를 표시할 OpenCV 창 생성
        cv2.namedWindow('Lane Detection Output', cv2.WINDOW_AUTOSIZE)

        # --------------------------------------------------
        # 슬라이딩 윈도우 파라미터 초기화
        # --------------------------------------------------
        self.num_windows = 8    # 슬라이딩 윈도우 개수
        self.margin = 25       # 윈도우 폭의 절반 (차선 픽셀을 검색할 마진)
        self.minpix = 50        # 윈도우가 다음 윈도우의 중앙을 재설정하기 위해 필요한 최소 픽셀 수

        # --------------------------------------------------
        # Kalman Filter 설정
        # 상태 벡터 (x): [A, B, C] (ax^2 + bx + c 형태의 2차 다항식 계수)
        # 2차 다항식: x = Ay^2 + By + C (여기서 y는 이미지의 세로축, x는 가로축)
        # --------------------------------------------------
        
        # left_lane Kalman Filter
        self.kf_left = cv2.KalmanFilter(3, 3) # 3개의 상태 (A, B, C), 3개의 측정 (A, B, C)
        self.kf_left.transitionMatrix = np.array([[1, 0, 0],
                                                  [0, 1, 0],
                                                  [0, 0, 1]], np.float32) # A, B, C 계수가 시간에 따라 유지된다고 가정
        self.kf_left.measurementMatrix = np.array([[1, 0, 0],
                                                   [0, 1, 0],
                                                   [0, 0, 1]], np.float32) # A, B, C 계수를 직접 측정
        self.kf_left.processNoiseCov = np.array([[1e-4, 0, 0],
                                                  [0, 1e-4, 0],
                                                  [0, 0, 1e-4]], np.float32) # 시스템 모델 노이즈 (계수가 얼마나 변동하는지)
        self.kf_left.measurementNoiseCov = np.array([[1e-2, 0, 0],
                                                      [0, 1e-2, 0],
                                                      [0, 0, 1e-2]], np.float32) # 측정 노이즈 (계수 계산의 불확실성)
        self.kf_left.errorCovPost = np.array([[1, 0, 0],
                                                [0, 1, 0],
                                                [0, 0, 1]], np.float32) * 1 # 초기 오차 공분산
        self.kf_left.statePost = np.array([[0.], [0.], [0.]], np.float32) # 초기 상태 (A, B, C)

        # right_lane Kalman Filter
        self.kf_right = cv2.KalmanFilter(3, 3) # 3개의 상태 (A, B, C), 3개의 측정 (A, B, C)
        self.kf_right.transitionMatrix = np.array([[1, 0, 0],
                                                   [0, 1, 0],
                                                   [0, 0, 1]], np.float32)
        self.kf_right.measurementMatrix = np.array([[1, 0, 0],
                                                    [0, 1, 0],
                                                    [0, 0, 1]], np.float32)
        self.kf_right.processNoiseCov = np.array([[1e-4, 0, 0],
                                                   [0, 1e-4, 0],
                                                   [0, 0, 1e-4]], np.float32)
        self.kf_right.measurementNoiseCov = np.array([[1e-2, 0, 0],
                                                       [0, 1e-2, 0],
                                                       [0, 0, 1e-2]], np.float32)
        self.kf_right.errorCovPost = np.array([[1, 0, 0],
                                                 [0, 1, 0],
                                                 [0, 0, 1]], np.float32) * 1
        self.kf_right.statePost = np.array([[0.], [0.], [0.]], np.float32)


        # --------------------------------------------------
        # 차선 유효성 검증을 위한 임계값
        # --------------------------------------------------
        self.curvature_threshold_A = 0.0005 # A 계수 (곡률) 변화량 임계값
        self.offset_threshold_C = 100        # C 계수 (차선 위치 오프셋) 변화량 임계값

        # 이전 프레임의 필터링된 차선 계수 저장
        self.prev_left_coeffs_filtered = None
        self.prev_right_coeffs_filtered = None
        

    def original_image_callback(self, msg):
        """
        원본 카메라 이미지를 구독하여 저장합니다.
        """
        try:
            self.latest_original_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error in original_image_callback: {e}")


    def image_callback(self, msg):
        """
        image_processor_node에서 전처리된 BEV 엣지 이미지를 구독하여,
        슬라이딩 윈도우를 적용하고, 2차 다항식을 피팅한 후, 칼만 필터로 스무딩하여
        최종 차선 및 주행선을 원본 이미지 위에 시각화하고 발행합니다.
        """

        try:
            # 1. image_processor_node에서 전처리된 BEV 엣지 이미지를 받아옵니다.
            binary_warped = self.bridge.compressed_imgmsg_to_cv2(msg, "mono8") # /processed_image/edges는 단일 채널
            height, width = binary_warped.shape[:2]
            self.window_height = np.int32(height // self.num_windows)

            # 원본 이미지 복사 (차선 검출 결과를 이 위에 그릴 것임)
            output_image = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR)
            # output_image = np.dstack((binary_warped, binary_warped, binary_warped)) * 255 # 디버깅용: 이진화된 이미지에 그리기

            # 히스토그램을 계산하여 차선 베이스 픽셀 찾기
            histogram = np.sum(binary_warped[height // 2:, :], axis=0)
            midpoint = np.int32(width // 2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            current_leftx = leftx_base
            current_rightx = rightx_base

            left_lane_inds = []
            right_lane_inds = []

            # 윈도우 중심 좌표를 저장할 리스트 (polyfit용)
            left_center_y = []
            left_center_x = []
            right_center_y = []
            right_center_x = []


            # 슬라이딩 윈도우 시작
            for window in range(self.num_windows):
                win_y_low = height - (window + 1) * self.window_height
                win_y_high = height - window * self.window_height

                win_x_left_low = current_leftx - self.margin
                win_x_left_high = current_leftx + self.margin
                win_x_right_low = current_rightx - self.margin
                win_x_right_high = current_rightx + self.margin

                # 윈도우 그리기 (초록색)
                cv2.rectangle(output_image, (win_x_left_low, win_y_low), (win_x_left_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(output_image, (win_x_right_low, win_y_low), (win_x_right_high, win_y_high), (0, 255, 0), 2)

                # 윈도우 중심 좌표 계산 및 초록색 점 그리기
                center_x_left = (win_x_left_low + win_x_left_high) // 2
                center_y_left = (win_y_low + win_y_high) // 2
                cv2.circle(output_image, (center_x_left, center_y_left), 1, (0, 255, 0), -1) 

                center_x_right = (win_x_right_low + win_x_right_high) // 2
                center_y_right = (win_y_low + win_y_high) // 2
                cv2.circle(output_image, (center_x_right, center_y_right), 1, (0, 255, 0), -1) 

                # 현재 윈도우 내에 있는 non-zero 픽셀 식별
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                (nonzerox >= win_x_left_low) & (nonzerox < win_x_left_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                (nonzerox >= win_x_right_low) & (nonzerox < win_x_right_high)).nonzero()[0]

                # 차선 픽셀 인덱스 추가
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)

                # 만약 윈도우 내에 충분한 픽셀이 있다면, 다음 윈도우의 중앙을 재설정하고 중심 좌표 저장
                if len(good_left_inds) > self.minpix:
                    current_leftx = np.int32(np.mean(nonzerox[good_left_inds]))
                    left_center_x.append(current_leftx)
                    left_center_y.append(np.int32(np.mean(nonzeroy[good_left_inds])))
                else: # 픽셀이 충분치 않으면 윈도우 중심 그대로 사용 (Kalman filter가 보정해 줄 것)
                    left_center_x.append(center_x_left)
                    left_center_y.append(center_y_left)

                if len(good_right_inds) > self.minpix:
                    current_rightx = np.int32(np.mean(nonzerox[good_right_inds]))
                    right_center_x.append(current_rightx)
                    right_center_y.append(np.int32(np.mean(nonzeroy[good_right_inds])))
                else: # 픽셀이 충분치 않으면 윈도우 중심 그대로 사용
                    right_center_x.append(center_x_right)
                    right_center_y.append(center_y_right)
            
            # 모든 윈도우에서 찾은 픽셀 인덱스들을 연결 (차선 피팅을 위한 준비) - 사용하지 않더라도 기존 코드 유지
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            ploty = np.linspace(0, height - 1, height)

            left_fit = None
            right_fit = None
            
            left_line_valid = False
            right_line_valid = False

            # 좌측 차선 피팅 및 칼만 필터 적용
            if len(left_center_x) > 2:
                left_fit_current = np.polyfit(left_center_y, left_center_x, 2)
                
                # Kalman filter 예측
                self.kf_left.predict()
                
                # Kalman filter 업데이트
                measured_state_left = np.array([[left_fit_current[0]],
                                                [left_fit_current[1]],
                                                [left_fit_current[2]]], np.float32)
                self.kf_left.correct(measured_state_left)
                left_coeffs_filtered = self.kf_left.statePost.flatten()

                # 유효성 검사 (첫 프레임이 아닐 때만)
                if self.prev_left_coeffs_filtered is not None:
                    diff_A = abs(left_coeffs_filtered[0] - self.prev_left_coeffs_filtered[0])
                    diff_C = abs(left_coeffs_filtered[2] - self.prev_left_coeffs_filtered[2])

                    if diff_A < self.curvature_threshold_A and diff_C < self.offset_threshold_C:
                        left_line_valid = True
                    else:
                        self.get_logger().warn(f"Left lane invalid: A_diff={diff_A:.6f}, C_diff={diff_C:.2f}")
                        # 유효하지 않으면 이전 필터링된 값을 사용하거나, 예측값 사용 (현재는 이전값 유지)
                        left_coeffs_filtered = self.prev_left_coeffs_filtered
                else:
                    left_line_valid = True # 첫 프레임은 항상 유효하다고 가정

                left_fitx_filtered = left_coeffs_filtered[0]*ploty**2 + left_coeffs_filtered[1]*ploty + left_coeffs_filtered[2]
                pts_left = np.array([np.transpose(np.vstack([left_fitx_filtered, ploty]))])
                cv2.polylines(output_image, np.int32([pts_left]), False, (0, 255, 255), 5) # 노란색으로 차선 그리기
                self.prev_left_coeffs_filtered = left_coeffs_filtered # 현재 필터링된 값을 이전 값으로 저장
            else:
                self.get_logger().warn("Not enough points to fit left lane polynomial.")
                # 유효한 차선이 없으면 칼만 필터 예측만 사용하고, 이전 유효값 유지 시도
                self.kf_left.predict()
                if self.prev_left_coeffs_filtered is not None:
                    left_coeffs_filtered = self.prev_left_coeffs_filtered
                    left_fitx_filtered = left_coeffs_filtered[0]*ploty**2 + left_coeffs_filtered[1]*ploty + left_coeffs_filtered[2]
                    pts_left = np.array([np.transpose(np.vstack([left_fitx_filtered, ploty]))])
                    cv2.polylines(output_image, np.int32([pts_left]), False, (0, 255, 255), 5) # 노란색으로 차선 그리기
                left_line_valid = False

            # 우측 차선 피팅 및 칼만 필터 적용
            if len(right_center_x) > 2:
                right_fit_current = np.polyfit(right_center_y, right_center_x, 2)
                
                # Kalman filter 예측
                self.kf_right.predict()
                
                # Kalman filter 업데이트
                measured_state_right = np.array([[right_fit_current[0]],
                                                 [right_fit_current[1]],
                                                 [right_fit_current[2]]], np.float32)
                self.kf_right.correct(measured_state_right)
                right_coeffs_filtered = self.kf_right.statePost.flatten()

                # 유효성 검사 (첫 프레임이 아닐 때만)
                if self.prev_right_coeffs_filtered is not None:
                    diff_A = abs(right_coeffs_filtered[0] - self.prev_right_coeffs_filtered[0])
                    diff_C = abs(right_coeffs_filtered[2] - self.prev_right_coeffs_filtered[2])
                    
                    if diff_A < self.curvature_threshold_A and diff_C < self.offset_threshold_C:
                        right_line_valid = True
                    else:
                        self.get_logger().warn(f"Right lane invalid: A_diff={diff_A:.6f}, C_diff={diff_C:.2f}")
                        right_coeffs_filtered = self.prev_right_coeffs_filtered
                else:
                    right_line_valid = True # 첫 프레임은 항상 유효하다고 가정

                right_fitx_filtered = right_coeffs_filtered[0]*ploty**2 + right_coeffs_filtered[1]*ploty + right_coeffs_filtered[2]
                pts_right = np.array([np.transpose(np.vstack([right_fitx_filtered, ploty]))])
                cv2.polylines(output_image, np.int32([pts_right]), False, (0, 255, 0), 5) # 초록색으로 차선 그리기
                self.prev_right_coeffs_filtered = right_coeffs_filtered # 현재 필터링된 값을 이전 값으로 저장
            else:
                self.get_logger().warn("Not enough points to fit right lane polynomial.")
                self.kf_right.predict()
                if self.prev_right_coeffs_filtered is not None:
                    right_coeffs_filtered = self.prev_right_coeffs_filtered
                    right_fitx_filtered = right_coeffs_filtered[0]*ploty**2 + right_coeffs_filtered[1]*ploty + right_coeffs_filtered[2]
                    pts_right = np.array([np.transpose(np.vstack([right_fitx_filtered, ploty]))])
                    cv2.polylines(output_image, np.int32([pts_right]), False, (0, 255, 0), 5) # 초록색으로 차선 그리기
                right_line_valid = False


            # --------------------------------------------------
            # 가상의 주행선 그리기 
            # --------------------------------------------------
            if left_line_valid and right_line_valid:
                mid_line_x = (left_fitx_filtered + right_fitx_filtered) / 2
                mid_line_points = np.int32(np.array(list(zip(mid_line_x, ploty))))
                mid_line_points_reshaped = mid_line_points.reshape((-1, 1, 2))
                cv2.polylines(output_image, [mid_line_points_reshaped], False, (255, 0, 0), 5) # 파란색 (BGR) 곡선
                
                # 주행선 계수 발행
                mid_coeffs = np.polyfit(np.array(mid_line_points[:,1]), np.array(mid_line_points[:,0]), 2)
                mid_coeffs_msg = Float32MultiArray()
                mid_coeffs_msg.data = mid_coeffs.tolist()
                self.mid_lane_coeffs_publisher.publish(mid_coeffs_msg)
            else:
                self.get_logger().warn("Cannot calculate mid lane as one or both lanes are invalid.")


            # 최종 출력 이미지 표시
            output_image_resized = cv2.resize(output_image, (width * 2, height * 2)) # 시각화 크기 조절
            cv2.imshow('Lane Detection Output', output_image_resized)
            cv2.waitKey(1)

            # 최종 시각화된 이미지 ROS 토픽 발행
            try:
                processed_msg = self.bridge.cv2_to_compressed_imgmsg(output_image, "jpeg")
                self.publisher.publish(processed_msg)
            except Exception as e:
                self.get_logger().error(f"Error publishing output image: {e}")

        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    lane_detector_node = LaneDetectorNode()
    try:
        rclpy.spin(lane_detector_node)
    except KeyboardInterrupt:
        pass
    finally:
        lane_detector_node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()