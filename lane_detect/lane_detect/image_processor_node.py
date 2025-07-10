import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageProcessorNode(Node):
    def __init__(self):
        super().__init__('image_processor_node')
        self.get_logger().info('Image Processor Node has been started, performing perspective transformation based on user-defined points.')

        # Subscribing to the compressed image topic from TurtleBot3 camera
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed', # TurtleBot3 camera topic
            self.image_callback,
            10
        )
        self.subscription  # prevent unused variable warning

        # Publishing the processed (edge detected) image
        self.publisher = self.create_publisher(CompressedImage, '/processed_image/edges', 10)

        
        # Initialize CvBridge
        self.bridge = CvBridge()

        # Create OpenCV windows for visualization
        cv2.namedWindow('Original Image', cv2.WINDOW_AUTOSIZE) # Original image added
        cv2.namedWindow('Bird Eye View', cv2.WINDOW_AUTOSIZE) # BEV image added

        # --------------------------------------------------
        # 카메라 캘리브레이션 파라미터 로드
        # (camera_calibration.py에서 저장한 파일 경로를 정확히 지정하세요)
        # --------------------------------------------------
        try:
            # camera_calibration_params.npz 파일이 image_processor_node.py와 같은 디렉토리에 있다면 이렇게
            self.calibration_file_path = '/home/jiho/robot_ws/src/lane_detect/lane_detect/camera_calibration_params.npz'
            
            calibration_data = np.load(self.calibration_file_path)
            self.mtx = calibration_data['mtx']
            self.dist = calibration_data['dist']
            self.get_logger().info('Camera calibration parameters loaded successfully.')
        except FileNotFoundError:
            self.get_logger().error(f'"{self.calibration_file_path}" not found! Please run camera_calibration.py first or check the path.')
            self.mtx = None
            self.dist = None
        except Exception as e:
            self.get_logger().error(f'Error loading calibration parameters: {e}')
            self.mtx = None
            self.dist = None

        # --------------------------------------------------
        # 원본 이미지에서 정의된 소스 점 (사용자가 제공한 좌표)
        # 이 점들은 카메라 뷰에서 도로 차선이 보이는 경계에 해당하며,
        # 이제 ROI를 정의하는 데도 활용됩니다.
        # --------------------------------------------------
        self.src_points = np.float32([
            [0, 175],    # bottom_left
            [320, 175],  # bottom_right
            [270, 40],   # top_right
            [40, 0]      # top_left
        ])

        # --------------------------------------------------
        # 변환될 BEV 이미지에서 4개의 목표 점 (직사각형 형태)
        # --------------------------------------------------
        self.dst_points = np.float32([
            [40, 239],   # bottom-left (src_points의 top_left x좌표, 이미지 최하단 y좌표)
            [270, 239],  # bottom-right (src_points의 top_right x좌표, 이미지 최하단 y좌표)
            [270, 0],    # top-right (src_points의 top_right x좌표, 이미지 최상단 y좌표)
            [40, 0]      # top-left (src_points의 top_left x좌표, 이미지 최상단 y좌표)
        ])

        # 원근 변환 행렬은 image_callback 내에서 (왜곡 보정된 이미지에 대해) 다시 계산될 수 있습니다.
        # 초기화 시에는 일단 src_points와 dst_points를 기반으로 생성합니다.
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)


    def image_callback(self, msg):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            return

        height, width = cv_image.shape[:2]
        cv2.imshow('Original Image', cv_image)
        
        image_to_process = cv_image

        # --------------------------------------------------
        # 0. 이미지 왜곡 보정 (Undistortion)
        # --------------------------------------------------
        mask = np.zeros_like(image_to_process)
        roi_points = np.array([
        [0, 0],        # top-left
        [316, 0],      # top-right
        [316, 238],    # bottom-right
        [0, 238]       # bottom-left
        ], dtype=np.int32)
        cv2.fillPoly(mask, [roi_points], (255, 255, 255))  # 흰색 채우기
        roi_image = cv2.bitwise_and(image_to_process, mask)
        cv2.imshow('ROI Applied', roi_image)

    # ✅ ROI 제거 후 전체 이미지 사용
        image_for_bev = image_to_process  # 전체 이미지 BEV에 사용

        # --------------------------------------------------
        # ⚡️ 변경된 부분: 1. 원근 변환 (Perspective Transform) 
        # --------------------------------------------------
        # M이 아직 계산되지 않은 경우에만 계산 (한 번만 계산)
        if self.M is None:
            # src_points와 dst_points를 사용하여 변환 행렬 M 계산
            self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
            # 역변환 행렬 Minv 계산
            self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)


        # BEV 변환 수행
        bird_eye_view = cv2.warpPerspective(image_for_bev, self.M, (width, height), flags=cv2.INTER_LINEAR)
        cv2.imshow('Bird Eye View', bird_eye_view)

        # 이후 모든 이미지 처리는 bird_eye_view 이미지에 대해 수행됩니다.
        # --------------------------------------------------
        # 2. 색상 필터링 (흰색 및 노란색) - BEV 이미지에 적용
        # --------------------------------------------------
        hsv = cv2.cvtColor(bird_eye_view, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 200])
        upper_white = np.array([255, 15, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        lower_yellow = np.array([10, 50, 80])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        combined_mask = cv2.bitwise_or(mask_white, mask_yellow)
        color_filtered_image = cv2.bitwise_and(bird_eye_view, bird_eye_view, mask=combined_mask)
        
        # --------------------------------------------------
        # 3. 그레이스케일 변환 - BEV 이미지 기반
        # --------------------------------------------------
        #gray = cv2.cvtColor(color_filtered_image, cv2.COLOR_BGR2GRAY)

        # --------------------------------------------------
        # 4. 가우시안 블러 (노이즈 제거) - BEV 이미지 기반
        # --------------------------------------------------
        blurred = cv2.GaussianBlur(color_filtered_image, (5, 5), 0)
        cv2.imshow('Output Image', blurred) # 최종 발행되는 이미지     
        cv2.waitKey(1)

        # --------------------------------------------------
        # Publish the processed image
        # --------------------------------------------------
        try:
            processed_msg = self.bridge.cv2_to_compressed_imgmsg(blurred, "jpeg")
            self.publisher.publish(processed_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    image_processor_node = ImageProcessorNode()
    try:
        rclpy.spin(image_processor_node)
    except KeyboardInterrupt:
        pass
    finally:
        image_processor_node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
