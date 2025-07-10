import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class HsvColorDetectorNode(Node):
    def __init__(self):
        super().__init__('hsv_color_detector_node')
        self.get_logger().info('HSV Color Detector Node has been started, now accepting camera feed for color calibration.')

        # CvBridge 초기화
        self.bridge = CvBridge()

        # /camera/image/compressed 토픽 구독
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed', # 터틀봇3 카메라 토픽
            self.image_callback,
            10
        )
        self.subscription # prevent unused variable warning

        # OpenCV 윈도우 생성 및 트랙바 설정
        cv2.namedWindow("Original Image", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Color Mask", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Result Image", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Trackbars")
        cv2.resizeWindow("Trackbars", 640, 240) # 윈도우 크기 조정

        # 트랙바 초기값 설정 (예시로 주황색 종이 범위로 시작)
        # H(Hue): 0-179, S(Saturation): 0-255, V(Value): 0-255
        cv2.createTrackbar("Lower H", "Trackbars", 10, 179, self.nothing)
        cv2.createTrackbar("Lower S", "Trackbars", 50, 255, self.nothing)
        cv2.createTrackbar("Lower V", "Trackbars", 80, 255, self.nothing)
        cv2.createTrackbar("Upper H", "Trackbars", 25, 179, self.nothing)
        cv2.createTrackbar("Upper S", "Trackbars", 255, 255, self.nothing)
        cv2.createTrackbar("Upper V", "Trackbars", 255, 255, self.nothing)

        # 마우스 콜백 설정 (선택 사항: 픽셀의 HSV 값 확인용)
        # cv2.setMouseCallback("Original Image", self.mouse_callback)
        # self.current_hsv_frame = None # 마우스 콜백에서 사용할 HSV 프레임 저장

    def nothing(self, x):
        # 트랙바가 변경될 때 호출되는 더미 함수
        pass

    # 선택 사항: 마우스 콜백 함수 (주석 처리되어 있습니다. 필요 시 활성화)
    # def mouse_callback(self, event, x, y, flags, param):
    #     if event == cv2.EVENT_LBUTTONDOWN and self.current_hsv_frame is not None:
    #         h, s, v = self.current_hsv_frame[y, x]
    #         self.get_logger().info(f"Clicked pixel HSV at ({x}, {y}): H={h}, S={s}, V={v}")

    def image_callback(self, msg):
        try:
            # ROS CompressedImage 메시지를 OpenCV 이미지로 변환
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            return

        # BGR 이미지를 HSV로 변환
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        # self.current_hsv_frame = hsv_image # 마우스 콜백에서 사용하기 위해 저장

        # 트랙바에서 현재 HSV 하한 및 상한 값 가져오기
        l_h = cv2.getTrackbarPos("Lower H", "Trackbars")
        l_s = cv2.getTrackbarPos("Lower S", "Trackbars")
        l_v = cv2.getTrackbarPos("Lower V", "Trackbars")
        u_h = cv2.getTrackbarPos("Upper H", "Trackbars")
        u_s = cv2.getTrackbarPos("Upper S", "Trackbars")
        u_v = cv2.getTrackbarPos("Upper V", "Trackbars")

        # 하한 및 상한 배열 생성
        lower_bound = np.array([l_h, l_s, l_v])
        upper_bound = np.array([u_h, u_s, u_v])

        # HSV 범위 내의 픽셀만 마스크로 추출
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # 마스크를 원본 이미지에 적용하여 해당 색상만 표시
        result_image = cv2.bitwise_and(cv_image, cv_image, mask=mask)

        # 결과 이미지 표시
        cv2.imshow("Original Image", cv_image)
        cv2.imshow("Color Mask", mask)
        cv2.imshow("Result Image", result_image)

        # OpenCV 윈도우 이벤트를 처리하고, 1ms 대기
        # 이 함수가 없으면 윈도우가 업데이트되지 않음
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    hsv_color_detector_node = HsvColorDetectorNode()
    try:
        rclpy.spin(hsv_color_detector_node)
    except KeyboardInterrupt:
        pass
    finally:
        # 노드 종료 시 자원 해제 및 모든 OpenCV 윈도우 닫기
        hsv_color_detector_node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()