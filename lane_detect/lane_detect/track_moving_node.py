import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
import numpy as np

class TrackMovingNode(Node):
    def __init__(self):
        super().__init__('track_moving_node')
        self.get_logger().info('Track Moving Node has been started, now controlling TurtleBot3 based on advanced lane detection for cornering.')

        # ⚡️ /lane_detection/center_line_polynomial 토픽에서 파란색 주행선 계수를 구독합니다.
        # 기존 코드에서 mid_lane_coeffs를 구독했지만, 더 일반적인 center_line_polynomial로 변경합니다.
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/lane_detection/center_line_polynomial', # ⚡️ 토픽 이름 변경
            self.lane_coeffs_callback,
            10
        )
        self.subscription

        # TurtleBot3의 움직임을 제어하는 Twist 메시지를 발행합니다.
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # 제어 파라미터 초기화
        # PID (Proportional-Integral-Derivative) 제어 게인
        # ⚡️ 게인 값은 실제 환경에서 튜닝이 필요합니다.
        self.Kp_lateral = 0.005  # 횡방향 오차에 대한 비례 게인
        self.Kd_lateral = 0.01   # 횡방향 오차 변화율에 대한 미분 게인 (오버슈트 방지)
        # Ki_lateral는 현재 사용하지 않음 (필요 시 추가 가능)

        # ⚡️ 방향 오차에 대한 비례 게인 (기울기 기반)
        self.Kp_heading = 0.5

        # ⚡️ 곡률(A 계수)에 대한 비례 게인 (코너 주행 보조)
        self.Kp_curvature = 2000.0 # A 값은 매우 작으므로 큰 값을 곱하여 영향력을 높임 (튜닝 필요)

        self.previous_lateral_error = 0.0
        self.integral_lateral_error = 0.0 # 현재 사용 안 함 (Ki=0)

        # 이미지 중심 설정 (예: 640x480 이미지의 너비 중앙)
        self.image_width = 640
        self.image_height = 480
        self.center_x = self.image_width / 2

        # 기본 전진 속도
        self.base_linear_speed = 0.05 # m/s

        # --------------------------------------------------
        # ⚡️ 전방 주시 거리 설정 (y_lookahead_point와 y_point_for_lateral_error 조정)
        # 이미지 높이의 50% 지점을 횡방향 오차 계산 기준으로 사용하여 전방 주시 효과를 높입니다.
        # 이 값을 줄이면(예: 0.3) 더 먼 곳을 보고 제어하게 됩니다.
        self.lookahead_y_ratio = 0.5 # ⚡️ 이미지 높이의 50% 지점을 전방 주시점으로 설정 (튜닝 가능)
        self.y_lookahead_point = int(self.image_height * self.lookahead_y_ratio)

        # --------------------------------------------------
        # 파란선 이상 감지 및 정지 임계값 (기존 유지)
        self.curvature_threshold_A = 0.002
        self.slope_threshold_B = 0.8
        # --------------------------------------------------

        # 제어 주기 타이머 (예: 20ms마다 제어 명령 발행)
        self.timer = self.create_timer(0.02, self.publish_twist_command) # 50 Hz

        self.current_coeffs = None # 최신 파란색 주행선 계수 저장
        self.last_command_time = self.get_clock().now() # 마지막 제어 명령 시간


    def lane_coeffs_callback(self, msg):
        """
        /lane_detection/center_line_polynomial 토픽으로부터 파란색 주행선 계수를 수신합니다.
        """
        # msg.data는 [A, B, C] 형태의 리스트입니다. (x = Ay^2 + By + C)
        self.current_coeffs = np.array(msg.data)
        self.last_command_time = self.get_clock().now()


    def publish_twist_command(self):
        """
        주기적으로 Twist 메시지를 계산하고 발행합니다.
        """
        twist_msg = Twist()
        twist_msg.linear.x = self.base_linear_speed # 기본 전진 속도 유지
        twist_msg.angular.z = 0.0 # 기본 각속도

        # 계수가 없으면 (차선 미검출 또는 초기 상태) 차량 정지
        if self.current_coeffs is None:
            self.get_logger().warn("No lane coefficients received. Stopping the robot.")
            self.publisher.publish(twist_msg) # linear.x=0, angular.z=0 (정지)
            return

        # 마지막으로 계수를 수신한지 일정 시간이 지나면 (차선 정보 손실) 정지
        if (self.get_clock().now() - self.last_command_time).nanoseconds / 1e9 > 0.5: # 0.5초 이상 정보 없음
            self.get_logger().warn("Lane coefficients lost. Stopping the robot.")
            self.current_coeffs = None # 계수 초기화하여 다음에도 경고 발생
            self.publisher.publish(twist_msg) # linear.x=0, angular.z=0 (정지)
            return

        A, B, C = self.current_coeffs

        # --------------------------------------------------
        # 파란선 계수(기울기, 곡률) 이상 감지 시 정지 로직 (유지)
        if abs(A) > self.curvature_threshold_A or abs(B) > self.slope_threshold_B:
            self.get_logger().warn(
                f"Abnormal lane coefficients detected (A: {A:.5f}, B: {B:.2f}). Stopping the robot."
            )
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.publisher.publish(twist_msg)
            return # 정지 명령 후 더 이상의 제어 계산 없이 종료
        # --------------------------------------------------

        # --------------------------------------------------
        # 1. 횡방향 오차(Lateral Error) 계산 (⚡️ 전방 주시점 활용)
        # --------------------------------------------------
        # ⚡️ 전방 주시점(y_lookahead_point)에서의 주행선 x 좌표 계산
        predicted_x_at_lookahead = A * (self.y_lookahead_point**2) + B * self.y_lookahead_point + C
        
        # 이미지 중앙 x 좌표와의 차이가 횡방향 오차
        lateral_error = self.center_x - predicted_x_at_lookahead
        # lateral_error가 양수면 주행선이 이미지 중앙보다 왼쪽에 있다는 의미 (차량을 왼쪽으로 돌려야 함 -> angular.z 증가)
        # lateral_error가 음수면 주행선이 이미지 중앙보다 오른쪽에 있다는 의미 (차량을 오른쪽으로 돌려야 함 -> angular.z 감소)


        # --------------------------------------------------
        # 2. 방향 오차(Heading Error) 계산 (⚡️ 전방 주시점에서의 기울기 활용)
        # --------------------------------------------------
        # ⚡️ 전방 주시점에서의 주행선 기울기 (dx/dy = 2Ay + B)
        # 이 기울기는 주행선의 현재 방향을 나타냅니다.
        slope_at_lookahead = 2 * A * self.y_lookahead_point + B
        
        # ⚡️ 방향 오차로 기울기를 직접 사용
        # 기울기가 양수면 차선이 오른쪽으로 향하고 있음 (로봇을 왼쪽으로 돌려야 함 -> angular.z 증가)
        # 기울기가 음수면 차선이 왼쪽으로 향하고 있음 (로봇을 오른쪽으로 돌려야 함 -> angular.z 감소)
        heading_error = slope_at_lookahead

        # ⚡️ 곡률을 직접 각속도 계산에 반영 (추가적인 제어 항)
        # A가 양수면 오른쪽으로 휘는 곡선 (카메라 시점에서), A가 음수면 왼쪽으로 휘는 곡선
        # A가 양수일 때 angular.z는 음수로 (오른쪽으로 조향), A가 음수일 때 angular.z는 양수로 (왼쪽으로 조향)
        # 따라서 -A에 Kp_curvature를 곱합니다.
        curvature_control_term = -A * self.Kp_curvature


        # --------------------------------------------------
        # 3. PID 제어 및 각속도 계산
        # --------------------------------------------------
        dt = (self.get_clock().now() - self.last_command_time).nanoseconds / 1e9
        if dt == 0: dt = 0.001 # 0으로 나누는 것 방지

        # 횡방향 오차의 미분항 계산
        derivative_lateral_error = (lateral_error - self.previous_lateral_error) / dt
        self.previous_lateral_error = lateral_error

        # ⚡️ 각속도 계산: 횡방향 오차 + 미분항 + 방향 오차(기울기) + 곡률 제어 항
        angular_z = (self.Kp_lateral * lateral_error) + \
                    (self.Kd_lateral * derivative_lateral_error) + \
                    (self.Kp_heading * heading_error) + \
                    curvature_control_term # ⚡️ 곡률 항 추가

        # 각속도 제한 (안정성을 위해)
        max_angular_z = 1.0 # rad/s
        angular_z = np.clip(angular_z, -max_angular_z, max_angular_z)

        # Twist 메시지 발행
        twist_msg.angular.z = float(angular_z)
        self.publisher.publish(twist_msg)

        self.get_logger().info(
            f"L_Err: {lateral_error:.2f}, H_Err(Slope): {heading_error:.2f}, "
            f"Curv_A: {A:.5f}, Angular Z: {angular_z:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    track_moving_node = TrackMovingNode()
    try:
        rclpy.spin(track_moving_node)
    except KeyboardInterrupt:
        pass
    finally:
        track_moving_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()