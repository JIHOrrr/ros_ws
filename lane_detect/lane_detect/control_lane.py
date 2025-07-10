#!/usr/bin/env python3

from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, UInt8
import rclpy
from rclpy.node import Node


class ControlLane(Node):
    def __init__(self):
        super().__init__('control_lane')

        # 구독 토픽들
        self.create_subscription(Float64, '/detect/lane', self.callback_lane_center, 1)
        self.create_subscription(UInt8, '/detect/yellow_line_reliability', self.callback_yellow_conf, 1)
        self.create_subscription(UInt8, '/detect/white_line_reliability', self.callback_white_conf, 1)
        self.create_subscription(UInt8, '/detect/lane_state', self.callback_lane_status, 1)

        # 속도 발행 퍼블리셔
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_lf', 10)

        # 내부 상태값 저장
        self.lane_center = 160.0 # 320x240 해상도의 중심
        self.yellow_conf = 100.0
        self.white_conf = 100.0
        self.lane_status = 0

        # 제어 변수
        self.last_error = 0.0
        self.MAX_VEL = 0.12

        # 주기적으로 제어 수행
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20Hz

    def callback_lane_center(self, msg):
        self.lane_center = msg.data

    def callback_yellow_conf(self, msg):
        self.yellow_conf = msg.data

    def callback_white_conf(self, msg):
        self.white_conf = msg.data

    def callback_lane_status(self, msg):
        self.lane_status = msg.data

    def control_loop(self):
        twist = Twist()

        # 주행모션(1)
        # if self.lane_status == 0:
        #     # 🔁 모든 차선을 인식하지 못함 — 신뢰도 비교하여 회전 방향 결정
        #     if self.yellow_conf < self.white_conf:
        #         # 노란선 신뢰도가 낮음 → 왼쪽 회전
        #         twist.linear.x = 0.5 # m/s
        #         twist.angular.z = 0.45 # rad/s x * 180/pi
        #     elif self.white_conf < self.yellow_conf:
        #         # 흰 선 신뢰도가 낮음 → 오른쪽 회전
        #         twist.linear.x = 0.5
        #         twist.angular.z = -0.45
        #     else:
        #         # 둘 다 같거나 판단 불가 → 정지
        #         twist.linear.x = 0.0
        #         twist.angular.z = 0.0

        # elif self.lane_status == 1:
        #     # 왼쪽 차선만 탐지
        #     if self.white_conf <= 25.0:
        #         # 오른쪽으로 회전
        #         twist.linear.x = 0.1
        #         twist.angular.z = -0.25
        #     else:
        #         twist.linear.x = 0.1
        #         twist.angular.z = 0.0

        # elif self.lane_status == 2:
        #     # 모든 차선 탐지 → PD 제어
        #     error = self.lane_center - 160
        #     Kp = 0.005 # 증가시킬 수록 반응성 향상
        #     Kd = 0.01 # 흔들림 방지 조정
        #     angular_z = Kp * error + Kd * (error - self.last_error)
        #     self.last_error = error

        #     twist.linear.x = min(self.MAX_VEL * (max(1 - abs(error) / 500, 0) ** 2.2), 0.1)
        #     twist.angular.z = -max(angular_z, -2.0) if angular_z < 0 else -min(angular_z, 2.0)

        # elif self.lane_status == 3:
        #     # 오른쪽 차선만 탐지
        #     if self.yellow_conf <= 25.0:
        #         # 왼쪽으로 회전
        #         twist.linear.x = 0.1
        #         twist.angular.z = 0.25
        #     else:
        #         twist.linear.x = 0.1
        #         twist.angular.z = 0.0
        
        # 주행모션(2)
        if self.lane_status == 2 and self.yellow_conf > 50.0 and self.white_conf > 50.0:
            # 양쪽 차선 감지, 신뢰도 충분 → PD 제어
            error = self.lane_center - 160
            Kp = 0.005  # 반응성 향상
            Kd = 0.01   # 흔들림 방지 조정
            angular_z = Kp * error + Kd * (error - self.last_error)
            self.last_error = error

            twist.linear.x = min(self.MAX_VEL * (max(1 - abs(error) / 160, 0) ** 2.0), 0.1)
            twist.angular.z = -max(angular_z, -2.0) if angular_z < 0 else -min(angular_z, 2.0)
        else:
            # 차선 미감지 또는 신뢰도 낮음 → 신뢰도 비교로 회전
            if self.yellow_conf < self.white_conf :
                # 노란색 차선 신뢰도 낮음 → 왼쪽 회전
                twist.linear.x = 0.1  # 협소한 도로에 맞게 속도 감소
                twist.angular.z = 0.2   # 부드러운 회전
            elif self.white_conf < self.yellow_conf :
                # 흰색 차선 신뢰도 낮음 → 오른쪽 회전
                twist.linear.x = 0.1
                twist.angular.z = -0.2
            else:
                # 둘 다 신뢰도 낮거나 판단 불가 → 정지
                twist.linear.x = 0.0
                twist.angular.z = 0.0

        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = ControlLane()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
