import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import time

class CmdVelSelector(Node):
    def __init__(self):
        super().__init__('cmd_vel_selector')

        self.lf_cmd = None
        self.of_cmd = None
        self.object_active = False

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_subscription(Twist, '/cmd_vel_lf', self.lf_callback, 10)
        self.create_subscription(Twist, '/cmd_vel_of', self.of_callback, 10)
        self.create_subscription(Bool, '/object_tracking/active', self.object_active_callback, 10)

        self.create_timer(0.05, self.select_and_publish_cmd)

        # 전진 임시 타이머 관련 변수
        self.forwarding = False
        self.forward_start_time = None
        self.forward_duration = 2.0  # 초 단위

        self.get_logger().info("CmdVelSelector 노드 시작됨")

    def lf_callback(self, msg):
        self.lf_cmd = msg

    def of_callback(self, msg):
        self.of_cmd = msg

    def object_active_callback(self, msg):
        self.object_active = msg.data

    def select_and_publish_cmd(self):
        now = time.time()

#        if self.forwarding:
 #           elapsed = now - self.forward_start_time
  #          if elapsed < self.forward_duration:
   #             cmd = Twist()
    #            cmd.linear.x = 0.05  # 일정 속도로 전진
     #           cmd.angular.z = 0.0
      #          self.cmd_pub.publish(cmd)
       #         self.get_logger().info(f"⚠️ 전진 유지 중 ({elapsed:.1f}/{self.forward_duration:.1f}s)")
        #        return
         #   else:
          #      self.forwarding = False
           #     self.get_logger().info("✅ 전진 종료")

        if self.lf_cmd is None:
            return

        # 라인팔로우와 오브젝트팔로우 모두 멈췄다면 → 전진 모드 진입
#        if not self.object_active and self.of_cmd is not None and self.of_cmd.linear.x == 0.0 and self.lf_cmd.linear.x == 0.0:
 #           self.forwarding = True
  #          self.forward_start_time = now
   #         self.get_logger().warn("⏳ 탐지 실패 → 임시 직진 시작")
    #        return

        # 평상시 로직
        if self.object_active and self.of_cmd is not None:
            angular_z = abs(self.of_cmd.angular.z)
            if angular_z > 0.3:
                cmd = self.lf_cmd
                self.get_logger().info("선회 중 → 라인팔로우 사용")
            else:
                cmd = self.of_cmd
                self.get_logger().info("오브젝트 추적 중 → 오브젝트 명령 사용")
        else:
            cmd = self.lf_cmd
            self.get_logger().info("오브젝트 없음 → 라인팔로우 사용")

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = CmdVelSelector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
