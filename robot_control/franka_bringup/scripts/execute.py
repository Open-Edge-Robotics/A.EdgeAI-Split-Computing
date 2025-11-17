
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import subprocess
import time

class CaseControllerNode(Node):
    def __init__(self):
        super().__init__('case_controller_node')
        self.subscription = None
        self.current_class = None
        self.start_subscription()

    def start_subscription(self):
        self.subscription = self.create_subscription(
            String,
            '/predicted_case_info', 
            self.listener_callback,
            10
        )
        self.get_logger().info("Started subscription.")

    def stop_subscription(self):
        if self.subscription:
            self.destroy_subscription(self.subscription)
            self.subscription = None
            self.get_logger().info("Stopped subscription.")

    def listener_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.current_class = data.get("predicted_class")
            if self.current_class is not None:
                self.get_logger().info(f"Received predicted class: {self.current_class}")
                self.stop_subscription()  # 구독 멈추기
                self.run_cycle()
        except Exception as e:
            self.get_logger().error(f"Failed to parse message: {e}")

    def run_controller(self, controller_name):
        while True:
            try:
                self.get_logger().info(f"Executing controller: {controller_name}")
                result = subprocess.run([
                    "ros2", "launch", "franka_bringup", "example.launch.py",
                    f"controller_name:={controller_name}"
                ])
                if result.returncode == 0:
                    self.get_logger().info(f"{controller_name} finished successfully")
                    break
                else:
                    self.get_logger().warn(f"{controller_name} exited with code {result.returncode}, retrying...")
            except Exception as e:
                self.get_logger().error(f"Error running {controller_name}: {e}, retrying...")
            time.sleep(1)

    def run_cycle(self):
        case_controller = f"move_to_case{self.current_class}_controller"

        # self.run_controller("open_gripper_controller")

        self.run_controller(case_controller)

        self.run_controller("gripper_example_controller")

        self.run_controller("move_to_start_example_controller")

        self.run_controller("open_gripper_controller")

        input("Press Enter to start next cycle...")
        self.start_subscription()

def main(args=None):
    rclpy.init(args=args)
    node = CaseControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

