import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from xycar_msgs.msg import XycarMotor


HELP = """
Xycar Step Steering Teleop Control (Auto-Center on Accel/Brake)
---------------------------------------------------------------
w : speed stage up & center steering   (0 -> 10 -> 20 -> 30 km/h)
s : speed stage down & center steering (30 -> 20 -> 10 -> 0 km/h)

a : steering stage left  ( 100 ->   0 -> -100 deg)
d : steering stage right (-100 ->   0 ->  100 deg)

Steering does not auto-center while idle.
Pressing w or s centers steering immediately.

spacebar : immediate stop and center steering
CTRL-C   : quit
"""


class XycarStepSteerTeleop(Node):
    def __init__(self):
        super().__init__("xycar_step_steer_teleop_node")
        self.publisher_ = self.create_publisher(XycarMotor, "/xycar_motor", 10)

        self.speed_stages = [0.0, 10.0, 20.0, 30.0]
        self.current_speed_idx = 0

        self.angle_stages = [-100.0, 0.0, 100.0]
        self.current_angle_idx = 1

        self.scale_factor = 2.0
        self.settings = termios.tcgetattr(sys.stdin)

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ""
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def publish_motor(self, speed: float, angle: float) -> None:
        motor_msg = XycarMotor()
        motor_msg.header.stamp = self.get_clock().now().to_msg()
        motor_msg.angle = float(angle)
        motor_msg.speed = float(speed)
        self.publisher_.publish(motor_msg)

    def run(self):
        print(HELP)
        print("publishing xycar_msgs/msg/XycarMotor on /xycar_motor")
        last_key = ""

        try:
            while rclpy.ok():
                key = self.get_key()

                if key == "\x03":
                    break

                if key == "w" and last_key != "w":
                    if self.current_speed_idx < len(self.speed_stages) - 1:
                        self.current_speed_idx += 1
                    self.current_angle_idx = 1
                elif key == "s" and last_key != "s":
                    if self.current_speed_idx > 0:
                        self.current_speed_idx -= 1
                    self.current_angle_idx = 1

                if key == "a" and last_key != "a":
                    if self.current_angle_idx > 0:
                        self.current_angle_idx -= 1
                elif key == "d" and last_key != "d":
                    if self.current_angle_idx < len(self.angle_stages) - 1:
                        self.current_angle_idx += 1

                if key in ["w", "s", "a", "d"]:
                    last_key = key
                elif key == "":
                    last_key = ""

                if key == " ":
                    self.current_speed_idx = 0
                    self.current_angle_idx = 1

                target_kmh = self.speed_stages[self.current_speed_idx]
                target_mps = target_kmh / 3.6
                final_speed_cmd = target_mps * self.scale_factor
                target_angle = self.angle_stages[self.current_angle_idx]

                self.publish_motor(final_speed_cmd, target_angle)
                print(
                    f"Speed Stage: {self.current_speed_idx} ({target_kmh:2.0f} km/h) | "
                    f"Angle Stage: {self.current_angle_idx} ({target_angle:6.1f} deg)",
                    end="\r",
                    flush=True,
                )
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.publish_motor(0.0, 0.0)
            except Exception:
                pass
            print("\nStopped and exiting...")


def main(args=None):
    rclpy.init(args=args)
    teleop_node = XycarStepSteerTeleop()
    try:
        teleop_node.run()
    finally:
        teleop_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
