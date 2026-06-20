from __future__ import annotations

import argparse
import select
import sys
import termios
import tty


HELP = """
ROS2 Xycar teleop

keys:
  w / s : speed up / speed down
  a / d : steer left / steer right
  x     : center steering
  space : stop speed
  q     : quit
"""


class RawTerminal:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


def read_key(timeout_s: float) -> str:
    readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not readable:
        return ""
    return sys.stdin.read(1)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def assign_field(msg, names: tuple[str, ...], value: float) -> None:
    for name in names:
        if not hasattr(msg, name):
            continue
        current = getattr(msg, name)
        if isinstance(current, int):
            setattr(msg, name, int(round(value)))
        else:
            setattr(msg, name, type(current)(value))
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard teleop publisher for /xycar_motor.")
    parser.add_argument("--topic", default="/xycar_motor")
    parser.add_argument("--msg-type", default="xycar_msgs/msg/XycarMotor")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--speed-step", type=float, default=1.0)
    parser.add_argument("--steer-step", type=float, default=5.0)
    parser.add_argument("--max-speed", type=float, default=20.0)
    parser.add_argument("--max-angle", type=float, default=50.0)
    args = parser.parse_args()

    try:
        import rclpy
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise SystemExit("ROS2 rclpy is required. Run inside a sourced ROS2 environment.") from exc

    msg_cls = get_message(args.msg_type)
    rclpy.init()
    node = rclpy.create_node("xycar_keyboard_teleop")
    pub = node.create_publisher(msg_cls, args.topic, 10)
    period_s = 1.0 / max(args.rate, 1e-6)
    speed = 0.0
    angle = 0.0

    print(HELP)
    print(f"publishing {args.msg_type} on {args.topic}")
    try:
        with RawTerminal():
            while rclpy.ok():
                key = read_key(period_s)
                if key == "q":
                    break
                if key == "w":
                    speed += args.speed_step
                elif key == "s":
                    speed -= args.speed_step
                elif key == "a":
                    angle += args.steer_step
                elif key == "d":
                    angle -= args.steer_step
                elif key == "x":
                    angle = 0.0
                elif key == " ":
                    speed = 0.0

                speed = clamp(speed, 0.0, args.max_speed)
                angle = clamp(angle, -args.max_angle, args.max_angle)

                msg = msg_cls()
                assign_field(msg, ("angle", "steering", "steer"), angle)
                assign_field(msg, ("speed", "velocity", "throttle"), speed)
                pub.publish(msg)
                print(f"\rangle={angle:6.2f} speed={speed:6.2f}", end="", flush=True)
                rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        stop_msg = msg_cls()
        assign_field(stop_msg, ("angle", "steering", "steer"), 0.0)
        assign_field(stop_msg, ("speed", "velocity", "throttle"), 0.0)
        pub.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()
        print("\nstopped")


if __name__ == "__main__":
    main()
