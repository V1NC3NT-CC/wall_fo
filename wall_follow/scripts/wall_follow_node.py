import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import numpy as np
from math import atan2, cos, sin, isfinite
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class WallFollow(Node):
    """
    ROS2 Wall Following (Python)
    - 订阅 /scan（Best Effort QoS）
    - 预处理：截取 ±(truncated_coverage_angle/2) 的扇区 + 简单平滑
    - 两点法（a=0.5rad, b=1.4rad, theta=0.9rad）估计 alpha、预测距离 D_{t+1}
    - 误差: e = D_{t+1} - desired_distance_left
    - PID -> 转角（限幅 ±0.4），按角度分档给速度
    """
    def __init__(self):
        super().__init__('wall_follow_node')

        # —— Topics（可运行时 remap）——
        self.lidar_topic = self.declare_parameter('scan_topic', '/scan').get_parameter_value().string_value
        self.drive_topic = self.declare_parameter('drive_topic', '/drive').get_parameter_value().string_value

        # —— PID 与几何参数（可从 YAML 覆盖）——
        self.kp = self.declare_parameter('kp', 1.5).get_parameter_value().double_value
        self.ki = self.declare_parameter('ki', 0.0).get_parameter_value().double_value
        self.kd = self.declare_parameter('kd', 0.3).get_parameter_value().double_value

        self.desired_left = self.declare_parameter('desired_distance_left', 0.9).get_parameter_value().double_value
        self.lookahead_L  = self.declare_parameter('lookahead_distance', 1.0).get_parameter_value().double_value

        # 截取的扇区宽度（弧度），需覆盖到 b=1.4rad：默认 π（±90°）
        self.trunc_angle  = self.declare_parameter('truncated_coverage_angle', float(np.pi)).get_parameter_value().double_value
        self.smooth_N     = int(self.declare_parameter('smoothing_filter_size', 5).get_parameter_value().integer_value)

        # 速度分档（角度越大→用较高速度变量名，仅作占位，可按需调整含义）
        self.v_high = self.declare_parameter('vel_high',   1.5).get_parameter_value().double_value
        self.v_med  = self.declare_parameter('vel_medium', 1.0).get_parameter_value().double_value
        self.v_low  = self.declare_parameter('vel_low',    0.5).get_parameter_value().double_value

        # 状态
        self.prev_error = 0.0
        self.integral   = 0.0
        self.prev_t     = None

        self.max_steer = 0.4
        self.theta     = 0.9
        self.a_angle   = 0.5
        self.b_angle   = 1.4

        # 订阅/发布
        self.sub_scan = self.create_subscription(
            LaserScan, self.lidar_topic, self.scan_callback, qos_profile_sensor_data
        )
        self.pub_drive = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10
        )

        self.get_logger().info(
            f"WallFollow up. scan={self.lidar_topic} drive={self.drive_topic} "
            f"kp={self.kp} kd={self.kd} ki={self.ki} desired_left={self.desired_left}"
        )

    # ---------- utils ----------
    def _moving_avg(self, arr: np.ndarray, w: int) -> np.ndarray:
        if w <= 1:
            return arr
        w = max(1, int(w))
        kernel = np.ones(w, dtype=float) / w
        pad = w // 2
        arrp = np.pad(arr, (pad, pad), mode='edge')
        return np.convolve(arrp, kernel, mode='valid')

    def _truncate_ranges(self, msg: LaserScan) -> (np.ndarray, float):
        """截取以前向为0的对称扇区：[-trunc/2, +trunc/2]。返回(truncated_ranges, angle_increment)"""
        angle_min = msg.angle_min
        inc       = msg.angle_increment
        full_min  = -self.trunc_angle / 2.0
        full_max  = +self.trunc_angle / 2.0

        i0 = int(np.ceil((full_min - angle_min) / inc))
        i1 = int(np.floor((full_max - angle_min) / inc))
        i0 = max(0, i0)
        i1 = min(len(msg.ranges) - 1, i1)
        if i1 < i0:
            ranges = np.array(msg.ranges, dtype=float)
        else:
            ranges = np.array(msg.ranges[i0:i1 + 1], dtype=float)

        # NaN/inf/非正 处理为 0
        bad = ~np.isfinite(ranges) | (ranges <= 0.0)
        ranges[bad] = 0.0

        # 平滑
        ranges = self._moving_avg(ranges, self.smooth_N)
        return ranges, inc

    def _get_range_at(self, filtered: np.ndarray, angle: float, angle_increment: float) -> float:
        """在截取后的扇区内获得 angle（以车前为0，左为正）的距离。"""
        corrected = angle + (self.trunc_angle / 2.0)
        idx = int(np.floor(corrected / max(angle_increment, 1e-6)))
        idx = max(0, min(idx, len(filtered) - 1))
        return float(filtered[idx])

    # ---------- core ----------
    def _compute_error(self, filtered: np.ndarray, angle_increment: float) -> float:
        a = self._get_range_at(filtered, self.a_angle, angle_increment)
        b = self._get_range_at(filtered, self.b_angle, angle_increment)
        if a <= 0.0 or b <= 0.0 or not (isfinite(a) and isfinite(b)):
            return 0.0

        alpha = atan2(a * cos(self.theta) - b, a * sin(self.theta))
        D     = b * cos(alpha)
        D1    = D + self.lookahead_L * sin(alpha)
        err   = D1 - self.desired_left
        return err

    def _pid(self, err: float) -> (float, float):
        now = self.get_clock().now().nanoseconds / 1e9
        dt  = 0.02 if self.prev_t is None else max(1e-3, now - self.prev_t)
        self.prev_t = now

        self.integral += err * dt
        deriv = (err - self.prev_error) / dt
        self.prev_error = err

        steer = self.kp * err + self.kd * deriv + self.ki * self.integral
        steer = max(-self.max_steer, min(self.max_steer, steer))

        # 分档速度（可按需求改成“角度越大→越慢”等策略）
        if abs(steer) > 0.349:
            speed = self.v_high
        elif abs(steer) > 0.174:
            speed = self.v_med
        else:
            speed = self.v_low
        return steer, speed

    # ---------- callbacks ----------
    def scan_callback(self, msg: LaserScan):
        filtered, inc = self._truncate_ranges(msg)
        err = self._compute_error(filtered, inc)
        steer, speed = self._pid(err)

        drive = AckermannDriveStamped()
        drive.header.stamp = self.get_clock().now().to_msg()
        drive.header.frame_id = 'laser'
        drive.drive.steering_angle = float(steer)
        drive.drive.speed = float(speed)
        self.pub_drive.publish(drive)

        self.get_logger().info(f"e={err:+.3f} steer={steer:+.3f} v={speed:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = WallFollow()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
