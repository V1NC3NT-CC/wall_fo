import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from math import radians, isfinite


class WallFollow(Node):
    """
    - 订阅 /scan
    - 发布 /drive (AckermannDriveStamped)
    - 采用左侧两点法求 α，预瞄 L 推算未来距离 D_{t+1}
      error = d_desired - D_{t+1}
    - PID -> 转角（带饱和）；转角越小车速越高
    """
    def __init__(self):
        super().__init__('wall_follow_node')

        # topics
        self.lidar_topic = '/scan'
        self.drive_topic = '/drive'

        # pub/sub
        self.sub_scan = self.create_subscription(
            LaserScan, self.lidar_topic, self.scan_callback, 10
        )
        self.pub_drive = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10
        )

        # PID gains（按需改/可改成 declare_parameter 再从 YAML 覆盖）
        self.kp = 2.0
        self.kd = 0.4
        self.ki = 0.0

        # PID states
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

        # 目标左墙距离（米）与预瞄距离（米）
        self.desired_dist = 0.9
        self.lookahead_L = 1.0

        # LiDAR 元信息（在第一次回调里填）
        self.angle_min = None
        self.angle_inc = None

        # 机械限幅（F1TENTH 常用转向最大约 0.418rad ≈ 24度）
        self.max_steer = 0.418

        self.get_logger().info('WallFollow node started')

    # ---------- helpers ----------
    def _clamp(self, x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def get_range(self, ranges, angle_rad):
        """
        返回给定“雷达坐标系角度”的量测值（单位：米）
        - angle_rad: 以**车体前方为 0**，逆时针为正（左侧是 ~+90°）
        - 内部用到 self.angle_min / self.angle_inc（来自 LaserScan 首帧）
        - 自动跳过 NaN/inf/<=0 的异常值，向邻近窗口寻找可用量测
        """
        if self.angle_min is None or self.angle_inc is None:
            return np.nan

        idx = int(round((angle_rad - self.angle_min) / self.angle_inc))
        n = len(ranges)
        idx = self._clamp(idx, 0, n - 1)

        # 若该处无效，则在邻域内搜索最近的有效值
        if not (0 <= idx < n):
            return np.nan
        if isfinite(ranges[idx]) and ranges[idx] > 0.0:
            return float(ranges[idx])

        # 邻域搜索
        for off in range(1, 6):  # 最多查±5个bin
            for j in (idx - off, idx + off):
                if 0 <= j < n and isfinite(ranges[j]) and ranges[j] > 0.0:
                    return float(ranges[j])

        # 实在找不到，返回一个保守的大值
        return 10.0

    def get_error(self, ranges, desired_dist):
        """
        左墙两点法（a=60°, b=90°）：
          θ = b - a = 30°
          alpha = atan((a*cosθ - b)/(a*sinθ))
          D = b * cos(alpha)
          D_future = D + L * sin(alpha)
          error = desired_dist - D_future
        """
        theta = radians(30.0)
        # 左侧：+90° 和 +60°
        b = self.get_range(ranges, radians(90.0))
        a = self.get_range(ranges, radians(60.0))

        # 保护
        if not isfinite(a) or not isfinite(b) or a <= 0.0:
            return 0.0

        alpha = np.arctan2(a * np.cos(theta) - b, a * np.sin(theta))
        D = b * np.cos(alpha)
        D_future = D + self.lookahead_L * np.sin(alpha)
        error = desired_dist - D_future
        return float(error)

    def pid_control(self, error):
        """
        用 PID 计算转角，并联动给速度（角度小→速度大）
        返回 (steer_angle, speed)
        """
        now = self.get_clock().now().nanoseconds / 1e9
        if self.prev_time is None:
            dt = 0.02  # 假设 50Hz
        else:
            dt = max(1e-3, now - self.prev_time)
        self.prev_time = now

        # PID
        self.integral += error * dt
        deriv = (error - self.prev_error) / dt
        self.prev_error = error

        u = self.kp * error + self.ki * self.integral + self.kd * deriv

        # 左墙：若 error>0（离墙太近/或几何推导结果），需要向右打角（负）
        steer = -u
        steer = self._clamp(steer, -self.max_steer, self.max_steer)

        # 简单速度逻辑：角度越小开得越快
        abs_s = abs(steer)
        if abs_s < 0.05:
            speed = 2.0
        elif abs_s < 0.15:
            speed = 1.5
        else:
            speed = 1.0

        return steer, speed

    # ---------- callbacks ----------
    def scan_callback(self, msg: LaserScan):
        # 记录 LiDAR 元信息（只需记录一次）
        if self.angle_min is None:
            self.angle_min = msg.angle_min
            self.angle_inc = msg.angle_increment

        # 1) 算误差（基于左墙）
        error = self.get_error(msg.ranges, self.desired_dist)

        # 2) PID -> 转角 + 速度
        steer, speed = self.pid_control(error)

        # 3) 发布控制
        drive = AckermannDriveStamped()
        drive.header.stamp = self.get_clock().now().to_msg()
        drive.drive.steering_angle = float(steer)
        drive.drive.speed = float(speed)
        self.pub_drive.publish(drive)

        # 调试输出（可注释）
        self.get_logger().info(f"err={error:+.3f}  steer={steer:+.3f}  v={speed:.1f}")


def main(args=None):
    rclpy.init(args=args)
    node = WallFollow()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
