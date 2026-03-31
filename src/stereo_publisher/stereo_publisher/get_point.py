import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

class CursorTo3D(Node):
    def __init__(self):
        super().__init__('cursor_to_3d')
        self.points_sub = self.create_subscription(
            PointCloud2, '/points2', self.cloud_cb, 10)
        self.cursor_sub = self.create_subscription(
            Point, '/left/image_rect_mouse_left', self.cursor_cb, 10)
        self.pc = None

    def cloud_cb(self, msg):
        self.pc = msg

    def cursor_cb(self, msg):
        if self.pc is None:
            self.get_logger().warn('No point cloud received yet')
            return

        u = int(msg.x)
        v = int(msg.y)
        if u < 0 or v < 0 or u >= self.pc.width or v >= self.pc.height:
            self.get_logger().warn(f'Pixel ({u},{v}) out of range')
            return

        index = v * self.pc.width + u
        for i, p in enumerate(pc2.read_points(self.pc, field_names=("x", "y", "z"), skip_nans=False)):
            if i == index:
                x, y, z = p
                self.get_logger().info(f'Pixel ({u},{v}) → 3D point: x={x:.3f}, y={y:.3f}, z={z:.3f}')
                break

def main():
    rclpy.init()
    node = CursorTo3D()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
