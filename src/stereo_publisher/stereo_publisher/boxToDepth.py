import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from stereo_interfaces.msg import Detections

class BoxToDepth(Node):
    def __init__(self):
        super().__init__('box_to_depth')
        self.points_sub = self.create_subscription(
            PointCloud2, '/points2', self.cloud_cb, 10)
        # self.cursor_sub = self.create_subscription(
        #     Point, '/left/image_rect_mouse_left', self.cursor_cb, 10)
        # self.box_sub = self.create_subscription(Detections, 
        #                                         "detect",
        #                                         self.listenerCb,
        #                                         10)
        self.create_timer(1, self.cursor_cb)
        self.box_pub = self.create_publisher(Detections,
                                             "detect",
                                             10)
        self.pc = None

    def cloud_cb(self, msg):
        self.pc = msg

    def cursor_cb(self): #, msg):
        # # if self.pc is None:
        # #     self.get_logger().warn('No point cloud received yet')
        # #     return
        
        # msg = {
        #     x: self.pc.width / 2,
        #     y: self.pc.height / 2,
        #     height: self.pc.height / 10,
        #     width: self.pc.width / 10
        # }

        boxMsg = Detections()
        boxMsg.id = "detecting1"
        boxMsg.center.x = 10.0
        boxMsg.center.y = 20.0
        boxMsg.height = 3
        boxMsg.width = 4

        self.box_pub.publish(boxMsg)

        # centreX = int(msg.x)
        # centreY = int(msg.y)
        # height = int(msg.height)
        # width = int(msg.width)
        # vTopRight = height / 2 + centreY
        # uTopRight = width / 2 + centreX
        # vBottomLeft = height / 2 - centreY
        # uBottomLeft = width / 2 - centreX
        # diagCoord = [[vTopRight, uTopRight], [vBottomLeft, uBottomLeft]]
        # for v, u in diagCoord:
        #     if u < 0 or v < 0 or u >= self.pc.width or v >= self.pc.height:
        #         self.get_logger().warn(f'Pixel ({u},{v}) out of range')
        #         return

        # sumDepths = 0
        # validCount = 0
        # points = pc2.read_points(self.pc, field_names=("x", "y", "z"), skip_nans=False)
        # for row in range(vBottomLeft, vTopRight + 1):
        #     for col in range(uBottomLeft, uTopRight + 1):
        #         index = row * self.pc.width + col
        #         x, y, z = points[index]
        #         if z<100 and z>0:
        #             validCount += 1
        #             sumDepths += z
        
        # avgDepth = sumDepths / validCount
        # self.get_logger().info(f'Centre ({centreX},{centreY}), Height ({height}), Width ({width}) → Depth: {avgDepth:.3f}')

        # #for i, p in enumerate(pc2.read_points(self.pc, field_names=("x", "y", "z"), skip_nans=False)):
        #     #if i == index:
        #         #x, y, z = p
        #         #self.get_logger().info(f'Pixel ({u},{v}) → 3D point: x={x:.3f}, y={y:.3f}, z={z:.3f}')
        #         #break

def main():
    rclpy.init()
    node = BoxToDepth()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
