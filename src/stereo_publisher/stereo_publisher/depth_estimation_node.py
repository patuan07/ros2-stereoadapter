import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection2DArray
from stereo_interfaces.msg import Detection, DetectionArray
from message_filters import TimeSynchronizer, ApproximateTimeSynchronizer
import message_filters
from sensor_msgs_py import point_cloud2 as pc2
import numpy as np


# ---------------------------------------------------------------------------
# Projection matrix (P) from your left camera calibration.
# ---------------------------------------------------------------------------
P = np.array([
    [801.326263,   0.0,       210.018585],
    [  0.0,       801.326263, 239.742247],
    [  0.0,         0.0,        1.0     ]
])


def filter_points_mad(depth_points, k=2.0):
    """
    Filters out background points using Median Absolute Deviation on the
    z-axis. Keeps points within k scaled-MAD units of the median depth.
    """
    if len(depth_points) == 0:
        return depth_points

    z_values = depth_points[:, 2]
    median_z = np.median(z_values)
    mad_z = np.median(np.abs(z_values - median_z))
    sigma_z = 1.4826 * mad_z

    if sigma_z < 1e-6:
        return depth_points

    mask = np.abs(z_values - median_z) < k * sigma_z
    return depth_points[mask]


def estimate_position_symmetry(filtered_points, bbox_center_uv, P_matrix):
    """
    Estimates the object's true 3D center by back-projecting the 2D bounding
    box center using depth from the filtered stereo points.
    """
    fx = P_matrix[0, 0]
    fy = P_matrix[1, 1]
    cx = P_matrix[0, 2]
    cy = P_matrix[1, 2]

    z_estimate = np.median(filtered_points[:, 2])
    u_center, v_center = bbox_center_uv

    x_center = (u_center - cx) * z_estimate / fx
    y_center = (v_center - cy) * z_estimate / fy

    return np.array([x_center, y_center, z_estimate])


class DepthEstimationNode(Node):
    def __init__(self):
        super().__init__('depth_estimation_node')

        self.declare_parameter('mad_k', 2.0)
        self.declare_parameter('min_points', 10)
        self.mad_k = self.get_parameter('mad_k').value
        self.min_points = self.get_parameter('min_points').value

        # Subscribers
        self.det_sub = message_filters.Subscriber(self, Detection2DArray, '/detections')
        self.pc_sub = message_filters.Subscriber(self, PointCloud2, '/points2')

        self.sync = ApproximateTimeSynchronizer(
            [self.det_sub, self.pc_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.sync_cb)

        # Publisher using custom DetectionArray message
        self.position_pub = self.create_publisher(DetectionArray, '/detections_3d', 10)

    def sync_cb(self, det_msg, pc_msg):
        # Convert point cloud to numpy array (H x W x 3)
        cloud_arr = pc2.read_points_numpy(pc_msg, field_names=('x', 'y', 'z'), skip_nans=False)
        cloud_arr = cloud_arr.reshape((pc_msg.height, pc_msg.width, 3))

        # Create output DetectionArray
        det_array = DetectionArray()
        det_array.header = pc_msg.header

        for det in det_msg.detections:
            # Extract bounding box pixel bounds
            u_center = det.bbox.center.position.x
            v_center = det.bbox.center.position.y
            half_w = det.bbox.size_x / 2.0
            half_h = det.bbox.size_y / 2.0

            u_min, u_max = int(u_center - half_w), int(u_center + half_w)
            v_min, v_max = int(v_center - half_h), int(v_center + half_h)

            # Clamp to image bounds
            u_min = max(u_min, 0)
            v_min = max(v_min, 0)
            u_max = min(u_max, pc_msg.width - 1)
            v_max = min(v_max, pc_msg.height - 1)

            # Slice the 3D points directly using numpy
            box_points = cloud_arr[v_min:v_max+1, u_min:u_max+1].reshape(-1, 3)

            # Remove NaN points
            valid_mask = ~np.isnan(box_points).any(axis=1)
            box_points = box_points[valid_mask]

            if len(box_points) < self.min_points:
                continue

            # Filter and estimate position
            filtered_points = filter_points_mad(box_points, k=self.mad_k)
            if len(filtered_points) < self.min_points:
                continue

            position = estimate_position_symmetry(filtered_points, (u_center, v_center), P)

            # Create Detection message with custom format
            detection = Detection()
            
            # Extract class name from the 2D detection
            if len(det.results) > 0:
                detection.class_name = det.results[0].hypothesis.class_id
                detection.confidence = det.results[0].hypothesis.score
            else:
                detection.class_name = "unknown"
                detection.confidence = 0.0

            # Set 3D position
            detection.position.x = float(position[0])
            detection.position.y = float(position[1])
            detection.position.z = float(position[2])

            det_array.detections.append(detection)

            # Log detection
            self.get_logger().info(
                f'Found {detection.class_name} at: '
                f'x={position[0]:.2f}m, y={position[1]:.2f}m, z={position[2]:.2f}m, '
                f'conf={detection.confidence:.2f}'
            )

        if det_array.detections:
            self.position_pub.publish(det_array)


def main():
    rclpy.init()
    node = DepthEstimationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()