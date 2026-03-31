#!/usr/bin/env python3
"""
ROS2 Depth Estimation Node using TiO_Depth model with LIVE VISUALIZATION.

Subscribes to:
    - /left/image_rect_color (sensor_msgs/Image)
    - /right/image_rect_color (sensor_msgs/Image)
    - /detections (vision_msgs/Detection2DArray)

Publishes:
    - /points2 (sensor_msgs/PointCloud2) - Full point cloud from depth map
    - /detections_3d (stereo_interfaces/DetectionArray) - 3D object positions

Shows:
    - Live depth map visualization window
"""
import os
import sys

from streamlit import header
package_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, package_dir)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField, CompressedImage
from stereo_interfaces.msg import Detection, DetectionArray
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import message_filters
from sensor_msgs_py import point_cloud2 as pc2
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from PIL import Image as PILImage


# Add current directory to path for model imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.get_models import get_model_with_opts
from saver import load_model_for_evaluate
from utils.platform_loader import read_yaml_options


# ---------------------------------------------------------------------------
# Camera Intrinsic Matrix (from your calibration)
# ---------------------------------------------------------------------------
P = np.array([
    [801.326263,   0.0,       210.018585],
    [  0.0,       801.326263, 239.742247],
    [  0.0,         0.0,        1.0     ]
])

# Extract camera parameters
fx = P[0, 0]
fy = P[1, 1]
cx = P[0, 2]
cy = P[1, 2]

def depth_to_pointcloud(depth_map, fx, fy, cx, cy):
    """
    Convert a depth map to a 3D point cloud using camera intrinsics.
    
    Args:
        depth_map: (H, W) numpy array of depth values
        fx, fy: focal lengths
        cx, cy: principal point coordinates
    
    Returns:
        points: (H, W, 3) numpy array with (x, y, z) coordinates
    """
    h, w = depth_map.shape
    
    # Create pixel coordinate grids
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)
    
    # Back-project to 3D
    z = depth_map
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack into (H, W, 3) array
    points = np.stack([x, y, z], axis=-1)
    
    return points


def colorize_depth(depth_map):
    """
    Convert depth map to a colorized visualization.
    
    Args:
        depth_map: (H, W) numpy array
    
    Returns:
        Colorized depth image (H, W, 3) BGR
    """
    # Normalize to 0-255
    depth_min = np.min(depth_map)
    depth_max = np.max(depth_map)
    
    if depth_max - depth_min < 1e-6:
        depth_norm = np.zeros_like(depth_map)
    else:
        depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
    
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    
    # Apply JET colormap (blue=close, red=far)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    
    return depth_color


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

    z_estimate = np.median(filtered_points[:, 2]) * 0.6
    u_center, v_center = bbox_center_uv

    x_center = (u_center - cx) * z_estimate / fx
    y_center = (v_center - cy) * z_estimate / fy

    return np.array([x_center, y_center, z_estimate])


class DepthEstimationNode(Node):
    """
    ROS2 node that:
    1. Runs TiO_Depth model on stereo images
    2. Publishes full point cloud to /points2
    3. Extracts 3D positions for detected objects
    4. Shows live depth visualization
    """

    def __init__(self):
        super().__init__('new_depth_estimation_node')

        # --- Parameters ---
        self.declare_parameter('model_path', '[insert model full path]')
        self.declare_parameter('exp_opts', '[insert yaml full path for inference config]')
        self.declare_parameter('mad_k', 2.0)
        self.declare_parameter('min_points', 10)
        self.declare_parameter('use_cuda', True)
        self.declare_parameter('publish_rate_hz', 30.0)
        
        model_path = self.get_parameter('model_path').value
        exp_opts = self.get_parameter('exp_opts').value
        self.mad_k = self.get_parameter('mad_k').value
        self.min_points = self.get_parameter('min_points').value
        use_cuda = self.get_parameter('use_cuda').value
        publish_rate = self.get_parameter('publish_rate_hz').value

        # --- Device setup ---
        self.device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        # --- Load depth model ---
        self.get_logger().info(f'Loading model from: {model_path}')
        self.get_logger().info(f'Loading config from: {exp_opts}')
        
        if not os.path.exists(exp_opts):
            self.get_logger().error(f'exp_opts.yaml not found: {exp_opts}, recheck path in declare_parameter')
            raise FileNotFoundError(f'exp_opts.yaml not found: {exp_opts}, recheck path in declare_parameter')
        
        if not os.path.exists(model_path):
            self.get_logger().error(f'model.pth not found: {model_path}, recheck path in declare_parameter')
            raise FileNotFoundError(f'model.pth not found: {model_path}, recheck path in declare_parameter')
        
        opts_dic = read_yaml_options(exp_opts)
        self.opts_dic = opts_dic
        self.depth_network = get_model_with_opts(opts_dic, self.device)
        self.depth_network = load_model_for_evaluate(model_path, self.depth_network)

        self.depth_network.eval()
        
        self.get_logger().info(f'Model loaded: {opts_dic["model"]["type"]}')

        # --- Image preprocessing setup ---
        self.to_tensor = tf.ToTensor()
        self.normalize = tf.Normalize(
            mean=opts_dic.get('pred_norm', [0.5, 0.5, 0.5]), 
            std=[1, 1, 1]
        )
        self.input_size = opts_dic.get('pred_size', [480, 640])  # [height, width]
        self.resize = tf.Resize(self.input_size, interpolation=PILImage.Resampling.LANCZOS)
        
        self.get_logger().info(f'Input size: {self.input_size}')

        # --- ROS components ---
        self.bridge = CvBridge()

        # Subscribers for stereo images
        self.left_sub = message_filters.Subscriber(
            self, CompressedImage, '/left/image_rect_color'
        )
        self.right_sub = message_filters.Subscriber(
            self, CompressedImage, '/right/image_rect_color'
        )

        # Synchronize stereo images
        self.stereo_sync = message_filters.ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub], 
            queue_size=10, 
            slop=0.05
        )
        self.stereo_sync.registerCallback(self.stereo_callback)

        # Store latest depth data
        self.latest_pointcloud = None
        self.latest_depth_map = None
        self.latest_pc_header = None
        
        # Subscribe to detections separately
        self.det_subscription = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        # Publishers
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, '/points2', 10
        )
        self.detection3d_pub = self.create_publisher(
            DetectionArray, '/detections_3d', 10
        )
        self.depth_image_pub = self.create_publisher(
            CompressedImage, '/depth/image_color', 10
        )

        # Rate limiting for point cloud publishing
        self.last_pc_publish_time = self.get_clock().now()
        self.pc_publish_period = 1.0 / publish_rate  # seconds

        self.get_logger().info('Depth estimation node initialized successfully')

    def preprocess_image(self, cv_img):
        """
        Preprocess OpenCV image for model input.
        
        Args:
            cv_img: BGR image from cv_bridge
        
        Returns:
            Preprocessed tensor ready for model
        """
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_img = PILImage.fromarray(rgb_img)
        
        # Apply transforms
        img_tensor = self.normalize(self.to_tensor(self.resize(pil_img)))
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        return img_tensor

    def stereo_callback(self, left_msg, right_msg):
        """
        Process stereo images, run depth estimation, and publish point cloud.
        """
        try:
            import time
            t0 = time.time()
            
            # Convert ROS images to OpenCV
            left_img = self.bridge.compressed_imgmsg_to_cv2(left_msg)
            right_img = self.bridge.compressed_imgmsg_to_cv2(right_msg)
            t1 = time.time()
            # Get original dimensions
            orig_h, orig_w = left_img.shape[:2]

            # Preprocess images
            left_tensor = self.preprocess_image(left_img).to(self.device)
            right_tensor = self.preprocess_image(right_img).to(self.device)
            
            t2 = time.time()
            # Run depth inference
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                    inputs = {
                        'color_s': left_tensor,
                        'color_o': right_tensor
                    }
                    outputs = self.depth_network.inference_forward(inputs, is_train=False)
                    self.get_logger().info(f"Available keys in outputs: {list(outputs.keys())}")
                    #depth_tensor = outputs[('depth', 's')]
                    depth_tensor = outputs['stereo_depth_0_s']
                    
            t3 = time.time()        

            # Resize depth to original image size
            depth_tensor = F.interpolate(
                depth_tensor,
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=True
            )

            # Convert to numpy
            depth_map = depth_tensor.squeeze(0).squeeze(0).cpu().numpy()
            depth_map = depth_map * (2.5/5.5)
            self.get_logger().info(f'Depth: min={depth_map.min():.3f}, max={depth_map.max():.3f}, mean={depth_map.mean():.3f}')
            t4 = time.time()
             # --- PUBLISH COLORIZED DEPTH IMAGE ---
            depth_color = colorize_depth(depth_map)
            
            # Convert to ROS Image message
            depth_msg = self.bridge.cv2_to_compressed_imgmsg(depth_color)
            depth_msg.header = left_msg.header
            
            # Publish colorized depth
            self.depth_image_pub.publish(depth_msg)

            # Store for detection callback
            self.latest_depth_map = depth_map
            self.latest_pc_header = left_msg.header

            # Convert depth map to point cloud
            pointcloud_xyz = depth_to_pointcloud(depth_map, fx, fy, cx, cy)
            t5 = time.time()
            self.latest_pointcloud = pointcloud_xyz

            # Publish point cloud (with rate limiting)
            current_time = self.get_clock().now()
            time_since_last = (current_time - self.last_pc_publish_time).nanoseconds / 1e9
            
            if time_since_last >= self.pc_publish_period:
                self.publish_pointcloud(pointcloud_xyz, left_msg.header, left_img)
                t6 = time.time()
                self.last_pc_publish_time = current_time

        except Exception as e:
            self.get_logger().error(f'Error in stereo callback: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def publish_pointcloud(self, points_xyz, header, left_img=None):
        """
        Publish point cloud to /points2 with RGB colors.
        """
        h, w, _ = points_xyz.shape
        points_flat = points_xyz.reshape(-1, 3)
        
        # Filter by depth range
        valid_depth = (points_flat[:, 2] > 0.5) & (points_flat[:, 2] < 30.0)
        valid_finite = np.isfinite(points_flat).all(axis=1)
        valid_mask = valid_depth & valid_finite
        
        points_valid = points_flat[valid_mask]
        
        if left_img is not None and len(points_valid) > 0:
            # Get RGB colors from left image
            rgb_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            colors_flat = rgb_img.reshape(-1, 3)
            colors_valid = colors_flat[valid_mask]
            
            # Create structured array with XYZRGB
            cloud_data = np.zeros(len(points_valid), dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('rgb', np.float32)  # ← IMPORTANT: float32, not uint32
            ])
            
            cloud_data['x'] = points_valid[:, 0].astype(np.float32)
            cloud_data['y'] = points_valid[:, 1].astype(np.float32)
            cloud_data['z'] = points_valid[:, 2].astype(np.float32)
            
            # Pack RGB as float (PCL standard format)
            # This is the KEY fix - pack as BGR order, then reinterpret as float
            rgb_int = (colors_valid[:, 2].astype(np.uint32) << 16 |  # B
                    colors_valid[:, 1].astype(np.uint32) << 8 |   # G
                    colors_valid[:, 0].astype(np.uint32))         # R
            
            cloud_data['rgb'] = rgb_int.view(np.float32)
            
            # Create PointCloud2 message
            from sensor_msgs.msg import PointField
            
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            
            pc_msg = PointCloud2()
            pc_msg.header = header
            pc_msg.height = 1
            pc_msg.width = len(points_valid)
            pc_msg.is_dense = False
            pc_msg.is_bigendian = False
            pc_msg.fields = fields
            pc_msg.point_step = 16
            pc_msg.row_step = pc_msg.point_step * pc_msg.width
            pc_msg.data = cloud_data.tobytes()
        else:
            # No color, just XYZ
            pc_msg = pc2.create_cloud_xyz32(header, points_valid)
        
        self.pointcloud_pub.publish(pc_msg)

    def detection_callback(self, det_msg):
        """
        Process detections and estimate 3D positions using latest depth map.
        """
        if self.latest_pointcloud is None or self.latest_depth_map is None:
            # No depth data available yet
            return

        try:
            # Create output DetectionArray
            det_array = DetectionArray()
            det_array.header = self.latest_pc_header

            for det in det_msg.detections:
                # Extract bounding box pixel bounds
                u_center = det.bbox.center.position.x
                v_center = det.bbox.center.position.y
                half_w = det.bbox.size_x / 2.0
                half_h = det.bbox.size_y / 2.0

                u_min, u_max = int(u_center - half_w), int(u_center + half_w)
                v_min, v_max = int(v_center - half_h), int(v_center + half_h)

                # Clamp to image bounds
                h, w = self.latest_depth_map.shape
                u_min = max(u_min, 0)
                v_min = max(v_min, 0)
                u_max = min(u_max, w - 1)
                v_max = min(v_max, h - 1)

                # Slice the 3D points from stored point cloud
                box_points = self.latest_pointcloud[v_min:v_max+1, u_min:u_max+1].reshape(-1, 3)

                # Remove invalid points
                valid_mask = np.isfinite(box_points).all(axis=1)
                box_points = box_points[valid_mask]

                if len(box_points) < self.min_points:
                    continue

                # Filter and estimate position
                filtered_points = filter_points_mad(box_points, k=self.mad_k)
                if len(filtered_points) < self.min_points:
                    continue

                position = estimate_position_symmetry(
                    filtered_points, (u_center, v_center), P
                )

                # Create Detection message
                detection = Detection()
                
                # Extract class name and confidence
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
                self.detection3d_pub.publish(det_array)

        except Exception as e:
            self.get_logger().error(f'Error in detection callback: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())


def main():
    rclpy.init()
    node = DepthEstimationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()