import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
import cv2 as cv
import numpy as np
import yaml
import os

"""
    This node publishes both raw and rectified stereo images:
    /left/image_raw
    /right/image_raw
    /left/image_rect_color 
    /right/image_rect_color    
    /left/camera_info
    /right/camera_info
"""

class StereoPublisher(Node):
    def __init__(self):
        super().__init__('stereo_combined_publisher')

        # Set compatible quality of service with disparity node
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publishers - RAW images
        self.pub_left_img  = self.create_publisher(CompressedImage, '/left/image_raw', qos)
        self.pub_right_img = self.create_publisher(CompressedImage, '/right/image_raw', qos)
        
        self.pub_left_rect  = self.create_publisher(CompressedImage, '/left/image_rect_color', qos)
        self.pub_right_rect = self.create_publisher(CompressedImage, '/right/image_rect_color', qos)
        
        # Publishers - Camera info
        self.pub_left_info  = self.create_publisher(CameraInfo, '/left/camera_info', qos)
        self.pub_right_info = self.create_publisher(CameraInfo, '/right/camera_info', qos)

        # Load calibration
        pkg_share = get_package_share_directory('stereo_publisher')
        left_yaml_path = os.path.join(pkg_share, 'left.yaml')
        right_yaml_path = os.path.join(pkg_share, 'right.yaml')
        
        self.left_info  = self.load_camera_info(left_yaml_path)
        self.right_info = self.load_camera_info(right_yaml_path)
        
        # Load calibration data for rectification
        self.left_calib = self.load_calibration_data(left_yaml_path)
        self.right_calib = self.load_calibration_data(right_yaml_path)
        
        # Create rectification maps
        self.create_rectification_maps()

        # Camera
        self.bridge = CvBridge()
        self.cap = cv.VideoCapture("[insert video feed path]", cv.CAP_FFMPEG)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video stream")

        # Publish timer (30 Hz)
        self.timer = self.create_timer(1/30, self.timer_cb)
        
        self.get_logger().info('Stereo publisher initialized with rectification')

    def load_camera_info(self, path):
        """Load camera info for ROS messages."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"YAML not found: {path}")
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        msg = CameraInfo()
        msg.width  = data.get('image_width', 0)
        msg.height = data.get('image_height', 0)
        msg.k = data.get('camera_matrix', {}).get('data', [0.0]*9)
        msg.d = data.get('distortion_coefficients', {}).get('data', [])
        msg.r = data.get('rectification_matrix', {}).get('data', [0.0]*9)
        msg.p = data.get('projection_matrix', {}).get('data', [0.0]*12)
        msg.distortion_model = data.get('distortion_model', '')
        return msg
    
    def load_calibration_data(self, path):
        """Load calibration data as numpy arrays for cv2."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        calib = {}
        calib['K'] = np.array(data.get('camera_matrix', {}).get('data', [])).reshape(3, 3)
        calib['D'] = np.array(data.get('distortion_coefficients', {}).get('data', []))
        calib['R'] = np.array(data.get('rectification_matrix', {}).get('data', [])).reshape(3, 3)
        calib['P'] = np.array(data.get('projection_matrix', {}).get('data', [])).reshape(3, 4)
        calib['width'] = data.get('image_width', 640)
        calib['height'] = data.get('image_height', 480)
        
        return calib
    
    def create_rectification_maps(self):
        """Create rectification maps for both cameras."""
        # Left camera rectification maps
        self.map1_left, self.map2_left = cv.initUndistortRectifyMap(
            self.left_calib['K'],    # Camera matrix
            self.left_calib['D'],    # Distortion coefficients
            self.left_calib['R'],    # Rectification transform
            self.left_calib['P'],    # Projection matrix
            (self.left_calib['width'], self.left_calib['height']),
            cv.CV_16SC2
        )
        
        # Right camera rectification maps
        self.map1_right, self.map2_right = cv.initUndistortRectifyMap(
            self.right_calib['K'],
            self.right_calib['D'],
            self.right_calib['R'],
            self.right_calib['P'],
            (self.right_calib['width'], self.right_calib['height']),
            cv.CV_16SC2
        )
        
        self.get_logger().info('Rectification maps created')

    def rectify_image(self, img, map1, map2):
        """Rectify a single image using precomputed maps."""
        return cv.remap(img, map1, map2, cv.INTER_LINEAR)

    def timer_cb(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('No frame')
            return

        # Split stereo frame
        resized_frame = cv.resize(frame, (1280, 480), interpolation=cv.INTER_CUBIC)
        h, w, _ = resized_frame.shape
        mid = w // 2
        left_img  = resized_frame[:, :mid, :]
        right_img = resized_frame[:, mid:, :]

        # Rectify images
        left_rect = self.rectify_image(left_img, self.map1_left, self.map2_left)
        right_rect = self.rectify_image(right_img, self.map1_right, self.map2_right)

        now = self.get_clock().now().to_msg()

        # Create RAW image messages
        msg_left_raw  = self.bridge.cv2_to_compressed_imgmsg(left_img)
        msg_right_raw = self.bridge.cv2_to_compressed_imgmsg(right_img)
        
        # Create RECTIFIED image messages
        msg_left_rect  = self.bridge.cv2_to_compressed_imgmsg(left_rect)
        msg_right_rect = self.bridge.cv2_to_compressed_imgmsg(right_rect)

        # Set headers for RAW images
        msg_left_raw.header.stamp  = now
        msg_right_raw.header.stamp = now
        msg_left_raw.header.frame_id  = "left_camera"
        msg_right_raw.header.frame_id = "right_camera"
        
        # Set headers for RECTIFIED images
        msg_left_rect.header.stamp  = now
        msg_right_rect.header.stamp = now
        msg_left_rect.header.frame_id  = "left_camera"
        msg_right_rect.header.frame_id = "right_camera"

        # Set headers for camera info
        self.left_info.header.stamp  = now
        self.right_info.header.stamp = now
        self.left_info.header.frame_id  = "left_camera"
        self.right_info.header.frame_id = "right_camera"

        # Publish all topics
        self.pub_left_img.publish(msg_left_raw)
        self.pub_right_img.publish(msg_right_raw)
        self.pub_left_rect.publish(msg_left_rect)
        self.pub_right_rect.publish(msg_right_rect)
        self.pub_left_info.publish(self.left_info)
        self.pub_right_info.publish(self.right_info)


def main():
    rclpy.init()
    node = StereoPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()