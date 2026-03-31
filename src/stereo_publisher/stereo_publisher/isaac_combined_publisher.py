import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
import cv2 as cv
import numpy as np
import yaml
import os

"""
    This node is used to publish the following four topics:
    left/image_rect
    right/image_rect
    left/camera_info
    right/camera_info
    disparity_node would subscribe to these 4 topics, process the images and produce the disparity
    points. disparity_node expects these 4 topics to:
    - Have the same stamp in header (making sure that it is synchronized)
    - Must have this exact topic name (or else it wouldn't subscribe to)
"""

class StereoPublisher(Node):
    def __init__(self):
        super().__init__('stereo_combined_publisher')

        #Set compatible quality of service with disparity node
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publishers
        self.pub_left_img  = self.create_publisher(Image, 'left/image_rect', qos)
        self.pub_right_img = self.create_publisher(Image, 'right/image_rect', qos)
        self.pub_left_info  = self.create_publisher(CameraInfo, 'left/camera_info_rect', qos)
        self.pub_right_info = self.create_publisher(CameraInfo, 'right/camera_info_rect', qos)

        # Load calibration
        pkg_share = get_package_share_directory('stereo_publisher')
        left_yaml_path = os.path.join(pkg_share, 'left.yaml')
        right_yaml_path = os.path.join(pkg_share, 'right.yaml')
        
        self.left_info  = self.load_camera_info(left_yaml_path)
        self.right_info = self.load_camera_info(right_yaml_path)

        # Load calibration data for rectification
        self.left_calib = self.load_yaml(left_yaml_path)
        self.right_calib = self.load_yaml(right_yaml_path)
        
        # Initialize rectification maps
        self.init_rectification()

        # Camera
        self.bridge = CvBridge()
        self.cap = cv.VideoCapture("[Put video stream here]", cv.CAP_FFMPEG)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video stream, recheck video stream path")

        # Publish timer (20 Hz)
        self.timer = self.create_timer(1/30, self.timer_cb)

    def load_yaml(self, path):
        """Load raw YAML data"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"YAML not found: {path}")
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    #Loading camera info from camera intrinsic .yaml files
    def load_camera_info(self, path):
        data = self.load_yaml(path)

        msg = CameraInfo()
        msg.width  = data.get('image_width', 0)
        msg.height = data.get('image_height', 0)
        msg.k = data.get('camera_matrix', {}).get('data', [0.0]*9)
        msg.d = data.get('distortion_coefficients', {}).get('data', [])
        msg.r = data.get('rectification_matrix', {}).get('data', [0.0]*9)
        msg.p = data.get('projection_matrix', {}).get('data', [0.0]*12)
        msg.distortion_model = data.get('distortion_model', '')
        return msg

    def init_rectification(self):
        """Initialize rectification maps from calibration data"""
        # Extract parameters from left camera
        K_left = np.array(self.left_calib['camera_matrix']['data']).reshape(3, 3)
        D_left = np.array(self.left_calib['distortion_coefficients']['data'])
        R_left = np.array(self.left_calib['rectification_matrix']['data']).reshape(3, 3)
        P_left = np.array(self.left_calib['projection_matrix']['data']).reshape(3, 4)
        
        # Extract parameters from right camera
        K_right = np.array(self.right_calib['camera_matrix']['data']).reshape(3, 3)
        D_right = np.array(self.right_calib['distortion_coefficients']['data'])
        R_right = np.array(self.right_calib['rectification_matrix']['data']).reshape(3, 3)
        P_right = np.array(self.right_calib['projection_matrix']['data']).reshape(3, 4)
        
        # Image size
        img_size = (self.left_calib['image_width'], self.left_calib['image_height'])
        
        # Compute rectification maps
        self.map_left_x, self.map_left_y = cv.initUndistortRectifyMap(
            K_left, D_left, R_left, P_left, img_size, cv.CV_32FC1
        )
        
        self.map_right_x, self.map_right_y = cv.initUndistortRectifyMap(
            K_right, D_right, R_right, P_right, img_size, cv.CV_32FC1
        )
        
        self.get_logger().info('Rectification maps initialized')

    def timer_cb(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('No frame')
            return

        #Our camera doesn't give two separate feeds, it produces one feed with
        #two images next to each other. We have to split it apart:
        resized_frame = cv.resize(frame, (1280, 480), interpolation=cv.INTER_CUBIC)
        h, w, _ = resized_frame.shape
        mid = w // 2
        left_img  = resized_frame[:, :mid, :]
        right_img = resized_frame[:, mid:, :]

        # RECTIFY the images
        left_rect = cv.remap(left_img, self.map_left_x, self.map_left_y, cv.INTER_LINEAR)
        right_rect = cv.remap(right_img, self.map_right_x, self.map_right_y, cv.INTER_LINEAR)

        now = self.get_clock().now().to_msg() #set start time

        #Bridging cv2 frame to image message (using rectified images)
        msg_left  = self.bridge.cv2_to_imgmsg(left_rect,  encoding='bgr8')
        msg_right = self.bridge.cv2_to_imgmsg(right_rect, encoding='bgr8')

        #Set header settings for left and right camera image feed (for synchronization) 
        msg_left.header.stamp  = now
        msg_right.header.stamp = now
        msg_left.header.frame_id  = "left_camera"
        msg_right.header.frame_id = "right_camera"

        #Set header settings for left and right camera info feed (for synchronization)
        self.left_info.header.stamp  = now
        self.right_info.header.stamp = now
        self.left_info.header.frame_id  = "left_camera"
        self.right_info.header.frame_id = "right_camera"

        # Publish all 4 synchronized topics
        self.pub_left_img.publish(msg_left)
        self.pub_right_img.publish(msg_right)
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