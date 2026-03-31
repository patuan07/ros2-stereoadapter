import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np

class StereoRepublisher(Node):
    def __init__(self):
        super().__init__('republisher')

        # Underwater preprocessing toggles
        self.declare_parameter('preprocess.clahe',        True)
        self.declare_parameter('preprocess.hist_stretch', True)
        self.declare_parameter('preprocess.custom_gray',  True)

        # CLAHE tuning
        self.declare_parameter('preprocess.clahe_clip_limit', 2.0)
        self.declare_parameter('preprocess.clahe_tile_size',  8)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.bridge = CvBridge()

        # Publishers - compressed output topics
        self.pub_left_img  = self.create_publisher(CompressedImage, '/left/image_raw/compressed', qos)
        self.pub_right_img = self.create_publisher(CompressedImage, '/right/image_raw/compressed', qos)

        # Subscribers - rosbag topics (replace with actual topic names if different)
        self.sub_left_img = self.create_subscription(
            CompressedImage,
            '/left/image_raw',
            self.left_image_cb,
            qos
        )
        self.sub_right_img = self.create_subscription(
            CompressedImage,
            '/right/image_raw',
            self.right_image_cb,
            qos
        )

    # ------------------------------------------------------------------
    # Preprocessing helpers (identical to stereo_publisher)
    # ------------------------------------------------------------------

    def hist_stretch(self, img):
        """Stretch each BGR channel to [0, 255] to recover red/green range."""
        out = np.empty_like(img)
        for i in range(3):
            ch = img[:, :, i].astype(np.float32)
            lo, hi = ch.min(), ch.max()
            if hi > lo:
                ch = (ch - lo) / (hi - lo) * 255.0
            out[:, :, i] = np.clip(ch, 0, 255).astype(np.uint8)
        return out

    def apply_clahe(self, img, clip_limit, tile_size):
        """CLAHE on the L channel in LAB space to boost contrast without colour shift."""
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        l = clahe.apply(l)
        return cv.cvtColor(cv.merge((l, a, b)), cv.COLOR_LAB2BGR)

    def preprocess(self, img):
        """Run hist stretch → CLAHE → optional custom grayscale conversion."""
        do_hist_stretch = self.get_parameter('preprocess.hist_stretch').value
        do_clahe        = self.get_parameter('preprocess.clahe').value
        do_custom_gray  = self.get_parameter('preprocess.custom_gray').value
        clip_limit      = self.get_parameter('preprocess.clahe_clip_limit').value
        tile_size       = int(self.get_parameter('preprocess.clahe_tile_size').value)

        if do_hist_stretch:
            img = self.hist_stretch(img)

        if do_clahe:
            img = self.apply_clahe(img, clip_limit, tile_size)

        if do_custom_gray:
            # Down-weight blue (dominant underwater): R=0.5, G=0.4, B=0.1
            b, g, r = cv.split(img.astype(np.float32))
            gray = np.clip(0.5*r + 0.4*g + 0.1*b, 0, 255).astype(np.uint8)
            img = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

        return img

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def process_and_republish(self, msg: CompressedImage, publisher):
        """Decode → preprocess → re-encode → republish, preserving the header."""
        img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img = self.preprocess(img)
        out = self.bridge.cv2_to_compressed_imgmsg(img)
        out.header = msg.header   # preserve original stamp + frame_id
        publisher.publish(out)

    def left_image_cb(self, msg: CompressedImage):
        self.process_and_republish(msg, self.pub_left_img)

    def right_image_cb(self, msg: CompressedImage):
        self.process_and_republish(msg, self.pub_right_img)


def main(args=None):
    rclpy.init(args=args)
    node = StereoRepublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()