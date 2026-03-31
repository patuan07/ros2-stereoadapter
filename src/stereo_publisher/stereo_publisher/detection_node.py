import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import Detection2D, Detection2DArray, BoundingBox2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import cv2
#from stereo_interfaces.msg import DetectionWithImage, Detection


class DetectionNode(Node):
    """
    Runs YOLOv8 inference on the rectified left image and publishes
    bounding boxes + class labels to /detections.
    """

    def __init__(self):
        super().__init__('detection_node')

        # --- Parameters ---
        # Update model_path to point to your trained .pt file
        self.declare_parameter('model_path', '[insert model path for inference]')
        # Confidence threshold — detections below this are discarded
        self.declare_parameter('confidence_threshold', 0.5)

        model_path = self.get_parameter('model_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value

        # --- Model ---
        self.model = YOLO(model_path)
        self.class_names = self.model.names  # {0: 'class_a', 1: 'class_b', ...}
        self.get_logger().info(f'Loaded model: {model_path}')
        self.get_logger().info(f'Classes: {self.class_names}')

        # --- ROS ---
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            CompressedImage, '/left/image_rect_color', self.image_cb, 10
        )
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/detections', 10
        )

    def image_cb(self, img_msg):
        # Convert ROS image to OpenCV
        try:
            cv_img = self.bridge.compressed_imgmsg_to_cv2(img_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # Run inference
        results = self.model(cv_img, verbose=False)[0]

        # Build Detection2DArray message
        det_array = Detection2DArray()
        det_array.header = img_msg.header  # preserve timestamp for sync

        # Create annotated image for visualization
        annotated_img = cv_img.copy()

        for box in results.boxes:
            if box.conf[0].item() < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            confidence = box.conf[0].item()

            # Draw bounding box
            cv2.rectangle(annotated_img, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         (0, 255, 0), 2)
            
            # Draw label with class name and confidence
            label = f'{self.class_names[class_id]}: {confidence:.2f}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_img,
                         (int(x1), int(y1) - label_size[1] - 5),
                         (int(x1) + label_size[0], int(y1)),
                         (0, 255, 0), -1)
            cv2.putText(annotated_img, label,
                       (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            det = Detection2D()
            det.bbox.center.position.x = (x1 + x2) / 2.0
            det.bbox.center.position.y = (y1 + y2) / 2.0
            det.bbox.size_x = x2 - x1
            det.bbox.size_y = y2 - y1
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.score = confidence
            hypothesis.hypothesis.class_id = str(class_id)
            det.results.append(hypothesis)
            det_array.detections.append(det)

        # Display the annotated image
        cv2.imshow('Detections', annotated_img)
        cv2.waitKey(1)  # 1ms wait to allow window to update

        self.detection_pub.publish(det_array)


def main():
    rclpy.init()
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()