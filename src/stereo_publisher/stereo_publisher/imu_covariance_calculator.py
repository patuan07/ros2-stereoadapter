#!/usr/bin/env python3
"""
IMU Covariance Calculator Node

This node subscribes to IMU data and calculates covariance matrices
for orientation (quaternion), angular velocity, and linear acceleration.

Usage:
1. Run this node: ros2 run <package> imu_covariance_calculator.py
2. Play your rosbag: ros2 bag play <your_bag_file>
3. Wait for data collection to complete
4. The node will output covariance matrices that you can use in your sensor_can_node.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np
from scipy.spatial.transform import Rotation
import math


class ImuCovarianceCalculator(Node):
    def __init__(self):
        super().__init__('imu_covariance_calculator')
        
        # Parameters
        self.declare_parameter('imu_topic', 'imu/data')
        self.declare_parameter('min_samples', 100)
        self.declare_parameter('max_samples', 100000)
        self.declare_parameter('calculate_orientation_from_quaternion', True)
        
        imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value
        self.min_samples = self.get_parameter('min_samples').get_parameter_value().integer_value
        self.max_samples = self.get_parameter('max_samples').get_parameter_value().integer_value
        self.calc_from_quat = self.get_parameter('calculate_orientation_from_quaternion').get_parameter_value().bool_value
        
        # Data storage
        self.orientation_data = []  # Store as roll, pitch, yaw for better covariance calculation
        self.angular_velocity_data = []
        self.linear_acceleration_data = []
        
        # Quaternion storage (for direct quaternion covariance if needed)
        self.quaternion_data = []
        
        # Subscription
        self.subscription = self.create_subscription(
            Imu,
            imu_topic,
            self.imu_callback,
            10
        )
        
        # Timer to check if data collection has stopped (bag finished)
        self.last_message_time = self.get_clock().now()
        self.check_timer = self.create_timer(2.0, self.check_for_completion)
        self.data_stopped = False
        
        self.get_logger().info(f'IMU Covariance Calculator started')
        self.get_logger().info(f'Subscribing to: {imu_topic}')
        self.get_logger().info(f'Will collect {self.min_samples} to {self.max_samples} samples')
        self.get_logger().info('Waiting for stationary IMU data...')
        
        self.sample_count = 0
        self.calculation_done = False

    def quaternion_to_euler(self, qx, qy, qz, qw):
        """Convert quaternion to Euler angles (roll, pitch, yaw) in radians"""
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

    def check_for_completion(self):
        """Check if bag playback has finished (no new messages for 2 seconds)"""
        if self.calculation_done or self.data_stopped:
            return
        
        if self.sample_count == 0:
            return  # Haven't started receiving data yet
        
        time_since_last_msg = (self.get_clock().now() - self.last_message_time).nanoseconds / 1e9
        
        if time_since_last_msg > 2.0:  # No data for 2 seconds
            if self.sample_count >= self.min_samples:
                self.get_logger().info(f'Bag playback appears finished. Collected {self.sample_count} samples.')
                self.data_stopped = True
                self.calculate_and_display_covariances()
            elif not self.data_stopped:
                self.get_logger().warn(f'Only collected {self.sample_count} samples (minimum: {self.min_samples}). Waiting for more data...')

    def imu_callback(self, msg: Imu):
        if self.calculation_done:
            return
        
        # Update last message time
        self.last_message_time = self.get_clock().now()
            
        # Stop collecting if we've reached max samples
        if self.sample_count >= self.max_samples:
            if not self.calculation_done:
                self.get_logger().info(f'Maximum samples ({self.max_samples}) reached. Calculating covariances...')
                self.calculate_and_display_covariances()
            return
        
        # Extract quaternion orientation and convert to Euler angles
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w
        
        # Store quaternion data
        self.quaternion_data.append([qx, qy, qz, qw])
        
        # Convert to Euler angles for orientation covariance
        roll, pitch, yaw = self.quaternion_to_euler(qx, qy, qz, qw)
        self.orientation_data.append([roll, pitch, yaw])
        
        # Store angular velocity
        gx = msg.angular_velocity.x
        gy = msg.angular_velocity.y
        gz = msg.angular_velocity.z
        self.angular_velocity_data.append([gx, gy, gz])
        
        # Store linear acceleration
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        self.linear_acceleration_data.append([ax, ay, az])
        
        self.sample_count += 1
        
        if self.sample_count % 50 == 0:
            self.get_logger().info(f'Collected {self.sample_count} samples...')
        
        # Don't calculate until we've collected all available data or hit max_samples
        # Calculation will happen when bag ends (node shutdown) or max_samples reached

    def calculate_and_display_covariances(self):
        if self.calculation_done:
            return
        
        self.calculation_done = True
        
        self.get_logger().info('\n' + '='*70)
        self.get_logger().info(f'COVARIANCE CALCULATION RESULTS ({self.sample_count} samples)')
        self.get_logger().info('='*70 + '\n')
        
        # Convert lists to numpy arrays
        orientation_array = np.array(self.orientation_data)
        angular_velocity_array = np.array(self.angular_velocity_data)
        linear_acceleration_array = np.array(self.linear_acceleration_data)
        quaternion_array = np.array(self.quaternion_data)
        
        # Calculate statistics
        self.get_logger().info('ORIENTATION (Roll, Pitch, Yaw in radians):')
        self.display_statistics(orientation_array, ['Roll', 'Pitch', 'Yaw'])
        orientation_cov = np.cov(orientation_array.T)
        self.display_covariance_matrix(orientation_cov, 'Orientation')
        
        self.get_logger().info('\nANGULAR VELOCITY (rad/s):')
        self.display_statistics(angular_velocity_array, ['gx', 'gy', 'gz'])
        angular_velocity_cov = np.cov(angular_velocity_array.T)
        self.display_covariance_matrix(angular_velocity_cov, 'Angular Velocity')
        
        self.get_logger().info('\nLINEAR ACCELERATION (m/s²):')
        self.display_statistics(linear_acceleration_array, ['ax', 'ay', 'az'])
        linear_acceleration_cov = np.cov(linear_acceleration_array.T)
        self.display_covariance_matrix(linear_acceleration_cov, 'Linear Acceleration')
        
        # Also calculate quaternion covariance for reference
        self.get_logger().info('\nQUATERNION (qx, qy, qz, qw):')
        self.display_statistics(quaternion_array, ['qx', 'qy', 'qz', 'qw'])
        
        # Display Python code to copy-paste
        self.get_logger().info('\n' + '='*70)
        self.get_logger().info('COPY-PASTE CODE FOR sensor_can_node.py:')
        self.get_logger().info('='*70)
        self.get_logger().info('\nReplace the covariance arrays in your _publish_rpy_if_possible() method with:')
        self.get_logger().info('')
        
        # Format orientation covariance for code
        self.get_logger().info('# Orientation covariance (roll, pitch, yaw):')
        self.get_logger().info('imu_msg.orientation_covariance = [')
        for i in range(3):
            row_str = '    ' + ', '.join([f'{orientation_cov[i,j]:.6e}' for j in range(3)])
            if i < 2:
                row_str += ','
            self.get_logger().info(row_str)
        self.get_logger().info(']')
        
        # Format angular velocity covariance for code
        self.get_logger().info('\n# Angular velocity covariance:')
        self.get_logger().info('imu_msg.angular_velocity_covariance = [')
        for i in range(3):
            row_str = '    ' + ', '.join([f'{angular_velocity_cov[i,j]:.6e}' for j in range(3)])
            if i < 2:
                row_str += ','
            self.get_logger().info(row_str)
        self.get_logger().info(']')
        
        # Format linear acceleration covariance for code
        self.get_logger().info('\n# Linear acceleration covariance:')
        self.get_logger().info('imu_msg.linear_acceleration_covariance = [')
        for i in range(3):
            row_str = '    ' + ', '.join([f'{linear_acceleration_cov[i,j]:.6e}' for j in range(3)])
            if i < 2:
                row_str += ','
            self.get_logger().info(row_str)
        self.get_logger().info(']')
        
        self.get_logger().info('\n' + '='*70)
        self.get_logger().info('Calculation complete. You can now stop the node (Ctrl+C)')
        self.get_logger().info('='*70 + '\n')

    def display_statistics(self, data, labels):
        """Display mean and standard deviation for each axis"""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        for i, label in enumerate(labels):
            self.get_logger().info(f'  {label}: mean = {mean[i]:.6f}, std = {std[i]:.6f}')

    def display_covariance_matrix(self, cov_matrix, name):
        """Display covariance matrix in a readable format"""
        self.get_logger().info(f'\n{name} Covariance Matrix:')
        for i in range(cov_matrix.shape[0]):
            row_str = '  [' + '  '.join([f'{cov_matrix[i,j]:12.6e}' for j in range(cov_matrix.shape[1])]) + ']'
            self.get_logger().info(row_str)


def main(args=None):
    rclpy.init(args=args)
    node = ImuCovarianceCalculator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down...')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
