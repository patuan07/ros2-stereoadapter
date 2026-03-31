from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # --- Launch arguments ---
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='[insert model for inference]',
        description='Path to your YOLOv8 .engine model file'
    )
    confidence_arg = DeclareLaunchArgument(
        'confidence',
        default_value='0.5',
        description='YOLO confidence threshold'
    )
    mad_k_arg = DeclareLaunchArgument(
        'mad_k',
        default_value='2.0',
        description='MAD threshold scaling factor (lower = stricter filtering)'
    )
    min_points_arg = DeclareLaunchArgument(
        'min_points',
        default_value='10',
        description='Minimum valid points required to estimate position'
    )

    return LaunchDescription([
        model_path_arg,
        confidence_arg,
        mad_k_arg,
        min_points_arg,

        # --- Detection node ---
        Node(
            package='stereo_publisher',
            executable='detection_node',
            name='detection_node',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'confidence_threshold': LaunchConfiguration('confidence'),
            }],
        ),

        # --- Depth estimation node ---
        # Node(
        #     package='stereo_publisher',
        #     executable='depth_estimation_node',
        #     name='depth_estimation_node',
        #     parameters=[{
        #         'mad_k': LaunchConfiguration('mad_k'),
        #         'min_points': LaunchConfiguration('min_points'),
        #     }],
        # ),

        Node(
            package='stereo_publisher',
            executable='combined_publisher',
            name='stereo_combined_publisher',
        ),

        Node(
            package='stereo_publisher',
            executable='new_depth_estimation_node',
            name='new_depth_estimation_node',
        ),
    ])