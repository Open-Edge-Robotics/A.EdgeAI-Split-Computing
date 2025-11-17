# import os
# from ament_index_python.packages import get_package_share_directory
# from launch import LaunchDescription
# from launch_ros.actions import Node
# from launch.actions import IncludeLaunchDescription, RegisterEventHandler
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from launch.event_handlers import OnProcessExit

# def generate_launch_description():
#     pkg_dir = get_package_share_directory('franka_bringup')
    
#     controllers_config = os.path.join(pkg_dir, 'config', 'controllers.yaml')

#     execute_node_path = os.path.join(os.environ['HOME'], 'ros2_ws', 'src', 'franka_bringup', 'scripts', 'execute.py')

#     franka_core_launch = IncludeLaunchDescription(
#         PythonLaunchDescriptionSource(
#             os.path.join(pkg_dir, 'launch', 'franka.launch.py')
#         ),
#         launch_arguments={'load_gripper': 'true'}.items(),
#     )

#     ros2_control_node = Node(
#         package='controller_manager',
#         executable='ros2_control_node',
#         parameters=[controllers_config],
#         output='screen',
#         emulate_tty=True
#     )

#     execute_node = Node(
#         package='franka_bringup',
#         executable='execute.py',
#         name='case_controller_node',
#         output='screen',
#         emulate_tty=True
#     )

#     joint_state_spawner = Node(
#         package='controller_manager',
#         executable='spawner',
#         arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
#         output='screen',
#         emulate_tty=True
#     )

#     move_to_start_spawner = Node(
#         package='controller_manager',
#         executable='spawner',
#         arguments=['move_to_start_example_controller', '--controller-manager', '/controller_manager'],
#         output='screen',
#         emulate_tty=True
#     )

#     return LaunchDescription([
#         franka_core_launch,
#         ros2_control_node,
#         execute_node,

#         RegisterEventHandler(
#             OnProcessExit(
#                 target_action=ros2_control_node,
#                 on_exit=[joint_state_spawner],
#             )
#         ),

#         RegisterEventHandler(
#             OnProcessExit(
#                 target_action=joint_state_spawner,
#                 on_exit=[move_to_start_spawner],
#             )
#         ),
#     ])

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='franka_bringup',
            executable='execute.py',
            arguments=['/ros2_ws/src/franka_bringup/scripts/execute.py'],
            name='case_controller_node',
            namespace='NS_1',
            output='screen'
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('franka_bringup'),
                    'launch',
                    'franka.launch.py'
                ])
            ])
        ),
    ])

