cmake_minimum_required(VERSION 3.5)
project(safety_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObstacleStateList.msg"
  DEPENDENCIES std_msgs
  SKIP_GROUP_MEMBERSHIP_CHECK
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
