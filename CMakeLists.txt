cmake_minimum_required(VERSION 3.5)
project(pandar_rain_filter)

set(CMAKE_CXX_STANDARD 11)

find_package(catkin REQUIRED COMPONENTS
        cv_bridge
        pcl_conversions
        pcl_ros
        roscpp
        rosbag
        nav_msgs
        sensor_msgs
        tf_conversions 
    	pandar_msgs
	pandar_pointcloud
        )

find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Qt5Core REQUIRED)

catkin_package(
        DEPENDS OpenCV
)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -Wall ${CMAKE_CXX_FLAGS}")

include_directories(
        src/include/
        ${catkin_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
)   
        
add_executable(rosbag_processor_pandar_rain_filter
        src/rosbag_processor_pandar_rain_filter.cpp)           

target_link_libraries(rosbag_processor_pandar_rain_filter PRIVATE
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
)
############
install(TARGETS
        rosbag_processor_pandar_rain_filter
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
      launch
      DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
