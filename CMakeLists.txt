cmake_minimum_required(VERSION 2.8.3)
project(lidar_object_detection)

find_package(catkin REQUIRED COMPONENTS 
	cv_bridge
	roscpp
	std_msgs
	nav_msgs
	sensor_msgs
	visualization_msgs
	geometry_msgs
	pcl_conversions
	pcl_ros
	eigen_conversions
	tf_conversions
)

find_package(PCL REQUIRED)
find_package(Eigen3)
find_package( OpenCV REQUIRED )

include_directories(include ${catkin_INCLUDE_DIRS})

add_message_files(FILES ClusterArray.msg)

generate_messages(DEPENDENCIES std_msgs sensor_msgs)

catkin_package(DEPENDS  tf_conversions eigen_conversions message_runtime std_msgs sensor_msgs)

add_executable(adaptive_clustering src/adaptive_clustering.cpp)
target_link_libraries(adaptive_clustering ${catkin_LIBRARIES})
if(catkin_EXPORTED_TARGETS)
  add_dependencies(adaptive_clustering ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
endif()

add_executable(save_detector_data src/save_detector_data.cpp)
target_link_libraries(save_detector_data ${catkin_LIBRARIES} ${OpenCV_LIBS})
if(catkin_EXPORTED_TARGETS)
  add_dependencies(save_detector_data ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
endif()

# install

install(TARGETS adaptive_clustering save_detector_data
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  PATTERN ".svn" EXCLUDE)