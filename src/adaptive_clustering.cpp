// Copyright (C) 2018  Zhi Yan and Li Sun

// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.

// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
// more details.

// You should have received a copy of the GNU General Public License along
// with this program.  If not, see <http://www.gnu.org/licenses/>.

// ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>
#include "lidar_object_detection/ClusterArray.h"
// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>

#include <visualization.h>

ros::Publisher cluster_array_pub_;
ros::Publisher cloud_filtered_pub_;
ros::Publisher pose_array_pub_;
ros::Publisher marker_array_pub_;

std::string sensor_model_;
std::string frame_id_;
bool print_fps_;
float z_axis_min_;
float z_axis_max_;
int cluster_size_min_;
int cluster_size_max_;
bool object_size_limit_;

const int region_max_ = 10;
int regions_[100];
int cluster_array_seq = 0;
int pose_array_seq = 0;

int frames; clock_t start_time; bool reset = true;//fps
void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& ros_pc2_in) {
  if(print_fps_)if(reset){frames=0;start_time=clock();reset=false;}//fps
  
  /*** Convert ROS message to PCL ***/
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_in(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*ros_pc2_in, *pcl_pc_in);
  
  /*** Remove ground and ceiling ***/
  pcl::IndicesPtr pc_indices(new std::vector<int>);
  pcl::PassThrough<pcl::PointXYZI> pt;
  pt.setInputCloud(pcl_pc_in);
  pt.setFilterFieldName("z");
  pt.setFilterLimits(z_axis_min_, z_axis_max_);
  pt.filter(*pc_indices);
  
  /*** Divide the point cloud into nested circular regions ***/
  boost::array<std::vector<int>, region_max_> indices_array;
  for(int i = 0; i < pc_indices->size(); i++) {
    float range = 0.0;
    for(int j = 0; j < region_max_; j++) {
      float d2 = pcl_pc_in->points[(*pc_indices)[i]].x * pcl_pc_in->points[(*pc_indices)[i]].x +
	pcl_pc_in->points[(*pc_indices)[i]].y * pcl_pc_in->points[(*pc_indices)[i]].y +
	pcl_pc_in->points[(*pc_indices)[i]].z * pcl_pc_in->points[(*pc_indices)[i]].z;
      if(d2 > range * range && d2 <= (range+regions_[j]) * (range+regions_[j])) {
      	indices_array[j].push_back((*pc_indices)[i]);
      	break;
      }
      range += regions_[j];
    }
  }
  
  /*** Euclidean clustering ***/
  float tolerance = 0.0;
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<pcl::PointXYZI>::Ptr > > clusters;
  
  for(int i = 0; i < region_max_; i++) {
    tolerance += 0.1;
    if(indices_array[i].size() > cluster_size_min_) {
      boost::shared_ptr<std::vector<int> > indices_array_ptr(new std::vector<int>(indices_array[i]));
      pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
      tree->setInputCloud(pcl_pc_in, indices_array_ptr);
      
      std::vector<pcl::PointIndices> cluster_indices;
      pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
      ec.setClusterTolerance(tolerance);
      ec.setMinClusterSize(cluster_size_min_);
      ec.setMaxClusterSize(cluster_size_max_);
      ec.setSearchMethod(tree);
      ec.setInputCloud(pcl_pc_in);
      ec.setIndices(indices_array_ptr);
      ec.extract(cluster_indices);
      
      for(std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++) {
      	pcl::PointCloud<pcl::PointXYZI>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZI>);
      	for(std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
      	  cluster->points.push_back(pcl_pc_in->points[*pit]);
  	}
      	cluster->width = cluster->size();
      	cluster->height = 1;
      	cluster->is_dense = true;

	/*** DEBUG ***/
	if(object_size_limit_) {
	  Eigen::Vector4f min, max;
	  pcl::getMinMax3D(*cluster, min, max);
	  if((max[0]-min[0] < 0.2 || max[0]-min[0] > 1.0 ||
	      max[1]-min[1] < 0.2 || max[1]-min[1] > 1.0 ||
	      max[2]-min[2] < 0.5 || max[2]-min[2] > 2.0)) { 
	    continue;
	  }
	}
	/*** DEBUG ***/

	clusters.push_back(cluster);
      }
    }
  }
  
  /*** Output ***/
  if(cloud_filtered_pub_.getNumSubscribers() > 0) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_out(new pcl::PointCloud<pcl::PointXYZI>);
    sensor_msgs::PointCloud2 ros_pc2_out;
    pcl::copyPointCloud(*pcl_pc_in, *pc_indices, *pcl_pc_out);
    pcl::toROSMsg(*pcl_pc_out, ros_pc2_out);
    cloud_filtered_pub_.publish(ros_pc2_out);
  }
  
  lidar_object_detection::ClusterArray cluster_array;
  geometry_msgs::PoseArray pose_array;
  visualization_msgs::MarkerArray marker_array;
  
  for(int i = 0; i < clusters.size(); i++) {
    if(cluster_array_pub_.getNumSubscribers() > 0) {
      sensor_msgs::PointCloud2 ros_pc2_out;
      pcl::toROSMsg(*clusters[i], ros_pc2_out);
      cluster_array.clusters.push_back(ros_pc2_out);
    }

    if(pose_array_pub_.getNumSubscribers() > 0) {
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*clusters[i], centroid);
      
      geometry_msgs::Pose pose;
      pose.position.x = centroid[0];
      pose.position.y = centroid[1];
      pose.position.z = centroid[2];
      pose.orientation.w = 1;
      pose_array.poses.push_back(pose);
    }
    
    if(marker_array_pub_.getNumSubscribers() > 0) {
      Eigen::Vector4f min, max;
      pcl::getMinMax3D(*clusters[i], min, max);

      visualization_msgs::Marker marker;
      marker.header.stamp = ros::Time::now();
      marker.header.frame_id = frame_id_;
      marker.ns = "adaptive_clustering";
      marker.id = i;
      marker.type = visualization_msgs::Marker::LINE_LIST;

      draw_3d_boundingbox(marker, 0, min, max);
      
      marker_array.markers.push_back(marker);
    }
  }
  
  if(cluster_array.clusters.size()) {
    cluster_array.header.seq = ++cluster_array_seq;
    cluster_array.header.stamp = ros::Time::now();
    cluster_array.header.frame_id = frame_id_;
    cluster_array_pub_.publish(cluster_array);
  }

  if(pose_array.poses.size()) {
    pose_array.header.seq = ++pose_array_seq;
    pose_array.header.stamp = ros::Time::now();
    pose_array.header.frame_id = frame_id_;
    pose_array_pub_.publish(pose_array);
    //std::cerr << "[adaptive_clustering] send " << pose_array.poses.size() << " poses" << std::endl;
  }
  
  if(marker_array.markers.size()) {
    marker_array_pub_.publish(marker_array);
  }
  
  if(print_fps_)if(++frames>10){std::cerr<<"[adaptive_clustering] fps = "<<float(frames)/(float(clock()-start_time)/CLOCKS_PER_SEC)<<", timestamp = "<<clock()/CLOCKS_PER_SEC<<std::endl;reset = true;}//fps
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "adaptive_clustering");
  
  /*** Subscribers ***/
  ros::NodeHandle nh;
  ros::Subscriber point_cloud_sub = nh.subscribe<sensor_msgs::PointCloud2>("velodyne_points", 1, pointCloudCallback);

  /*** Publishers ***/
  ros::NodeHandle private_nh("~");
  cluster_array_pub_ = private_nh.advertise<lidar_object_detection::ClusterArray>("clusters", 1);
  cloud_filtered_pub_ = private_nh.advertise<sensor_msgs::PointCloud2>("cloud_filtered", 1);
  pose_array_pub_ = private_nh.advertise<geometry_msgs::PoseArray>("poses", 100);
  marker_array_pub_ = private_nh.advertise<visualization_msgs::MarkerArray>("markers", 1);
  
  /*** Parameters ***/
  private_nh.param<std::string>("sensor_model", sensor_model_, "HDL-32E"); // VLP-16, HDL-32E, HDL-64E
  private_nh.param<std::string>("frame_id", frame_id_, "velodyne");
  private_nh.param<bool>("print_fps", print_fps_, false);
  private_nh.param<float>("z_axis_min", z_axis_min_, -0.5);
  private_nh.param<float>("z_axis_max", z_axis_max_, 5.0);
  private_nh.param<int>("cluster_size_min", cluster_size_min_, 5);
  private_nh.param<int>("cluster_size_max", cluster_size_max_, 700000);
  private_nh.param<bool>("object_size_limit", object_size_limit_, false);
  
  // Divide the point cloud into nested circular regions centred at the sensor.
  // For more details, see our IROS-17 paper "Online learning for human classification in 3D LiDAR-based tracking"
  if(sensor_model_ == "VLP-16") {
    regions_[0] = 2; regions_[1] = 3; regions_[2] = 3; regions_[3] = 3; regions_[4] = 3;
    regions_[5] = 3; regions_[6] = 3; regions_[7] = 2; regions_[8] = 3; regions_[9] = 3;
    regions_[10]= 3; regions_[11]= 3; regions_[12]= 3; regions_[13]= 3;
  } else if (sensor_model_ == "HDL-32E") {
    regions_[0] = 4; regions_[1] = 5; regions_[2] = 4; regions_[3] = 5; regions_[4] = 4;
    regions_[5] = 5; regions_[6] = 5; regions_[7] = 4; regions_[8] = 5; regions_[9] = 4;
    regions_[10]= 5; regions_[11]= 5; regions_[12]= 4; regions_[13]= 5;
  } else if (sensor_model_ == "HDL-64E") {
    regions_[0] = 14; regions_[1] = 14; regions_[2] = 14; regions_[3] = 15; regions_[4] = 14;
  }
  
  ros::spin();
  return 0;
}
