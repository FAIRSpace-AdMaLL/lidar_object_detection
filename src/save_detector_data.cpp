#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/Odometry.h>
#include "lidar_object_detection/ClusterArray.h"

// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>


#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>

#include <iostream>
#include <fstream>
#include <cmath>

// TF
#include <visualization.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <pcl_ros/transforms.h> 

//Eigen
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace sensor_msgs;
using namespace std;
using namespace geometry_msgs;
using namespace message_filters;


double dist2d(double x1, double y1, double x2, double y2)
{
  return sqrt(pow(x1-x2,2) + pow(y1-y2,2));
}

double dist_detection_gt(const Eigen::Vector4f& cluster, const std::vector<geometry_msgs::Pose> gts)
{
  double closest = 100;
  for(vector<geometry_msgs::Pose>::const_iterator it = gts.begin(); it != gts.end(); it++)
  {
    double distance = dist2d(cluster[0], cluster[1], it->position.x, it->position.y);
    if (distance < closest)
      closest = distance;
  }
  return closest;
}

void nomalizeCluster(pcl::PointCloud<pcl::PointXYZI>& pc, Eigen::Vector4f& centroid)
{
  for(int i = 0; i < pc.size(); i++)
  {
    pc.points[i].x -= centroid[0];
    pc.points[i].y -= centroid[1];
    pc.points[i].z -= centroid[2];
    pc.points[i].intensity /= 255.;
  }
}


class Detector_Saver
{
public:
  Detector_Saver()
  {
    
    map_file = "/home/kevin/data/ncfm/map/map.txt";
    string map_img_file = "/home/kevin/data/ncfm/map/label.jpg";

    map_res = 0.1;
    map_origin = Eigen::Vector2d(0.0, -9.9);

    ncfm_map = cv::imread(map_img_file, cv::IMREAD_UNCHANGED);

    map_array = Detector_Saver::readMap(map_file);
    example_pub = nh_.advertise<visualization_msgs::MarkerArray>("example_makers", 1000);
  }

  Eigen::Vector2d map2world(Eigen::Vector2d pos);
  Eigen::Vector2d world2map(Eigen::Vector2d pos);
  void detectorCallback(const lidar_object_detection::ClusterArrayConstPtr& detect);
  vector<int> readMap(string map_file);
  int getLabel(int i, int j);

public:
  string map_file;
  string detector_topic;
  string sensor_odom_topic;
  string frame_id;

  cv::Mat ncfm_map;
  float map_res;
  Eigen::Vector2d map_origin;
  visualization_msgs::MarkerArray marker_array;
  vector<int> map_array;

  ros::Publisher example_pub;
  ros::NodeHandle nh_;
};

int Detector_Saver::getLabel(int i, int j)
{
  return map_array[i*ncfm_map.cols+j];
}

vector<int> Detector_Saver::readMap(string map_file)
{
  ifstream File;
  File.open(map_file.c_str());

  vector<int>numbers;
  int number;

  while(File >> number)
  {
    numbers.push_back(number);
    //cout << number << endl; 
  }

  return numbers;
}

Eigen::Vector2d Detector_Saver::map2world(Eigen::Vector2d pos)
{
  Eigen::Vector2d new_pos((pos(0)-ncfm_map.cols)*map_res, (pos(1)-ncfm_map.rows)*map_res);
  new_pos += map_origin;
  
  return new_pos;
}

Eigen::Vector2d Detector_Saver::world2map(Eigen::Vector2d pos)
{
  Eigen::Vector2d new_pos;

  new_pos = pos - map_origin;

  new_pos(0) = new_pos(0)/map_res;
  new_pos(1) = ncfm_map.rows - new_pos(1)/map_res;
  
  return new_pos;
}

std::ofstream myfile("../info.txt");

void Detector_Saver::detectorCallback(const lidar_object_detection::ClusterArrayConstPtr& detect)
{

  tf::TransformListener listener;
  tf::StampedTransform transform;

  try{
    listener.waitForTransform("world", frame_id, ros::Time(0.), ros::Duration(2.0));
    listener.lookupTransform("world", frame_id, ros::Time(0.), transform);
  }
    catch (tf::TransformException ex){
    ROS_ERROR("%s",ex.what());
    ros::Duration(1.0).sleep();
  }

  // Solve all of perception here...
  cout << "frame id of clusters " << detect->header.frame_id << endl;
  //cout << "frame id of fuser odom " << fuser_odom->header.frame_id << endl;

  Eigen::Matrix4f T(4,4); 
  pcl_ros::transformAsMatrix (transform, T); 
  // std::cout << "transform from sensor to world: " << T << std::endl;
  
  // std::cout << t << endl;

  // std::vector<geometry_msgs::Pose> vehi_poses = vehi->poses;
  // std::vector<geometry_msgs::Pose> ped_poses = ped->poses;

  vector<sensor_msgs::PointCloud2> clusters = detect->clusters;
  marker_array.markers.resize(0);
  double time = detect->header.seq;

  ostringstream strs;
  strs << time;
  string time_s = strs.str();
  int i = 0;
  bool visualize = false;

  for(vector<sensor_msgs::PointCloud2>::const_iterator it = clusters.begin(); it != clusters.end(); it++)
  {
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*it, pcl_pc2);

    pcl::PointCloud<pcl::PointXYZI> cloud;
    pcl::fromPCLPointCloud2(pcl_pc2, cloud);

    Eigen::Vector4f min, max, centroid;
    pcl::getMinMax3D(cloud, min, max);
    pcl::compute3DCentroid(cloud, centroid);

    nomalizeCluster(cloud, centroid);

    Eigen::Matrix4f object = Eigen::MatrixXf::Identity(4,4);
    object(0,3) = centroid(0);
    object(1,3) = centroid(1);
    object(2,3) = centroid(2);

    //std::cout << "object before transformation: " << endl;
    //cout << object << std::endl;
    object = T * object;
    //std::cout << "object after transformation: " << endl;
    //cout << object << std::endl;

    Eigen::Vector2d map_pos =  Detector_Saver::world2map(Eigen::Vector2d(object(0,3), object(1,3)));
    //cout << "2D position on the map: " << map_pos << endl;

    int label = 0;
    if(map_pos(0)<0 || map_pos(0)>ncfm_map.cols || map_pos(1)<0 || map_pos(1)>ncfm_map.rows)
    {
      label = 0;
    }
    else
    {  
      label = getLabel( int(map_pos(1)), int(map_pos(0)) );
      cout << " label: " << label << endl;
    }

    if(visualize)
    {
      if(label==0)
        cv::circle(ncfm_map, cv::Point(map_pos(0), map_pos(1)), 5.0, cv::Scalar( 255, 255, 255 ), 1, 8);
      else
        cv::circle(ncfm_map, cv::Point(map_pos(0), map_pos(1)), 5.0, cv::Scalar( 0, 255, 0 ), 1, 8);
    }

    // visualization
    visualization_msgs::Marker marker;
    marker.header.stamp = ros::Time::now();
    marker.header.frame_id = frame_id;
    marker.ns = "adaptive_clustering";
    marker.id = it - clusters.begin();
    marker.type = visualization_msgs::Marker::LINE_LIST;

    draw_3d_boundingbox(marker, int(label), min, max);
    marker_array.markers.push_back(marker);

    ostringstream strs2;
    strs2 << ++i;
    string i_s = strs2.str();

    //pcl::io::savePCDFileASCII(time_s + "_" + i_s + ".pcd", cloud);

    std::ofstream cloud_file((time_s + "_" + i_s + ".txt").c_str());
    
    cout << "-------------------------------------------------------------------------" << endl;
    cout << "Saved " << cloud.points.size () << " data points to " << time_s + "_" + i_s + ".txt" << " label: " << label << std::endl;
    for (size_t k = 0; k < cloud.points.size (); ++k)
    {
       // std::cerr << "    " << cloud.points[k].x << " " << cloud.points[k].y << " " << cloud.points[k].z << " " << cloud.points[k].intensity << std::endl;
       cloud_file << cloud.points[k].x << " " << cloud.points[k].y << " " << cloud.points[k].z << " " << cloud.points[k].intensity << endl;
    }


    myfile << time_s + "_" + i_s + ".txt " << label << endl;
    i += 1;
  }

  if(visualize)
  {
    cv::imshow("Image", ncfm_map);
    cv::waitKey( 0 );
  }
 
  //cout << marker_array << endl;
  example_pub.publish(marker_array);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "save_simulation_data_node");
  
  ros::NodeHandle nh;

  Detector_Saver ds;

  nh.param<std::string>("frame_id", ds.frame_id, std::string("/velodyne"));

  string detector_topic = "/adaptive_clustering/clusters";
  ros::Subscriber detector_sub = nh.subscribe<lidar_object_detection::ClusterArray>(detector_topic, 1, &Detector_Saver::detectorCallback, &ds);



  ros::spin();

  myfile.close();

  return 0;
}


