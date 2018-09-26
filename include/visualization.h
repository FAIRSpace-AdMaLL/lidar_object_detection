#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>

void draw_3d_boundingbox(visualization_msgs::Marker& marker, const int label, const Eigen::Vector4f min, const Eigen::Vector4f max);


void draw_3d_boundingbox(visualization_msgs::Marker& marker, const int label, const Eigen::Vector4f min, const Eigen::Vector4f max)
{
    double r = 0, g = 0, b = 0;

    switch(label)
    {
        case 0: b = 0.0;// background
           break;
        case 1: b = 1.0; // pedestrains
           break;
        case 2: 
           g = 1.0; // trucks
           break;
        case 3:
           b = 1.0;
           r = 1.0; // pallet
           break;
        case 4: 
           g = 1.0; // pedestain
           break;
        case 5:
           b = 1.0; // truck
           break;
        case 6:
           r = 1.0;
           g = 1.0;
           b = 1.0; // wall
           break;
    }

    geometry_msgs::Point p[24];
    p[0].x = max[0]; p[0].y = max[1]; p[0].z = max[2];
    p[1].x = min[0]; p[1].y = max[1]; p[1].z = max[2];
    p[2].x = max[0]; p[2].y = max[1]; p[2].z = max[2];
    p[3].x = max[0]; p[3].y = min[1]; p[3].z = max[2];
    p[4].x = max[0]; p[4].y = max[1]; p[4].z = max[2];
    p[5].x = max[0]; p[5].y = max[1]; p[5].z = min[2];
    p[6].x = min[0]; p[6].y = min[1]; p[6].z = min[2];
    p[7].x = max[0]; p[7].y = min[1]; p[7].z = min[2];
    p[8].x = min[0]; p[8].y = min[1]; p[8].z = min[2];
    p[9].x = min[0]; p[9].y = max[1]; p[9].z = min[2];
    p[10].x = min[0]; p[10].y = min[1]; p[10].z = min[2];
    p[11].x = min[0]; p[11].y = min[1]; p[11].z = max[2];
    p[12].x = min[0]; p[12].y = max[1]; p[12].z = max[2];
    p[13].x = min[0]; p[13].y = max[1]; p[13].z = min[2];
    p[14].x = min[0]; p[14].y = max[1]; p[14].z = max[2];
    p[15].x = min[0]; p[15].y = min[1]; p[15].z = max[2];
    p[16].x = max[0]; p[16].y = min[1]; p[16].z = max[2];
    p[17].x = max[0]; p[17].y = min[1]; p[17].z = min[2];
    p[18].x = max[0]; p[18].y = min[1]; p[18].z = max[2];
    p[19].x = min[0]; p[19].y = min[1]; p[19].z = max[2];
    p[20].x = max[0]; p[20].y = max[1]; p[20].z = min[2];
    p[21].x = min[0]; p[21].y = max[1]; p[21].z = min[2];
    p[22].x = max[0]; p[22].y = max[1]; p[22].z = min[2];
    p[23].x = max[0]; p[23].y = min[1]; p[23].z = min[2];
    for(int i = 0; i < 24; i++) 
    {
      marker.points.push_back(p[i]);
      marker.scale.x = 0.05;
      marker.scale.y = 0.05;
      marker.scale.z = 0.05;
      marker.color.a = 1.0;

      marker.color.r = r;
      marker.color.g = g;
      marker.color.b = b;
    }
    
    marker.lifetime = ros::Duration(0.2);
}
