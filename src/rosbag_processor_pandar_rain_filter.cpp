#include <fstream>
#include <sstream>
#include <istream>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <map>
#include <thread>

#include <boost/foreach.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/c_local_time_adjustor.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem.hpp>

#include <ros/ros.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <pandar_pointcloud/point_types.hpp>

#include <pcl/octree/octree_search.h>
#include <pcl/search/search.h>
#include <pcl/segmentation/segment_differences.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/crop_box.h>

#include <pcl/common/common.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include<sstream>
#include <algorithm>
#include "opencv2/highgui/highgui.hpp"
//#include <pcl/filters/statistical_outlier_removal.h>

struct Range_point
{
  float distance;
  int rain_label;
  int ring_id;
  float intensity;
  int8_t return_type;
  float azimuth;
  int position;
  float x;
  float y;
  float z;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

typedef pcl::PointXYZI PointT;
typedef pandar_pointcloud::PointXYZIRADT PointXYZIRADT;
pcl::Filter<PointT>::Ptr outlier_removal_filter;
Eigen::Matrix4f prev_trans; 
std::string downsample_method;
double downsample_resolution;
pcl::Filter<PointT>::Ptr downsample_filter;
bool use_distance_filter;
double distance_near_thresh;
double distance_far_thresh;
int count = 0;


pcl::PointCloud<PointT>::Ptr cloud_full_top(new pcl::PointCloud<PointT>());
pcl::PointCloud<PointT>::Ptr cloud_full_left(new pcl::PointCloud<PointT>());
pcl::PointCloud<PointT>::Ptr cloud_full_right(new pcl::PointCloud<PointT>());
std::vector<nav_msgs::Odometry::ConstPtr> odom_msgs_top;
std::vector<nav_msgs::Odometry::ConstPtr> odom_msgs_left;
std::vector<nav_msgs::Odometry::ConstPtr> odom_msgs_right;
std::vector<sensor_msgs::PointCloud2::ConstPtr> cloud_msgs_top;
std::vector<sensor_msgs::PointCloud2::ConstPtr> cloud_msgs_left;
std::vector<sensor_msgs::PointCloud2::ConstPtr> cloud_msgs_right;
pcl::PointCloud<PointT>::ConstPtr filtered_top (new pcl::PointCloud<PointT>);
pcl::PointCloud<PointT>::ConstPtr filtered_left (new pcl::PointCloud<PointT>);
pcl::PointCloud<PointT>::ConstPtr filtered_right (new pcl::PointCloud<PointT>);

struct PointComparator
{
    bool operator()(const PointT & pt1, const PointT & pt2) const
    {
      if ( pt1.x != pt2.x ) return pt1.x  < pt2.x;
      if ( pt1.y != pt2.y ) return pt1.y < pt2.y;
      return pt1.z < pt2.z; 
    }
};

void init_directories(std::string &in_outpath)
{
  std::string path_label_str,
      path_ranges_str, path_point_cloud_str;

  path_label_str = std::string(in_outpath) + "/labels/";
  path_ranges_str = std::string(in_outpath) + "/range_images/";
  path_point_cloud_str = std::string(in_outpath) + "/point_cloud_images/";

  boost::filesystem::path base_dir(std::string(in_outpath).c_str());
  if(!boost::filesystem::exists(base_dir))
  {
    boost::filesystem::create_directories(base_dir);
  }

  boost::filesystem::path path_labels(path_label_str.c_str());
  boost::filesystem::path path_ranges(path_ranges_str.c_str());
  boost::filesystem::path path_point_cloud(path_point_cloud_str.c_str());

  boost::filesystem::create_directories(path_labels);
  boost::filesystem::create_directories(path_ranges);
  boost::filesystem::create_directories(path_point_cloud);
}

pcl::PointCloud<PointT>::ConstPtr outlier_removal(const pcl::PointCloud<PointT>::ConstPtr& cloud) {
  if(!outlier_removal_filter) {
    return cloud;
  }

  pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
  outlier_removal_filter->setInputCloud(cloud);
  outlier_removal_filter->filter(*filtered);
  filtered->header = cloud->header;

  return filtered;
}
  /**
 * @brief downsample a point cloud
 * @param cloud  input cloud
 * @return downsampled point cloud
 */
pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) {
  if(!downsample_filter) {
    return cloud;
  }
  double resolution = 0.01;
  pcl::octree::OctreePointCloud<PointT> octree(resolution);
  octree.setInputCloud(cloud);
  octree.addPointsFromInputCloud();

  pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
  octree.getOccupiedVoxelCenters(filtered->points);

  filtered->width = filtered->size();
  filtered->height = 1;
  filtered->is_dense = false;

  // pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
  // downsample_filter->setInputCloud(cloud);
  // downsample_filter->filter(*filtered);

  return filtered;
}

pcl::PointCloud<PointT>::ConstPtr distance_filter(const pcl::PointCloud<PointT>::ConstPtr& cloud) {
  pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
  filtered->reserve(cloud->size());

  std::copy_if(cloud->begin(), cloud->end(), std::back_inserter(filtered->points), [&](const PointT& p) {
    double d = p.getVector3fMap().norm();
    return d > distance_near_thresh && d < distance_far_thresh;
  });

  filtered->width = filtered->size();
  filtered->height = 1;
  filtered->is_dense = false;

  filtered->header = cloud->header;

  return filtered;
}

std::string get_string_time(std::chrono::time_point<std::chrono::system_clock> time)
{
  std::time_t t_time = std::chrono::system_clock::to_time_t(time);
  return std::ctime(&t_time);
}

bool file_exists(const std::string &name)
{
  std::ifstream f(name.c_str());
  return f.good();
}

bool dir_exist(const std::string &s)
{
  struct stat buffer;
  return (stat(s.c_str(), &buffer) == 0);
}

bool cmpf(float A, float B, float epsilon = 0.05f)
{
    return (fabs(A - B) < epsilon);
}

// Count number of missing points between azimuth scans
int missing_pts_counter(int num){
  if (num % 20 == 0 && num > 20)
      return (num / 20) - 1;
  else if ((num - 1) % 20 == 0 && num-1 > 20)
      return ((num - 1) / 20) - 1;
  else if ((num + 1) % 20 == 0 && num+1 > 20)
      return ((num + 1) / 20) - 1;
  else
      return 0;      
}

// fill points ring by ring, also add blank points when lidar points are skipped in a ring
std::vector<Range_point> fill_points(int num, std::vector<Range_point> ring_pts, PointXYZIRADT pt_c){
  //add missing points as blank points
  for (int i = 0; i < num ; i++){
    Range_point pt;
    count += 1;
    pt.distance = 0; //we set distance as -1 for blank points
    pt.intensity = 0;
    pt.return_type = -1;    
    pt.azimuth = -1;
    pt.rain_label = -1;
    pt.x = -1;
    pt.y = -1;
    pt.z = -1;
    //ROS_WARN("Blank point: %f, count: %d!!", pt.azimuth, count);
    ring_pts.push_back(pt);
  }
  return ring_pts;
}

void remove_ground_points(pcl::PointCloud<PointT>::ConstPtr cloud_no_rain, pcl::PointCloud<PointT>::Ptr cloud_no_rain_ngnd,
                    pcl::PointCloud<PointT>::ConstPtr cloud_top, pcl::PointCloud<PointT>::Ptr cloud_top_ngnd){
  pcl::ModelCoefficients::Ptr plane (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices);
  pcl::PointCloud<PointT>::Ptr cloud_plane (new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr cloud_inliers (new pcl::PointCloud<PointT>);
  // Make room for a plane equation (ax+by+cz+d=0)
  plane->values.resize (4);

  pcl::SACSegmentation<PointT> seg;				// Create the segmentation object
  seg.setOptimizeCoefficients (true);				// Optional
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setDistanceThreshold (0.1f);
  seg.setInputCloud (cloud_no_rain);
  seg.segment (*inliers_plane, *plane);

  if (inliers_plane->indices.size () == 0) {
    PCL_ERROR ("Could not estimate a planar model for the given dataset.\n");
    return;
  }

  // Extract inliers
  pcl::ExtractIndices<PointT> extract1;
  extract1.setInputCloud (cloud_no_rain);
  extract1.setIndices (inliers_plane);
  extract1.setNegative (false);			// Extract the inliers
  extract1.filter (*cloud_inliers);		// cloud_inliers contains the plane


  PointT minPt, maxPt;
  pcl::getMinMax3D (*cloud_inliers, minPt, maxPt);

  pcl::ExtractIndices<PointT> extract;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  for (int i = 0; i < (*cloud_no_rain).size(); i++)
  {
    if (cloud_no_rain->points[i].z < (minPt.z + 1.35)) // e.g. remove all pts below ground
    {
      inliers->indices.push_back(i);
    }
  }
  extract.setInputCloud(cloud_no_rain);
  extract.setIndices(inliers);

  // Extract outliers
  extract.setNegative (true);				// Extract the outliers
  extract.filter (*cloud_no_rain_ngnd);		// original no rain cloud without ground points and below points

  pcl::ExtractIndices<PointT> extract2;
  pcl::PointIndices::Ptr inliers1(new pcl::PointIndices());
  for (int i = 0; i < (*cloud_top).size(); i++) //
  {
    if (cloud_top->points[i].z < (minPt.z + 1.35)) // e.g. remove all pts below zAvg
    {
      inliers1->indices.push_back(i);
    }
  }
  extract2.setInputCloud(cloud_top);
  extract2.setIndices(inliers1);

  // Extract outliers
  extract2.setNegative (true);				// Extract the outliers
  extract2.filter (*cloud_top_ngnd);		// original cloud without ground points and below points
}

void remove_non_building_points(pcl::PointCloud<PointT>::Ptr cloud_top_ngnd, pcl::PointCloud<PointT>::Ptr cloud_no_rain_ngnd,
                                pcl::PointCloud<PointT>::Ptr cloud_top_boxed, pcl::PointCloud<PointT>::Ptr cloud_no_rain_boxed){
    pcl::CropBox<PointT> boxFilter;
    boxFilter.setMin(Eigen::Vector4f(-21.0, -30.0, -16, 1.0));
    boxFilter.setMax(Eigen::Vector4f(56.0, 20.0, 16, 1.0));
    boxFilter.setInputCloud(cloud_top_ngnd);
    boxFilter.filter(*cloud_top_boxed);
    boxFilter.setInputCloud(cloud_no_rain_ngnd);
    boxFilter.filter(*cloud_no_rain_boxed);
}

void make_range_vectors(pcl::PointCloud<PointXYZIRADT>::Ptr cloud_t_orig, pcl::PointCloud<PointT>::Ptr cloud_t_xyz,
                        pcl::PointCloud<PointT>::Ptr rain_points, std::vector<std::vector<Range_point>> &ring_ids_first,
                        std::vector<std::vector<Range_point>> &ring_ids_last){
    //build KDtree of rain points to populate label image
  pcl::KdTreeFLANN<PointT> kdtree;
  kdtree.setInputCloud (rain_points);
  // K nearest neighbor search
  int K = 1;
  std::vector<int> pointIdxNKNSearch(K);
  std::vector<float> pointNKNSquaredDistance(K);
  for (int ring_id = 0; ring_id < 40 ; ring_id++)
    {
      for (int i = 0; i < (*cloud_t_orig).size(); i++) {
        //std::cout << "Azimuth: " << cloud_t_orig->points[i].azimuth << " Ret_type:" << static_cast<int16_t> (cloud_t_orig->points[i].return_type) << std::endl;
        PointXYZIRADT pt_c = cloud_t_orig->points[i];
        PointT pt_trunc = cloud_t_xyz->points[i];
        if (pt_c.ring == ring_id){
          Range_point pt;
          pt.ring_id = ring_id;
          pt.distance = pt_c.distance;
          pt.intensity = pt_c.intensity;
          pt.return_type = pt_c.return_type;    
          pt.azimuth = pt_c.azimuth;
          pt.x = pt_c.x;
          pt.y = pt_c.y;
          pt.z = pt_c.z;
          if ( kdtree.nearestKSearch (pt_trunc, K, pointIdxNKNSearch, pointNKNSquaredDistance) == 1 && pointNKNSquaredDistance[0] == 0.0 ) {
            pt.rain_label = 1; //point found in rain points so we mark the label as 1
          }
          else{
            pt.rain_label = 0;
          }
          if (pt.return_type == 6){ //add first & last ranges
            if(ring_ids_first[ring_id].empty()){ //No points stored in the ring yet.
              // Check if any points are skipped in the beginning add blank or no points are skipped store the point #TODO
              pt.position = 0; //first point in ring
              ring_ids_first[ring_id].push_back(pt);
              count += 1;
              //ROS_WARN("First point first range: %f, count: %d, ret_type: %d, ring_id: %d !!", pt.azimuth, count, static_cast<int16_t>(pt.return_type), ring_id);
            }
            else if(ring_ids_last[ring_id].empty()){ //No points stored in the ring yet.
              // Check if any points are skipped in the beginning add blank or no points are skipped store the point #TODO
              pt.position = 0; //first point in ring
              ring_ids_last[ring_id].push_back(pt);
              count += 1;
              //ROS_WARN("First point last range: %f, count: %d, ret_type: %d, ring_id: %d !!", pt.azimuth, count, static_cast<int16_t>(pt.return_type), ring_id);
            }            
            else{ //ring row has already some points, check and add blank points, then add current point
              int diff_azi_first = abs(ring_ids_first[ring_id].back().azimuth - pt_c.azimuth);
              if (diff_azi_first <= 0){
                ROS_WARN("Error: This can't happen!!");
                return;
              }
              else{
                int no_missing_pts = missing_pts_counter(diff_azi_first);
                //ROS_WARN("missing point first+last: %d!!", no_missing_pts);
                  if (no_missing_pts == 0){ //just add the next point
                    pt.position = ring_ids_first[ring_id].size(); 
                    ring_ids_first[ring_id].push_back(pt);
                    count += 1;
                    //ROS_WARN("Next point first: %f, count: %d, ret_type: %d, ring_id: %d !!", pt.azimuth, count, static_cast<int16_t>(pt.return_type), ring_id);
                  }
                  else{ // add null points to first ranges
                    ring_ids_first[ring_id] = fill_points(no_missing_pts, ring_ids_first[ring_id], pt_c);
                    pt.position = ring_ids_first[ring_id].size(); 
                    ring_ids_first[ring_id].push_back(pt);
                    count += 1;
                    //ROS_WARN("Null: %d, next point first: %f, count: %d, ret_type: %d, ring_id: %d !!",no_missing_pts, pt.azimuth, count, static_cast<int16_t>(pt.return_type), ring_id);
                  }
              }
              int diff_azi_last = abs(ring_ids_last[ring_id].back().azimuth - pt_c.azimuth);
              if (diff_azi_last <= 0){
                ROS_WARN("Error: This can't happen!!");
                return;
              }
              else{
                int no_missing_pts = missing_pts_counter(diff_azi_last);
                //ROS_WARN("missing point first+last: %d!!", no_missing_pts);
                  if (no_missing_pts == 0){ //just add the next point
                    pt.position = ring_ids_last[ring_id].size(); 
                    ring_ids_last[ring_id].push_back(pt);
                    count += 1;
                    //ROS_WARN("Next point last: %f, count: %d, ret_type: %d, ring_id: %d !!", pt.azimuth, count, static_cast<int16_t>(pt.return_type), ring_id);
                  }
                  else{ // add null points to last ranges
                    ring_ids_last[ring_id] = fill_points(no_missing_pts, ring_ids_last[ring_id], pt_c);
                    pt.position = ring_ids_last[ring_id].size(); 
                    ring_ids_last[ring_id].push_back(pt);
                    count += 1;
                    //ROS_WARN("Null: %d, next point last: %f, count: %d, ret_type: %d, ring_id: %d !!",no_missing_pts, pt.azimuth, count, static_cast<int16_t>(pt.return_type), ring_id);
                  }
              }
            }
          }        
          else if (pt.return_type == 2 || pt.return_type == 4){ //only first ranges (2,4)
            if(ring_ids_first[ring_id].empty()){ //No points stored in the ring yet.
              // Check if any points are skipped in the beginning add blank or no points are skipped store the point #TODO
              pt.position = 0; 
              ring_ids_first[ring_id].push_back(pt);
              count += 1;                  
              //ROS_WARN("First point first: %f, count: %d, ret_type: %d, ring_id: %d !!", pt.azimuth, count, static_cast<int16_t>(pt.return_type), ring_id);
            }
            else{ //ring row has already some points, check and add blank points, then add current point
              int diff_azi = abs(ring_ids_first[ring_id].back().azimuth - pt_c.azimuth);
              if (diff_azi <= 0){
                ROS_WARN("Error: This can't happen!!");
                return;
              }
              else{
                int no_missing_pts = missing_pts_counter(diff_azi);
              // ROS_WARN("missing point first: %d!!", no_missing_pts);
                if (no_missing_pts == 0){ //just add the next point
                  pt.position = ring_ids_first[ring_id].size(); 
                  ring_ids_first[ring_id].push_back(pt);
                  count += 1;                     
                  //ROS_WARN("Next point first: %f, count: %d, ret_type: %d, ring_id: %d !!", pt.azimuth, count, static_cast<int16_t>(pt.return_type), ring_id);
                }
                else{ // add null points to both first ranges
                  ring_ids_first[ring_id] = fill_points(no_missing_pts, ring_ids_first[ring_id], pt_c);
                  pt.position = ring_ids_first[ring_id].size(); 
                  ring_ids_first[ring_id].push_back(pt);
                  count += 1;      
                  //ROS_WARN("Null: %d, next point first: %f, count: %d, ret_type: %d, ring_id: %d !!", no_missing_pts, pt.azimuth, count, static_cast<int16_t>(pt.return_type), ring_id);
                }
              }
            }
          }
          else if (pt.return_type == 3 || pt.return_type == 5){ //only add last ranges (3,5)
            if(ring_ids_last[ring_id].empty()){ //No points stored in the ring yet.
              // Check if any points are skipped in the beginning add blank or no points are skipped store the point #TODO
              pt.position = 0; 
              ring_ids_last[ring_id].push_back(pt);
              count += 1;                  
              //ROS_WARN("First point last: %f, count: %d, ret_type: %d, ring_id: %d !!", pt.azimuth, count, static_cast<int16_t>(pt.return_type), ring_id);
            }
            else{ //ring row has already some points, check and add blank points, then add current point
              int diff_azi = abs(ring_ids_last[ring_id].back().azimuth - pt_c.azimuth);
              if (diff_azi == 0){
                ROS_WARN("Error: This can't happen!!");
                return;
              }
              else{
                int no_missing_pts = missing_pts_counter(diff_azi);
              // ROS_WARN("missing point first: %d!!", no_missing_pts);
                if (no_missing_pts == 0){ //just add the next point
                  pt.position = ring_ids_last[ring_id].size(); 
                  ring_ids_last[ring_id].push_back(pt);
                  count += 1;                     
                  //ROS_WARN("Next point last: %f, count: %d, ret_type: %d, ring_id: %d !!", pt.azimuth, count, static_cast<int16_t>(pt.return_type), ring_id);
                }
                else{ // add null points to last ranges
                  ring_ids_last[ring_id] = fill_points(no_missing_pts, ring_ids_last[ring_id], pt_c);
                  pt.position = ring_ids_last[ring_id].size(); 
                  ring_ids_last[ring_id].push_back(pt);
                  count += 1;      
                  //ROS_WARN("Null: %d, next point last: %f, count: %d, ret_type: %d, ring_id: %d !!", no_missing_pts, pt.azimuth, count, static_cast<int16_t>(pt.return_type), ring_id);
                }
              }
            }
          }       
        }
      }
    }
}

void range_image_generator(cv::Mat first_range_img, cv::Mat last_range_img, cv::Mat labels,
                          std::vector<std::vector<Range_point>> &ring_ids_first,
                          std::vector<std::vector<Range_point>> &ring_ids_last){
  cv::Mat chans[3], chans1[3];
  split(first_range_img, chans);
  split(last_range_img, chans1);
  for (int i = 0; i < 40; i++) {
    //ROS_WARN("Ring length first: %d, ring id: %d", ring_ids_first[i].size(), i);
    for (int j = 0; j < static_cast<int>(ring_ids_first[i].size()); j++) {
        chans[0].row(i).col(j) = ring_ids_first[i].at(j).distance;
        chans[1].row(i).col(j) = ring_ids_first[i].at(j).intensity;
        chans[2].row(i).col(j) = static_cast<float_t>(ring_ids_first[i].at(j).return_type);

        if (ring_ids_first[i].at(j).rain_label == 1)
          labels.row(i).col(j) = 1;
        // std::cout << "distance: " << "i: " << i << "j: " << j << chans[0].row(i).col(j) << std::endl;
        // std::cout << "intensity: " << "i: " << i << "j: " << j << chans[1].row(i).col(j) << std::endl;
        // std::cout << "return_type: " << "i: " << i << "j: " << j << chans[2].row(i).col(j) << std::endl;
    }
    //ROS_WARN("Ring length last: %d, ring id: %d", ring_ids_last[i].size(), i);     
    for (int j = 0; j < static_cast<int>(ring_ids_last[i].size()); j++) {
        chans1[0].row(i).col(j) = ring_ids_last[i].at(j).distance;
        chans1[1].row(i).col(j) = ring_ids_last[i].at(j).intensity;
        chans1[2].row(i).col(j) = static_cast<float_t>(ring_ids_last[i].at(j).return_type);
    }
  }

//  std::cout << "No of labels: " << cv::countNonZero(labels) << std::endl;

  cv::merge(chans, 3, first_range_img);
  cv::merge(chans1, 3, last_range_img);
}

void point_cloud_image_checker(cv::Mat point_cloud_img_first, cv::Mat point_cloud_img_last, std::vector<std::vector<Range_point>> &ring_ids_first, 
                          std::vector<std::vector<Range_point>> &ring_ids_last){
  cv::Mat chans[3], chans1[3];
  split(point_cloud_img_first, chans);
  split(point_cloud_img_last, chans1);
  for (int i = 0; i < 40; i++) {
    //ROS_WARN("Ring length first: %d, ring id: %d", ring_ids_first[i].size(), i);
    for (int j = 0; j < static_cast<int>(ring_ids_first[i].size()); j++) {
      if (ring_ids_first[i].at(j).return_type != -1 ){
        chans[0].row(i).col(j) = ring_ids_first[i].at(j).x;
        chans[1].row(i).col(j) = ring_ids_first[i].at(j).y;
        chans[2].row(i).col(j) = ring_ids_first[i].at(j).z;
      }
    }
    //ROS_WARN("Ring length last: %d, ring id: %d", ring_ids_last[i].size(), i);      
    for (int j = 0; j < static_cast<int>(ring_ids_last[i].size()); j++) {
      if (ring_ids_last[i].at(j).return_type != -1 ){      
        chans1[0].row(i).col(j) = ring_ids_last[i].at(j).x;
        chans1[1].row(i).col(j) = ring_ids_last[i].at(j).y;
        chans1[2].row(i).col(j) = ring_ids_last[i].at(j).z;
      }
    }
  }

  cv::merge(chans, 3, point_cloud_img_first);
  cv::merge(chans1, 3, point_cloud_img_last);  
}

void reconstruct_point_cloud(pcl::PointCloud<PointT>::Ptr reconstruct_pt_cloud, std::string point_cloud_first_name, std::string point_cloud_last_name){
  cv::Mat point_cloud_first_image = cv::imread(point_cloud_first_name, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
  for( int i = 0; i < point_cloud_first_image.rows; ++i){
    for( int j = 0; j < point_cloud_first_image.cols; ++j ){
      PointT point;
      point.x = point_cloud_first_image.at<cv::Vec3f>(i,j)[0];
      point.y = point_cloud_first_image.at<cv::Vec3f>(i,j)[1];
      point.z = point_cloud_first_image.at<cv::Vec3f>(i,j)[2];    
      if (point.x != 0.0 && point.y != 0.0 && point.z != 0.0 )
        reconstruct_pt_cloud->push_back(point);    
    }
  }
  //build KDtree of first range points 
  pcl::KdTreeFLANN<PointT> kdtree;
  kdtree.setInputCloud (reconstruct_pt_cloud);  
  // K nearest neighbor search
  int K = 1;
  std::vector<int> pointIdxNKNSearch(K);
  std::vector<float> pointNKNSquaredDistance(K);
  cv::Mat point_cloud_last_image = cv::imread(point_cloud_last_name, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
  for( int i = 0; i < point_cloud_last_image.rows; ++i){
    for( int j = 0; j < point_cloud_last_image.cols; ++j ){
      PointT point;
      point.x = point_cloud_last_image.at<cv::Vec3f>(i,j)[0];
      point.y = point_cloud_last_image.at<cv::Vec3f>(i,j)[1];
      point.z = point_cloud_last_image.at<cv::Vec3f>(i,j)[2];    
      if (point.x != 0.0 && point.y != 0.0 && point.z != 0.0 ){
        if ( kdtree.nearestKSearch (point, K, pointIdxNKNSearch, pointNKNSquaredDistance) == 1 && pointNKNSquaredDistance[0] != 0.0 ){
          //non duplicate points only
          reconstruct_pt_cloud->push_back(point);    
        }
      }
    }
  }  
}

void process_pointclouds(std::vector<sensor_msgs::PointCloud2::ConstPtr> &clouds_top, const std::string output_path, const std::string no_rain_pcd_path){

  pcl::PointCloud<PointT>::Ptr cloud_no_rain_orig (new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr cloud_t_xyz (new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointXYZIRADT>::Ptr cloud_t_orig(new pcl::PointCloud<PointXYZIRADT>);
  pcl::io::loadPCDFile<PointT> (no_rain_pcd_path, *cloud_no_rain_orig);

  for (std::vector<int>::size_type ind = 0; ind != clouds_top.size(); ind++) //
  {
    pcl::fromROSMsg(*clouds_top[ind], *cloud_t_orig);
    pcl::fromROSMsg(*clouds_top[ind], *cloud_t_xyz);

    pcl::PointCloud<PointT>::ConstPtr cloud_t = cloud_t_xyz;//downsample(cloud_t_orig);
    pcl::PointCloud<PointT>::ConstPtr cloud_no_rain = cloud_no_rain_orig;//downsample(cloud_no_rain_orig);

    // Segment the ground and remove all points below ground
    pcl::PointCloud<PointT>::Ptr cloud_no_rain_ngnd (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_top_ngnd (new pcl::PointCloud<PointT>);

    remove_ground_points(cloud_no_rain, cloud_no_rain_ngnd, cloud_t, cloud_top_ngnd);

    // Segment and remove points outside the building
    pcl::PointCloud<PointT>::Ptr cloud_top_boxed (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_no_rain_boxed (new pcl::PointCloud<PointT>);
    remove_non_building_points(cloud_top_ngnd, cloud_no_rain_ngnd, cloud_top_boxed, cloud_no_rain_boxed);

    pcl::PointCloud<PointT>::Ptr out (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr rain_points (new pcl::PointCloud<PointT>);
    //Segment differences
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    pcl::SegmentDifferences<PointT> sdiff;
    sdiff.setInputCloud(cloud_top_boxed);
    sdiff.setTargetCloud(cloud_no_rain_boxed);
    sdiff.setSearchMethod(tree);
    sdiff.setDistanceThreshold(0);
    sdiff.segment(*out);
    
    // Get point cloud difference
    pcl::getPointCloudDifference<PointT> (*cloud_top_boxed,*cloud_no_rain_boxed,0.1f,tree,*rain_points);
    // std::cout << *rain_points; 
    // std::cout << *cloud_t; 
    // std::cout << *cloud_no_rain; 


    // Making range image 40 x 1800 size and label image 40 x 1400
    std::vector<std::vector<Range_point>> ring_ids_first(40) ; 
    std::vector<std::vector<Range_point>> ring_ids_last(40) ;
    make_range_vectors(cloud_t_orig, cloud_t_xyz, rain_points, ring_ids_first, ring_ids_last);
    //checking if all ring ranges are 1800 points
    // Sometimes 1796, 1798 #Todo (check why??)

    //Generate labels for rain and non rain points
    cv::Mat labels(40, 1800, CV_8UC1, cv::Scalar(0,0));
    //Creating range images first and last
    cv::Mat first_range_img(40, 1800, CV_32FC3, cv::Scalar(0.0,0.0,0.0));
    cv::Mat last_range_img(40, 1800, CV_32FC3, cv::Scalar(0.0,0.0,0.0));
    range_image_generator(first_range_img, last_range_img, labels, ring_ids_first, ring_ids_last);

    //Check if the range image matches the original point cloud
    cv::Mat point_cloud_img_first(40, 1800, CV_32FC3, cv::Scalar(0.0,0.0,0.0));
    cv::Mat point_cloud_img_last(40, 1800, CV_32FC3, cv::Scalar(0.0,0.0,0.0));    
    point_cloud_image_checker(point_cloud_img_first, point_cloud_img_last, ring_ids_first, ring_ids_last);

    // cv::Vec3f intensity = first_range_img.at<cv::Vec3f>(29, 1599);
    // std::cout << "value is original: " << "i: " << 29 << "j: " << 1599 << intensity << std::endl;

    std::stringstream first_range_name, last_range_name, point_cloud_first_name, point_cloud_last_name, label_name;
    std::string ss1 = "/range_images/first_range_img_";
    std::string ss2 = "/range_images/last_range_img_";   
    std::string ss3 = "/point_cloud_images/point_cloud_first_img_";   
    std::string ss4 = "/point_cloud_images/point_cloud_last_img_";   
    std::string ss5 = "/labels/label_";    
    std::string type1 = ".exr";
    std::string type2 = ".png";    
    first_range_name<<output_path<<ss1<<(ind)<<type1;
    last_range_name<<output_path<<ss2<<(ind)<<type1;
    point_cloud_first_name<<output_path<<ss3<<(ind)<<type1;
    point_cloud_last_name<<output_path<<ss4<<(ind)<<type1;
    label_name<<output_path<<ss5<<(ind)<<type2;    

    //Saving the range images
    cv::imwrite(first_range_name.str(), first_range_img);
    cv::imwrite(last_range_name.str(), last_range_img);

    //Saving point cloud images
    cv::imwrite(point_cloud_first_name.str(), point_cloud_img_first);
    cv::imwrite(point_cloud_last_name.str(), point_cloud_img_last);
    
    //Saving the label images
    cv::imwrite(label_name.str(), labels*255);
    
    //Reconstruct point cloud from image
    pcl::PointCloud<PointT>::Ptr reconstruct_pt_cloud (new pcl::PointCloud<PointT>);
    reconstruct_point_cloud(reconstruct_pt_cloud, point_cloud_first_name.str(), point_cloud_last_name.str());    

    // std::cout << *cloud_t_xyz << std::endl;
    // std::cout << *reconstruct_pt_cloud << std::endl;

    //Verify if point cloud is reconstructed accurately from range image correspondences
    if ((*cloud_t_xyz).size() != (*reconstruct_pt_cloud).size()){
      ROS_WARN("Some points are skipped, check again!!");
      return;
    }

    // //Visualization
    // //Color handlers for red, green, blue and yellow color
    // pcl::visualization::PointCloudColorHandlerCustom<PointT> red(cloud_no_rain,255,0,0);
    // pcl::visualization::PointCloudColorHandlerCustom<PointT> blue(cloud_t_xyz,0,0,255);
    // pcl::visualization::PointCloudColorHandlerCustom<PointT> green(reconstruct_pt_cloud,0,255,0);
    // pcl::visualization::PointCloudColorHandlerCustom<PointT> yellow(rain_points,255,255,0);    
    // pcl::visualization::PCLVisualizer vis("3D View");
    // // // vis.addPointCloud(cloud_no_rain_boxed,red,"src",0);
    // vis.addPointCloud(cloud_t_xyz,blue,"tgt",0);
    // vis.addPointCloud(reconstruct_pt_cloud,green,"reconstruct_pt_cloud",0);
    // // vis.addPointCloud(rain_points,yellow,"rain_points",0);
    // while(!vis.wasStopped())
    // {
    //         vis.spinOnce();
    // }  
  }
}

void check_images(){
  cv::Mat input = cv::imread("/home/nithilan/Downloads/pointcloud_processed/range_images/first_range_img_0.exr", cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
  cv::Vec3f intensity = input.at<cv::Vec3f>(29, 1599);
  std::cout << "value is checking: " << "i: " << 29 << "j: " << 1599 << intensity << std::endl;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "rosbag_processor_pandar_rain_filter");

  ros::NodeHandle private_node_handle("~");

  std::string file_path;
  std::string output_path_;
  std::string no_rain_pcd_path_;

  use_distance_filter = private_node_handle.param<bool>("use_distance_filter", true);
  distance_near_thresh = private_node_handle.param<double>("distance_near_thresh", 1.0);
  distance_far_thresh = private_node_handle.param<double>("distance_far_thresh", 100.0);

  std::string downsample_method = private_node_handle.param<std::string>("downsample_method", "VOXELGRID");
  double downsample_resolution = private_node_handle.param<double>("downsample_resolution", 0.1);

  if(downsample_method == "VOXELGRID") {
    std::cout << "downsample: VOXELGRID " << downsample_resolution << std::endl;
    boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    downsample_filter = voxelgrid;
  } else if(downsample_method == "APPROX_VOXELGRID") {
    std::cout << "downsample: APPROX_VOXELGRID " << downsample_resolution << std::endl;
    boost::shared_ptr<pcl::ApproximateVoxelGrid<PointT>> approx_voxelgrid(new pcl::ApproximateVoxelGrid<PointT>());
    approx_voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    downsample_filter = approx_voxelgrid;
  } else {
    if(downsample_method != "NONE") {
      std::cerr << "warning: unknown downsampling type (" << downsample_method << ")" << std::endl;
      std::cerr << "       : use passthrough filter" << std::endl;
    }
    std::cout << "downsample: NONE" << std::endl;
  }
  //   std::string outlier_removal_method = private_node_handle.param<std::string>("outlier_removal_method", "STATISTICAL");
  // if(outlier_removal_method == "STATISTICAL") {
  //   int mean_k = private_node_handle.param<int>("statistical_mean_k", 20);
  //   double stddev_mul_thresh = private_node_handle.param<double>("statistical_stddev", 1.0);
  //   std::cout << "outlier_removal: STATISTICAL " << mean_k << " - " << stddev_mul_thresh << std::endl;

  //   pcl::StatisticalOutlierRemoval<PointT>::Ptr sor(new pcl::StatisticalOutlierRemoval<PointT>());
  //   sor->setMeanK(mean_k);
  //   sor->setStddevMulThresh(stddev_mul_thresh);
  //   outlier_removal_filter = sor;
  // } else {
  //   std::cout << "outlier_removal: NONE" << std::endl;
  // }

  private_node_handle.param<std::string>("file_path", file_path, "");
  ROS_INFO("[%s] file_path: %s", ros::this_node::getName().c_str(), file_path.c_str());

  private_node_handle.param<std::string>("output_path", output_path_, "");
  ROS_INFO("[%s] output_path: %s", ros::this_node::getName().c_str(), output_path_.c_str());

  private_node_handle.param<std::string>("no_rain_pcd_path", no_rain_pcd_path_, "");
  ROS_INFO("[%s] no_rain_pcd_path: %s", ros::this_node::getName().c_str(), no_rain_pcd_path_.c_str());

  if (!file_exists(no_rain_pcd_path_))
  {
    ROS_WARN("[%s] no_rain_pcd_path: %s does not exist. Terminating.", ros::this_node::getName().c_str(), no_rain_pcd_path_.c_str());
    return 1;
  }

  if (!file_exists(file_path))
  {
    ROS_WARN("[%s] file_path: %s does not exist. Terminating.", ros::this_node::getName().c_str(), file_path.c_str());
    return 1;
  }

  if (!dir_exist(output_path_))
  {
    ROS_WARN("[%s] output_path: %s does not exist. Creating Directory", ros::this_node::getName().c_str(),
              output_path_.c_str());
  }

  init_directories(output_path_);

  auto start_time = std::chrono::system_clock::now();
  std::cout << "Starting at: " << get_string_time(start_time) << std::endl;

  rosbag::Bag input_bag;

  std::cout << "Reading Input Rosbag: " << file_path << std::endl;
  input_bag.open(file_path, rosbag::bagmode::Read);

  std::cout << "Reading messages..." << std::endl;
  rosbag::View rosbag_view(input_bag);
  size_t messages = 0;
  size_t total_messages = rosbag_view.size();

  for(rosbag::MessageInstance const m: rosbag_view)
  {
    float progress = (float) messages / (float) total_messages * 100.;

    if(m.getTopic() == "/pandar_points_ex")
    {
      sensor_msgs::PointCloud2::ConstPtr cur = m.instantiate<sensor_msgs::PointCloud2>();
      if (cur != NULL)
      {
        cloud_msgs_top.push_back(cur);
      }
    }


    if (messages % 100 == 0)
      std::cout << "\rProgress: (" << messages << " / " << total_messages << ") " << progress << "%            ";
    messages++;
  }
  std::cout << std::endl;
  input_bag.close();
  std::cout << "No of point cloud msgs: " << cloud_msgs_top.size() << std::endl;

  process_pointclouds(cloud_msgs_top, output_path_, no_rain_pcd_path_);
  std::cout << "Range and Label images generated! " << std::endl;
  // check_images();
  // pcl::visualization::PCLVisualizer::Ptr viewer_final (new pcl::visualization::PCLVisualizer ("3D Viewer pointcloud"));
  // viewer_final->setBackgroundColor (255, 255, 255);
  // // Coloring and visualizing target cloud (green). (0, 255, 0)
  // pcl::visualization::PointCloudColorHandlerCustom<PointT>
  // target_color (cloud_t, 0, 0, 205);
  // viewer_final->addPointCloud<PointT> (cloud_t, target_color, "cloud full");
  // viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
  //                                                 1, "cloud full");
  // viewer_final->addCoordinateSystem (1.0, "global");
  // viewer_final->initCameraParameters ();
  
  // while (!viewer_final->wasStopped ())
  // {
  //   viewer_final->spinOnce (100);
  // }
  // ros::spin();
  return 0;
}
