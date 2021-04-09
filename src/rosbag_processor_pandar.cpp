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


#include <pcl/octree/octree_search.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/approximate_voxel_grid.h>
//#include <pcl/filters/statistical_outlier_removal.h>

typedef pcl::PointXYZI PointT;
pcl::Filter<PointT>::Ptr outlier_removal_filter;
Eigen::Matrix4f prev_trans; 
std::string downsample_method;
double downsample_resolution;
pcl::Filter<PointT>::Ptr downsample_filter;
bool use_distance_filter;
double distance_near_thresh;
double distance_far_thresh;

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
  double resolution = 0.05;
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

void get_rosbag_view_data(rosbag::View &view, ros::Time &start_time, ros::Time &end_time, ros::Duration &duration)
{
  start_time = view.getBeginTime();
  end_time = view.getEndTime();
  duration = view.getEndTime() - view.getBeginTime();
}

void write_data(const rosbag::MessageInstance &message, rosbag::Bag &out_rosbag)
{
  out_rosbag.write(message.getTopic(), message.getTime(), message);
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

void writeStringToFile(const std::string& in_string, std::string path)
{
  std::ofstream outfile(path);
  if(!outfile.is_open())
  {
    ROS_ERROR("Couldn't open 'output.txt'");
  }

  outfile << in_string << std::endl;
  outfile.close();
}

static Eigen::Isometry3d odom2isometry(const nav_msgs::OdometryConstPtr& odom_msg) {
  const auto& orientation = odom_msg->pose.pose.orientation;
  const auto& position = odom_msg->pose.pose.position;

  Eigen::Quaterniond quat;
  quat.w() = orientation.w;
  quat.x() = orientation.x;
  quat.y() = orientation.y;
  quat.z() = orientation.z;

  Eigen::Isometry3d isometry = Eigen::Isometry3d::Identity();
  isometry.linear() = quat.toRotationMatrix();
  isometry.translation() = Eigen::Vector3d(position.x, position.y, position.z);
  return isometry;
}


void process_pointclouds(std::vector<sensor_msgs::PointCloud2::ConstPtr> &clouds_top, std::vector<sensor_msgs::PointCloud2::ConstPtr> &clouds_left,
                        std::vector<sensor_msgs::PointCloud2::ConstPtr> &clouds_right, std::vector<nav_msgs::Odometry::ConstPtr> &odoms_top,
                        std::vector<nav_msgs::Odometry::ConstPtr> &odoms_left, std::vector<nav_msgs::Odometry::ConstPtr> &odoms_right){
  int size = 1;//odoms_top.size();
  for (int ind = 0; ind < size; ind++){
    Eigen::Isometry3d pose_t = odom2isometry(odoms_top[ind]);
    Eigen::Matrix4f pose_top = pose_t.matrix().cast <float>();
    sensor_msgs::PointCloud2 cloud_msg_top;
    pcl_ros::transformPointCloud(pose_top, *clouds_top[ind], cloud_msg_top);
    pcl::PointCloud<PointT>::Ptr cloud_t(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(cloud_msg_top, *cloud_t);
    // filtered_top = distance_filter(cloud_t);
    // filtered_top = downsample(filtered_top);
    // filtered_top = outlier_removal(filtered_top);
    *cloud_full_top += *cloud_t;
  }
  for (int ind = 0; ind < size; ind++){
    Eigen::Isometry3d pose_l = odom2isometry(odoms_left[ind]);
    Eigen::Matrix4f pose_left = pose_l.matrix().cast <float>();
    sensor_msgs::PointCloud2 cloud_msg_left;
    pcl_ros::transformPointCloud(pose_left, *clouds_left[ind], cloud_msg_left);
    pcl::PointCloud<PointT>::Ptr cloud_l(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(cloud_msg_left, *cloud_l);
    // filtered_left = distance_filter(cloud_l);
    // filtered_left = downsample(filtered_left);
    // filtered_left = outlier_removal(filtered_left);    
    *cloud_full_left += *cloud_l;

    Eigen::Isometry3d pose_r = odom2isometry(odoms_right[ind]);
    Eigen::Matrix4f pose_right = pose_r.matrix().cast <float>();
    sensor_msgs::PointCloud2 cloud_msg_right;
    pcl_ros::transformPointCloud(pose_right, *clouds_right[ind], cloud_msg_right);
    pcl::PointCloud<PointT>::Ptr cloud_r(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(cloud_msg_right, *cloud_r);
    // filtered_right = distance_filter(cloud_r);
    // filtered_right = downsample(filtered_right);
    // filtered_right = outlier_removal(filtered_right);   
    *cloud_full_right += *cloud_r;
  }

  // filtered_top = downsample(cloud_full_top);
  // filtered_right = downsample(cloud_full_right);
  // filtered_left = downsample(cloud_full_left);    
  pcl::io::savePCDFileBinary("/home/nithilan/catkin_ws/src/rosbag_processor/dense_map_top.pcd", *cloud_full_top);
  pcl::io::savePCDFileBinary("/home/nithilan/catkin_ws/src/rosbag_processor/dense_map_left.pcd", *cloud_full_left);
  pcl::io::savePCDFileBinary("/home/nithilan/catkin_ws/src/rosbag_processor/dense_map_right.pcd", *cloud_full_right);     
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "rosbag_processor_pandar");

  ros::NodeHandle private_node_handle("~");

  std::string file_path;
  std::string output_path_;
  std::string s3_path_;
  int frame_rate_;
  int skip_first_;

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

  private_node_handle.param<std::string>("file", file_path, "");
  ROS_INFO("[%s] file_path: %s", ros::this_node::getName().c_str(), file_path.c_str());

  if (!file_exists(file_path))
  {
    ROS_ERROR("[%s] file_path: %s does not exist. Terminating.", ros::this_node::getName().c_str(), file_path.c_str());
    return 1;
  }

  auto start_time = std::chrono::system_clock::now();
  std::cout << "Starting at: " << get_string_time(start_time) << std::endl;

  rosbag::Bag input_bag;

  std::cout << "Reading Input Rosbag: " << file_path << std::endl;
  input_bag.open(file_path, rosbag::bagmode::Read);

  std::cout << "Reading messages..." << std::endl;
  rosbag::View rosbag_view(input_bag);
  size_t messages = 0;
  size_t total_messages = rosbag_view.size();
  std::cout << "Reading " << total_messages << " messages..." << std::endl << std::endl;

  for(rosbag::MessageInstance const m: rosbag_view)
  {
    float progress = (float) messages / (float) total_messages * 100.;
    if(m.getTopic() == "/odom_new_left")
    {

      nav_msgs::Odometry::ConstPtr cur = m.instantiate<nav_msgs::Odometry>();
      if (cur != NULL)
      {
        odom_msgs_top.push_back(cur);
      }
    }

    if(m.getTopic() == "/odom_new_left")
    {

      nav_msgs::Odometry::ConstPtr cur = m.instantiate<nav_msgs::Odometry>();
      if (cur != NULL)
      {
        odom_msgs_left.push_back(cur);
      }
    }

    if(m.getTopic() == "/odom_new_right")
    {

      nav_msgs::Odometry::ConstPtr cur = m.instantiate<nav_msgs::Odometry>();
      if (cur != NULL)
      {
        odom_msgs_right.push_back(cur);
      }
    }

    if(m.getTopic() == "/sensing/lidar/left_upper/pandar_points_ex")
    {
      sensor_msgs::PointCloud2::ConstPtr cur = m.instantiate<sensor_msgs::PointCloud2>();
      if (cur != NULL)
      {
        cloud_msgs_top.push_back(cur);
      }
    }

    if(m.getTopic() == "/sensing/lidar/left_upper/pandar_points_ex")
    {
      sensor_msgs::PointCloud2::ConstPtr cur = m.instantiate<sensor_msgs::PointCloud2>();
      if (cur != NULL)
      {
        cloud_msgs_left.push_back(cur);
      }
    }

    if(m.getTopic() == "/sensing/lidar/right_upper/pandar_points_ex")
    {
      sensor_msgs::PointCloud2::ConstPtr cur = m.instantiate<sensor_msgs::PointCloud2>();
      if (cur != NULL)
      {
        cloud_msgs_right.push_back(cur);
      }
    }

    if (messages % 100 == 0)
      std::cout << "\rProgress: (" << messages << " / " << total_messages << ") " << progress << "%            ";
    messages++;
  }
  std::cout << std::endl;
  input_bag.close();
  std::cout << "No of odom msgs: " << odom_msgs_top.size() << std::endl;
  std::cout << "No of velodyne top cloud msgs: " << cloud_msgs_top.size() << std::endl;

  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end_time - start_time;
  std::cout << std::endl << std::endl << "Finished at: " << get_string_time(end_time) << std::endl
            << "Total: " << elapsed_seconds.count() << " seconds" << std::endl;

  process_pointclouds(cloud_msgs_top, cloud_msgs_left, cloud_msgs_right, odom_msgs_top, odom_msgs_left, odom_msgs_right);

  // pcl::visualization::PCLVisualizer::Ptr viewer_final (new pcl::visualization::PCLVisualizer ("3D Viewer pointcloud"));
  // viewer_final->setBackgroundColor (255, 255, 255);
  // // Coloring and visualizing target cloud (green). (0, 255, 0)
  // pcl::visualization::PointCloudColorHandlerCustom<PointT>
  // target_color (cloud_full, 0, 0, 205);
  // viewer_final->addPointCloud<PointT> (cloud_full, target_color, "cloud full");
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
