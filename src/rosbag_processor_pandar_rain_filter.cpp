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
//#include <pcl/filters/statistical_outlier_removal.h>

struct Range_point
{
  float distance;
  float intensity;
  int8_t return_type;
  float azimuth;
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

int count = 0;
// fill points ring by ring, also add blank points when lidar points are skipped in a ring
std::vector<Range_point> fill_points(int num, std::vector<Range_point> ring_pts, PointXYZIRADT pt_c){
  //add missing points as blank points
  for (int i = 0; i < num ; i++){
    Range_point pt;
    count += 1;
    pt.distance = -1; //we set distance as -1 for blank points
    pt.intensity = -1;
    pt.return_type = -1;    
    pt.azimuth = -1;
    ROS_ERROR("Blank point: %f, count: %d!!", pt.azimuth, count);
    ring_pts.push_back(pt);
  }
  return ring_pts;
}

void process_pointclouds(std::vector<sensor_msgs::PointCloud2::ConstPtr> &clouds_top){

  pcl::PointCloud<PointT>::Ptr cloud_no_rain_orig (new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr cloud_t_xyz (new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointXYZIRADT>::Ptr cloud_t_orig(new pcl::PointCloud<PointXYZIRADT>);
  pcl::io::loadPCDFile<PointT> ("/home/nithilan/catkin_ws/src/pandar_rain_filter/no_rain_1_frame.pcd", *cloud_no_rain_orig);

  int ind = 0;
  pcl::fromROSMsg(*clouds_top[ind], *cloud_t_orig);
  pcl::fromROSMsg(*clouds_top[ind], *cloud_t_xyz);
  
  std::vector<std::vector<Range_point>> ring_ids_first(40) ; 
  std::vector<std::vector<Range_point>> ring_ids_last(40) ;
	for (int ring_id = 0; ring_id < 40 ; ring_id++)//
	{
    for (int i = 0; i < (*cloud_t_orig).size(); i++) {
      //std::cout << "Azimuth: " << cloud_t_orig->points[i].azimuth << " Ret_type:" << static_cast<int16_t> (cloud_t_orig->points[i].return_type) << std::endl;
      PointXYZIRADT pt_c = cloud_t_orig->points[i];
      if (pt_c.ring == ring_id){
        Range_point pt;
        pt.distance = pt_c.distance;
        pt.intensity = pt_c.intensity;
        pt.return_type = pt_c.return_type;    
        pt.azimuth = pt_c.azimuth;
        if (pt.return_type == 6){ //add first & last ranges
          if(ring_ids_first[ring_id].empty()){ //No points stored in the ring yet.
            // Check if any points are skipped in the beginning add blank or no points are skipped store the point #TODO
            ring_ids_first[ring_id].push_back(pt);
            ring_ids_last[ring_id].push_back(pt);
            count += 1;
            ROS_ERROR("First & Last point: %f, count: %d, ret_type: %d, ring_id: %d !!", pt.azimuth, count, static_cast<int16_t>(pt.return_type), ring_id);
          }
          else{ //ring row has already some points, check and add blank points, then add current point
            int diff_azi = abs(ring_ids_first[ring_id].back().azimuth - pt_c.azimuth);
            if (diff_azi == 0){
              ROS_ERROR("Error: This can't happen!!");
              return;
            }
            else{
              int no_missing_pts = missing_pts_counter(diff_azi);
              ROS_ERROR("missing point first+last: %d!!", no_missing_pts);
                if (no_missing_pts == 0){ //just add the next point
                  ring_ids_first[ring_id].push_back(pt);
                  ring_ids_last[ring_id].push_back(pt);
                  count += 1;
                  ROS_ERROR("Next point first+last: %f, count: %d, ret_type: %d, ring_id: %d !!", pt.azimuth, count, static_cast<int16_t>(pt.return_type), ring_id);
                }
                else{ // add null points to both first and last ranges
                  ring_ids_first[ring_id] = fill_points(no_missing_pts, ring_ids_first[ring_id], pt_c);
                  ring_ids_last[ring_id] = fill_points(no_missing_pts, ring_ids_last[ring_id], pt_c);
                  ring_ids_first[ring_id].push_back(pt);
                  ring_ids_last[ring_id].push_back(pt);
                  count += 1;
                  ROS_ERROR("Next point first+last: %f, count: %d, ret_type: %d, ring_id: %d !!", pt.azimuth, count, static_cast<int16_t>(pt.return_type), ring_id);
                }
            }
          }
        }        
        if (pt.return_type == 2 || pt.return_type == 4){ //only first ranges (2,4), add last ranges (3,5)
          if(ring_ids_first[ring_id].empty()){ //No points stored in the ring yet.
            // Check if any points are skipped in the beginning add blank or no points are skipped store the point #TODO
            ring_ids_first[ring_id].push_back(pt);
            count += 1;
            if (pt.return_type == 2){
              pt.return_type = 5;
              ring_ids_last[ring_id].push_back(pt);
              count += 1;   
            }   
            if (pt.return_type == 4){
              pt.return_type = 3;
              ring_ids_last[ring_id].push_back(pt);
              count += 1;   
            }                      
            ROS_ERROR("First point first+last: %f, count: %d, ret_type: %d, ring_id: %d !!", pt.azimuth, count, static_cast<int16_t>(pt.return_type), ring_id);
          }
          else{ //ring row has already some points, check and add blank points, then add current point
            int diff_azi = abs(ring_ids_first[ring_id].back().azimuth - pt_c.azimuth);
            if (diff_azi == 0){
              ROS_ERROR("Error: This can't happen!!");
              return;
            }
            else{
              int no_missing_pts = missing_pts_counter(diff_azi);
              ROS_ERROR("missing point first: %d!!", no_missing_pts);
              if (no_missing_pts == 0){ //just add the next point
                ring_ids_first[ring_id].push_back(pt);
                count += 1;
              if (pt.return_type == 2){
                pt.return_type = 5;
                ring_ids_last[ring_id].push_back(pt);
                count += 1;   
              }   
              if (pt.return_type == 4){
                pt.return_type = 3;
                ring_ids_last[ring_id].push_back(pt);
                count += 1;   
              }                     
                ROS_ERROR("Next point first+last: %f, count: %d, ret_type: %d, ring_id: %d !!", pt.azimuth, count, static_cast<int16_t>(pt.return_type), ring_id);
              }
              else{ // add null points to both first and last ranges
                ring_ids_first[ring_id] = fill_points(no_missing_pts, ring_ids_first[ring_id], pt_c);
                ring_ids_last[ring_id] = fill_points(no_missing_pts, ring_ids_last[ring_id], pt_c);
                ring_ids_first[ring_id].push_back(pt);
                count += 1;
                if (pt.return_type == 2){
                  pt.return_type = 5;
                  ring_ids_last[ring_id].push_back(pt);
                  count += 1;   
                }   
                if (pt.return_type == 4){
                  pt.return_type = 3;
                  ring_ids_last[ring_id].push_back(pt);
                  count += 1;   
                }                    
                ROS_ERROR("Next point first+last: %f, count: %d, ret_type: %d, ring_id: %d !!", pt.azimuth, count, static_cast<int16_t>(pt.return_type), ring_id);
              }
            }
          }
        }
      }
    }
	}
  ROS_ERROR("Ring length: %d", ring_ids_first[39].size());
  ROS_ERROR("Ring length: %d", ring_ids_last[35].size());  

  pcl::PointCloud<PointT>::ConstPtr cloud_t = cloud_t_xyz;//downsample(cloud_t_orig);
  pcl::PointCloud<PointT>::ConstPtr cloud_no_rain = cloud_no_rain_orig;//downsample(cloud_no_rain_orig);

	// Segment the ground and remove all points below ground

	pcl::ModelCoefficients::Ptr plane (new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices);
	pcl::PointCloud<PointT>::Ptr cloud_plane (new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr cloud_inliers (new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr cloud_no_rain_ngnd (new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr cloud_top_ngnd (new pcl::PointCloud<PointT>);

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
	for (int i = 0; i < (*cloud_t).size(); i++) //
	{
    //std::cout << "ring num: " << cloud_t->points[i].intensity << std::endl;
		if (cloud_t->points[i].z < (minPt.z + 1.35)) // e.g. remove all pts below zAvg
		{
			inliers1->indices.push_back(i);
		}
	}
	extract2.setInputCloud(cloud_t);
	extract2.setIndices(inliers1);

	// Extract outliers
	extract2.setNegative (true);				// Extract the outliers
	extract2.filter (*cloud_top_ngnd);		// original cloud without ground points and below points


  pcl::PointCloud<PointT>::Ptr out (new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr out2 (new pcl::PointCloud<PointT>);
  //Segment differences
  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
  pcl::SegmentDifferences<PointT> sdiff;
  sdiff.setInputCloud(cloud_top_ngnd);
  sdiff.setTargetCloud(cloud_no_rain_ngnd);
  sdiff.setSearchMethod(tree);
  sdiff.setDistanceThreshold(0);
  sdiff.segment(*out);
  
  // Get point cloud difference
  pcl::getPointCloudDifference<PointT> (*cloud_top_ngnd,*cloud_no_rain_ngnd,0.1f,tree,*out2);
  //std::cout << *out2; 
  std::cout << *cloud_t; 
  std::cout << *cloud_no_rain; 
  //Visualiztion
  //Color handlers for red, green, blue and yellow color
  pcl::visualization::PointCloudColorHandlerCustom<PointT> red(cloud_no_rain,255,0,0);
  pcl::visualization::PointCloudColorHandlerCustom<PointT> blue(cloud_t,0,0,255);
  pcl::visualization::PointCloudColorHandlerCustom<PointT> green(out,0,255,0);
  pcl::visualization::PointCloudColorHandlerCustom<PointT> yellow(out2,255,255,0);    
  pcl::visualization::PCLVisualizer vis("3D View");
  //vis.addPointCloud(cloud_no_rain,red,"src",0);
  vis.addPointCloud(cloud_t,blue,"tgt",0);
  //vis.addPointCloud(out,green,"out",0);
  vis.addPointCloud(out2,yellow,"out2",0);
  while(!vis.wasStopped())
  {
          vis.spinOnce();
  }  
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "rosbag_processor_pandar_rain_filter");

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

  private_node_handle.param<std::string>("file_path", file_path, "");
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
  std::cout << "No of odom msgs: " << odom_msgs_top.size() << std::endl;
  std::cout << "No of velodyne top cloud msgs: " << cloud_msgs_top.size() << std::endl;

  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end_time - start_time;
  std::cout << std::endl << std::endl << "Finished at: " << get_string_time(end_time) << std::endl
            << "Total: " << elapsed_seconds.count() << " seconds" << std::endl;

  process_pointclouds(cloud_msgs_top);

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
