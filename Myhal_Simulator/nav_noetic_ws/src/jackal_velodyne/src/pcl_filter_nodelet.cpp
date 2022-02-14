
#include <limits>
#include <pluginlib/class_list_macros.h>
#include "pcl_filter_nodelet.h"
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <string>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <cstdlib>
#include <iostream>

namespace jackal_velodyne
{
PCLFilterNodelet::PCLFilterNodelet()
{
}

void PCLFilterNodelet::onInit()
{
  boost::mutex::scoped_lock lock(connect_mutex_);
  private_nh_ = getPrivateNodeHandle();

  private_nh_.param<std::string>("target_frame", target_frame_, "");
  private_nh_.param<double>("transform_tolerance", tolerance_, 0.01);
  private_nh_.param<double>("min_height", min_height_, std::numeric_limits<double>::min());
  private_nh_.param<double>("max_height", max_height_, std::numeric_limits<double>::max());

  if (!private_nh_.getParam("/filter_status", this->filter_status)){
    this->filter_status = true;
  }

  private_nh_.param<double>("angle_min", angle_min_, -M_PI);
  private_nh_.param<double>("angle_max", angle_max_, M_PI);
  private_nh_.param<double>("angle_increment", angle_increment_, M_PI / 180.0);
  private_nh_.param<double>("scan_time", scan_time_, 1.0 / 30.0);
  private_nh_.param<double>("range_min", range_min_, 0.0);
  private_nh_.param<double>("range_max", range_max_, std::numeric_limits<double>::max());
  private_nh_.param<double>("inf_epsilon", inf_epsilon_, 1.0);

  int concurrency_level;
  private_nh_.param<int>("concurrency_level", concurrency_level, 1);
  private_nh_.param<bool>("use_inf", use_inf_, true);
  private_nh_.param<std::vector<int>>("catagories", catagories, {0,1,2,3,4,5,6});
  // Check if explicitly single threaded, otherwise, let nodelet manager dictate thread pool size
  if (concurrency_level == 1)
  {
    nh_ = getNodeHandle();
  }
  else
  {
    nh_ = getMTNodeHandle();
  }

  // Only queue one pointcloud per running thread
  if (concurrency_level > 0)
  {
    input_queue_size_ = concurrency_level;
  }
  else
  {
    input_queue_size_ = boost::thread::hardware_concurrency();
  }

  // if pointcloud target frame specified, we need to filter by transform availability
  if (!target_frame_.empty())
  {
    tf2_.reset(new tf2_ros::Buffer());
    tf2_listener_.reset(new tf2_ros::TransformListener(*tf2_));
    message_filter_.reset(new MessageFilter(sub_, *tf2_, target_frame_, input_queue_size_, nh_));
    message_filter_->registerCallback(boost::bind(&PCLFilterNodelet::cloudCb, this, _1));
    message_filter_->registerFailureCallback(boost::bind(&PCLFilterNodelet::failureCb, this, _1, _2));
  }
  else  // otherwise setup direct subscription
  {
    sub_.registerCallback(boost::bind(&PCLFilterNodelet::cloudCb, this, _1));
  }

  pub_ = nh_.advertise<sensor_msgs::LaserScan>("scan", 10, boost::bind(&PCLFilterNodelet::connectCb, this),
                                               boost::bind(&PCLFilterNodelet::disconnectCb, this));
}

void PCLFilterNodelet::connectCb()
{
  boost::mutex::scoped_lock lock(connect_mutex_);
  if (pub_.getNumSubscribers() > 0 && sub_.getSubscriber().getNumPublishers() == 0)
  {
    NODELET_INFO("Got a subscriber to scan, starting subscriber to pointcloud");
    sub_.subscribe(nh_, "cloud_in", input_queue_size_);
  }
}

void PCLFilterNodelet::disconnectCb()
{
  boost::mutex::scoped_lock lock(connect_mutex_);
  if (pub_.getNumSubscribers() == 0)
  {
    NODELET_INFO("No subscibers to scan, shutting down subscriber to pointcloud");
    sub_.unsubscribe();
  }
}

void PCLFilterNodelet::failureCb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                                             tf2_ros::filter_failure_reasons::FilterFailureReason reason)
{
  NODELET_WARN_STREAM_THROTTLE(1.0, "Can't transform pointcloud from frame " << cloud_msg->header.frame_id << " to "
                                                                             << message_filter_->getTargetFramesString()
                                                                             << " at time " << cloud_msg->header.stamp
                                                                             << ", reason: " << reason);
}

void PCLFilterNodelet::cloudCb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
  // build laserscan output
  sensor_msgs::LaserScan output;
  output.header = cloud_msg->header;
  if (!target_frame_.empty())
  {
    output.header.frame_id = target_frame_;
  }

  output.angle_min = angle_min_;
  output.angle_max = angle_max_;
  output.angle_increment = angle_increment_;
  output.time_increment = 0.0;
  output.scan_time = scan_time_;
  output.range_min = range_min_;
  output.range_max = range_max_;

  // determine amount of rays to create
  uint32_t ranges_size = std::ceil((output.angle_max - output.angle_min) / output.angle_increment);

  // determine if laserscan rays with no obstacle data will evaluate to infinity or max_range
  if (use_inf_)
  {
    output.ranges.assign(ranges_size, std::numeric_limits<double>::infinity());
  }
  else
  {
    output.ranges.assign(ranges_size, output.range_max + inf_epsilon_);
  }

  sensor_msgs::PointCloud2ConstPtr cloud_out;
  sensor_msgs::PointCloud2Ptr cloud;

  // Transform cloud if necessary
  if (!(output.header.frame_id == cloud_msg->header.frame_id))
  {
    try
    {
      cloud.reset(new sensor_msgs::PointCloud2);
      tf2_->transform(*cloud_msg, *cloud, target_frame_, ros::Duration(tolerance_));
      cloud_out = cloud;
    }
    catch (tf2::TransformException& ex)
    {
      NODELET_ERROR_STREAM("Transform failure: " << ex.what());
      return;
    }
  }
  else
  {
    cloud_out = cloud_msg;
  }

  // Iterate through pointcloud
  
   sensor_msgs::PointCloud2ConstIterator<float> iter_i(*cloud_out, "intensity");
  for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud_out, "x"), iter_y(*cloud_out, "y"), iter_z(*cloud_out, "z");
      iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_i)
  { 

    if (this->filter_status && std::find(catagories.begin(), catagories.end(), *iter_i) == catagories.end()){
      continue;
    }

    if (std::isnan(*iter_x) || std::isnan(*iter_y) || std::isnan(*iter_z))
    {
      NODELET_DEBUG("rejected for nan in point(%f, %f, %f)\n", *iter_x, *iter_y, *iter_z);
      continue;
    }

    if (*iter_z > max_height_ || *iter_z < min_height_)
    {
      NODELET_DEBUG("rejected for height %f not in range (%f, %f)\n", *iter_z, min_height_, max_height_);
      continue;
    }

    double range = hypot(*iter_x, *iter_y);
    if (range < range_min_)
    {
      NODELET_DEBUG("rejected for range %f below minimum value %f. Point: (%f, %f, %f)", range, range_min_, *iter_x,
                    *iter_y, *iter_z);
      continue;
    }
    if (range > range_max_)
    {
      NODELET_DEBUG("rejected for range %f above maximum value %f. Point: (%f, %f, %f)", range, range_max_, *iter_x,
                    *iter_y, *iter_z);
      continue;
    }

    double angle = atan2(*iter_y, *iter_x);
    if (angle < output.angle_min || angle > output.angle_max)
    {
      NODELET_DEBUG("rejected for angle %f not in range (%f, %f)\n", angle, output.angle_min, output.angle_max);
      continue;
    }

    // overwrite range at laserscan ray if new range is smaller
    int index = (angle - output.angle_min) / output.angle_increment;
    if (range < output.ranges[index])
    {
      output.ranges[index] = range;
    }
  }
  pub_.publish(output);
}
}  // namespace jackal_velodyne

PLUGINLIB_EXPORT_CLASS(jackal_velodyne::PCLFilterNodelet, nodelet::Nodelet)
