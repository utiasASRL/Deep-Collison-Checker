#pragma once

// Use the same source code for CPU and GPU plugins
#ifndef GAZEBO_GPU_RAY
#define GAZEBO_GPU_RAY 0
#endif

// Custom Callback Queue
#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <ros/advertise_options.h>

#include <sdf/Param.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/TransportTypes.hh>
#include <gazebo/msgs/MessageTypes.hh>

#include <gazebo/common/Time.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/sensors/SensorTypes.hh>
#if GAZEBO_GPU_RAY
#include <gazebo/plugins/GpuRayPlugin.hh>
#else
#include <gazebo/plugins/RayPlugin.hh>
#endif

#include <boost/algorithm/string/trim.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_guard.hpp>
#include "utilities.hh"
#include "quadtree.hh"
#include <vector>
#include <std_srvs/Empty.h>
#include "gazebo/msgs/msgs.hh"
#include <map>

#if GAZEBO_GPU_RAY
#define GazeboRosVelodyneLaser GazeboRosVelodyneGpuLaser
#define RayPlugin GpuRayPlugin
#define RaySensorPtr GpuRaySensorPtr
#endif

struct CollObj{
	int cat;
	ignition::math::Box box;


	CollObj(int cat, ignition::math::Box box){
		this->cat = cat;
		this->box = box;
	}
};

namespace gazebo
{

	class GazeboRosVelodyneLaser : public RayPlugin
	{
		/// \brief Constructor
		/// \param parent The parent entity, must be a Model or a Sensor
	public:
		GazeboRosVelodyneLaser();

		/// \brief Destructor
	
		~GazeboRosVelodyneLaser();

		/// \brief Load the plugin
		/// \param take in SDF root element
	
		void Load(sensors::SensorPtr _parent, sdf::ElementPtr _sdf);

		/// \brief Subscribe on-demand
	private:
		void ConnectCb();

		/// \brief The parent ray sensor
	
		sensors::RaySensorPtr parent_ray_sensor_;

		/// \brief Pointer to ROS node
	
		ros::NodeHandle *nh_;

		/// \brief ROS publisher
	
		ros::Publisher pub_;

		/// \brief topic name
	
		std::string topic_name_;

		/// \brief frame transform name, should match link name
	
		std::string frame_name_;

		/// \brief the intensity beneath which points will be filtered
	
		double min_intensity_;

		/// \brief Minimum range to publish
	
		double min_range_;

		/// \brief Maximum range to publish
	
		double max_range_;

		/// \brief Gaussian noise
	
		double gaussian_noise_;

		/// \brief Gaussian noise generator
	
		static double gaussianKernel(double mu, double sigma)
		{
			// using Box-Muller transform to generate two independent standard normally distributed normal variables
			// see wikipedia
			double U = (double)rand() / (double)RAND_MAX; // normalized uniform random variable
			double V = (double)rand() / (double)RAND_MAX; // normalized uniform random variable
			return sigma * (sqrt(-2.0 * ::log(U)) * cos(2.0 * M_PI * V)) + mu;
		}

		/// \brief A mutex to lock access
	
		boost::mutex lock_;

		/// \brief For setting ROS name space
	
		std::string robot_namespace_;

		// Custom Callback Queue
	
		ros::CallbackQueue laser_queue_;

	
		void laserQueueThread();

	
		boost::thread callback_laser_queue_thread_;

		// Subscribe to gazebo laserscan
	
		gazebo::transport::NodePtr gazebo_node_;

		gazebo::transport::SubscriberPtr sub_;

		void OnScan(const ConstLaserScanStampedPtr &_msg);

		gazebo::physics::WorldPtr world;

		std::string robot_name = "jackal";

		std::string building_name = "building";

        gazebo::physics::ModelPtr robot = nullptr;

        gazebo::physics::EntityPtr building = nullptr; 

		boost::shared_ptr<QuadTree> static_quadtree; 

        boost::shared_ptr<QuadTree> active_quadtree; 

		ignition::math::Box building_box; 

		std::vector<gazebo::physics::EntityPtr> collision_entities;

		std::vector<gazebo::physics::ActorPtr> actors;

		std::vector<gazebo::physics::LinkPtr> robot_links;

		//ros::ServiceClient pauseGazebo;

        //ros::ServiceClient playGazebo;

        //std_srvs::Empty emptySrv;

		ros::NodeHandle nh;

		std::map<std::string, ignition::math::Pose3d> last_actor_pose;

		std::map<std::string, double> actor_speed;

		ros::Time last_update;

		void LoadWorld();
	};

} // namespace gazebo
