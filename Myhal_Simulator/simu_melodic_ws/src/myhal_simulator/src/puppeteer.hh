#ifndef PUPPETEER_HH
#define PUPPETEER_HH


#include <cmath>
#include <string>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Quaternion.hh>
#include <ignition/math/Vector3.hh>
#include <ignition/math/Vector4.hh>
#include "gazebo/physics/physics.hh"
#include "gazebo/common/common.hh"
#include "gazebo/gazebo.hh"
#include <vector>
#include <queue>
#include <map>
#include <utility>
#include "quadtree.hh"
#include "flowfield.hh"
#include <string>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
#include <boost/thread.hpp>
#include "vehicles.hh"
#include <std_srvs/Empty.h>
#include "frame.hh"
#include "gazebo/msgs/msgs.hh"
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <std_msgs/String.h>
#include <move_base_msgs/MoveBaseActionGoal.h>
#include "sensor_msgs/PointCloud2.h"
#include "tf2_msgs/TFMessage.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/TransformStamped.h>
#include <chrono>
#include <thread>
#include <iomanip>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef boost::shared_ptr<SmartCam> SmartCamPtr;

class Puppeteer: public gazebo::WorldPlugin{

    private:

        boost::shared_ptr<PathViz> global_path_viz;

        boost::shared_ptr<PathViz> local_path_viz;

        gazebo::event::ConnectionPtr update_connection;

        gazebo::physics::WorldPtr world;

        sdf::ElementPtr sdf;

        std::vector<boost::shared_ptr<Vehicle>> vehicles;

        std::vector<std::string> vehicles_names;

        std::vector<gazebo::physics::EntityPtr> collision_entities;

        std::string building_name;

        std::string robot_name = "";

        gazebo::physics::ModelPtr robot = nullptr;

        std::vector<SmartCamPtr> cams;

        gazebo::physics::EntityPtr building; 

        double update_freq = 60;

        gazebo::common::Time last_update;
        gazebo::common::Time last_real;

        std::map<std::string, double> vehicle_params;

        std::map<std::string, double> custom_actor_goal;

        std::map<std::string, double> boid_params;
        
        boost::shared_ptr<QuadTree> static_quadtree; 

        boost::shared_ptr<QuadTree> vehicle_quadtree; 

        boost::shared_ptr<QuadTree> vehicle_quadtree2; 

        ignition::math::Box building_box; 

        std::vector<boost::shared_ptr<Follower>> follower_queue;

        std::vector<gazebo::physics::LinkPtr> robot_links;

        ignition::math::Pose3d sensor_pose;

        std::string start_time, tour_name;

        boost::shared_ptr<Costmap> costmap;

        ros::NodeHandle nh;

        ros::Subscriber global_plan_sub;

        ros::Subscriber local_plan_sub;

        ros::Subscriber nav_goal_sub;

        ros::Subscriber tf_sub;

        ros::Publisher state_pub;
        ros::Publisher flow_pub;
        ros::Publisher flow_v_pub;
        
        std::vector<boost::shared_ptr<FlowField>> flow_fields;

        boost::shared_ptr<std::vector<ignition::math::Pose3d>> digits_coordinates;

        std::vector<std::vector<ignition::math::Vector3d>> paths;

        std::vector<ignition::math::Vector3d> robot_traj;

        nav_msgs::Path::ConstPtr global_plan;

        int old_global_ind = 0;

        int new_global_ind = 0;

        nav_msgs::Path::ConstPtr local_plan;

        int old_local_ind = 0;

        int new_local_ind = 0;

        move_base_msgs::MoveBaseActionGoal::ConstPtr nav_goal = nullptr;

        int old_nav_ind = 0;

        int new_nav_ind = 0;

        bool updating = false;

        ros::Publisher path_pub;
        
        bool filter_status;

        bool gt_class;

        int loc_method = 0;

        bool viz_gaz = false;

        geometry_msgs::Pose odom_to_base;

        geometry_msgs::TransformStamped map_to_odom;
        
        bool added_est = false;
        
        bool added_forces = false;

        gazebo::physics::ModelPtr pose_est = nullptr;
        
        std::vector<gazebo::physics::ModelPtr> showed_forces;

    public: 
        
        void Load(gazebo::physics::WorldPtr _world, sdf::ElementPtr _sdf);

        void OnUpdate(const gazebo::common::UpdateInfo &_info);

        void ReadSDF();

        void ReadParams();

        boost::shared_ptr<Vehicle> CreateVehicle(gazebo::physics::ActorPtr actor);

        SmartCamPtr CreateCamera(gazebo::physics::ModelPtr model);

        void TFCallback(const tf2_msgs::TFMessage::ConstPtr& msg);

        void GlobalPlanCallback(const nav_msgs::Path::ConstPtr& path);

        void LocalPlanCallback(const nav_msgs::Path::ConstPtr& path);

        void NavGoalCallback(const move_base_msgs::MoveBaseActionGoal::ConstPtr& goal);

        void ManagePoseEstimate(geometry_msgs::Pose est_pose);

        void AddPathMarkers(std::string name, const nav_msgs::Path::ConstPtr& plan, ignition::math::Vector4d color);

        //void AddFlowFieldMarker(std::string name, boost::shared_ptr<Costmap> costmap, ignition::math::Vector4d color);

        void AddGoalMarker(std::string name, const move_base_msgs::MoveBaseActionGoal::ConstPtr& goal, ignition::math::Vector4d color);
        
        void PublishPuppetState();

        void PublishFlowMarkers();

        void PublishIntegrationValue();

        void ShowFlowForces();
};


#endif
