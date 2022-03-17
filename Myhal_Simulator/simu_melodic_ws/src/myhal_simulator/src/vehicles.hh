#pragma once

#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include <ignition/math/Box.hh>
#include "gazebo/physics/physics.hh"
#include "gazebo/common/common.hh"
#include "gazebo/gazebo.hh"
#include <nav_msgs/Path.h>
#include <vector>
#include <map>
#include <utility>
#include <string>
#include "utilities.hh"
#include "costmap.hh"
#include "flowfield.hh"
#include <ros/console.h>

#define ALI 0
#define COH 1
#define SEP 2

class PathViz
{

protected:
    std::vector<gazebo::physics::LinkPtr> dots;

    gazebo::physics::ModelPtr model = nullptr;

    gazebo::physics::WorldPtr world;

    int num_dots;

    std::string name;

    ignition::math::Vector4d color;

public:
    PathViz(std::string name, int num_dots, ignition::math::Vector4d color, gazebo::physics::WorldPtr world);

    void OnUpdate(const nav_msgs::Path::ConstPtr &plan);
};

class SmartCam
{

protected:
    gazebo::physics::ModelPtr self = nullptr;

    ignition::math::Pose3d target_pose; // what are we looking at right now

    ignition::math::Vector3d heading;

    ignition::math::Vector3d updated_pos;

public:
    SmartCam(gazebo::physics::ModelPtr self, ignition::math::Vector3d initial_pos);

    virtual void OnUpdate(double dt, std::vector<ignition::math::Vector3d> &robot_traj) = 0;

    void UpdateModel();
};

class Sentry : public SmartCam
{

public:
    using SmartCam::SmartCam;

    void OnUpdate(double dt, std::vector<ignition::math::Vector3d> &robot_traj);
};

class Hoverer : public SmartCam
{

protected:
    double T;

    ignition::math::Vector3d relative_pos;

public:
    Hoverer(gazebo::physics::ModelPtr self, ignition::math::Vector3d initial_pos, double T);

    void OnUpdate(double dt, std::vector<ignition::math::Vector3d> &robot_traj);
};

class Stalker : public SmartCam
{

protected:
    double dist;

    double curr_dist = 0;

    int curr_ind = 0;

public:
    Stalker(gazebo::physics::ModelPtr self, ignition::math::Vector3d initial_pos, double dist);

    void OnUpdate(double dt, std::vector<ignition::math::Vector3d> &robot_traj);
};

class Vehicle
{

protected:
    double mass;

    ignition::math::Pose3d pose;

    ignition::math::Vector3d velocity;

    ignition::math::Vector3d acceleration;

    gazebo::physics::ActorPtr actor;

    ignition::math::Vector3d curr_target;

    std::map<std::string, std::shared_ptr<gazebo::physics::TrajectoryInfo>> trajectories;

    double animation_factor = 5.1; //TODO: read these in as common vehicle parameters

    double obstacle_margin = 0.5;

    double actor_margin = 0.5;

    double slowing_distance = 1;

    double arrival_distance = 1; //0.5

    std::vector<gazebo::physics::EntityPtr> all_objects;

    bool still = false;

    void UpdateModel();

    void ApplyForce(ignition::math::Vector3d force);

    void Seek(ignition::math::Vector3d target, double weight = 1);

    void Arrival(ignition::math::Vector3d target, double weight = 1);

    void UpdatePosition(double dt);

    void UpdatePositionContactObstacles(std::vector<gazebo::physics::EntityPtr> &objects, double dt);

    void AvoidActors(std::vector<boost::shared_ptr<Vehicle>> vehicles);

    void AvoidObstacles(std::vector<gazebo::physics::EntityPtr> objects);

    void AvoidRobot(std::vector<gazebo::physics::EntityPtr> objects, double robot_margin_factor=1.0, bool slowing=false);

    double height = 1;

public:

    double slow_factor;
    double max_speed;

    double max_force;

    ignition::math::Vector3d showed_force;
    ignition::math::Vector3d flow_force;

    Vehicle(gazebo::physics::ActorPtr _actor,
            double _mass,
            double _max_force,
            double _max_speed,
            ignition::math::Pose3d initial_pose,
            ignition::math::Vector3d initial_velocity,
            std::vector<gazebo::physics::EntityPtr> all_objects,
            double _obstacle_margin,
            double _actor_margin);

    virtual void OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects);

    void OnPoseUpdate(const gazebo::common::UpdateInfo &_info,
                      double dt,
                      std::vector<gazebo::physics::EntityPtr> objects);
    void OnPoseReprod(const gazebo::common::UpdateInfo &_info,
                      double dt,
                      std::vector<gazebo::physics::EntityPtr> objects,
                      ignition::math::Pose3d new_pose0);

    gazebo::physics::ActorPtr GetActor();

    void SetAllObjects(std::vector<gazebo::physics::EntityPtr> objects);

    ignition::math::Pose3d GetPose();

    ignition::math::Vector3d GetVelocity();

    std::string GetName();

    bool IsStill();
};

class Wanderer : public Vehicle
{

protected:
    double curr_theta = 0;

    double rand_amp = 0.4;

    void SetNextTarget();

public:
    using Vehicle::Vehicle;

    void OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects);
};

class RandomWalker : public Vehicle
{

protected:
    void SetNextTarget(std::vector<gazebo::physics::EntityPtr> objects);

public:
    using Vehicle::Vehicle;

    void OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects);
};

class Boid : public Vehicle
{

protected:
    double weights[3];

    double FOV_angle = 4;

    double FOV_radius = 3;

    void Separation(std::vector<boost::shared_ptr<Vehicle>> vehicles);

    void Alignement(double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles);

    void Cohesion(std::vector<boost::shared_ptr<Vehicle>> vehicles);

public:
    Boid(gazebo::physics::ActorPtr _actor,
         double _mass,
         double _max_force,
         double _max_speed,
         ignition::math::Pose3d initial_pose,
         ignition::math::Vector3d initial_velocity,
         std::vector<gazebo::physics::EntityPtr> objects,
         double _alignement,
         double _cohesion,
         double _separation,
         double angle,
         double radius,
         double _obstacle_margin,
         double _actor_margin);

    void OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects);
};

class Stander : public Wanderer
{

protected:
    bool standing = true;

    bool never_walk = false;

    double standing_duration;

    double walking_duration;

    gazebo::common::Time standing_start;

    gazebo::common::Time walking_start;

    void UpdateModel(double dt);

public:
    Stander(gazebo::physics::ActorPtr _actor,
            double _mass,
            double _max_force,
            double _max_speed,
            ignition::math::Pose3d initial_pose,
            ignition::math::Vector3d initial_velocity,
            std::vector<gazebo::physics::EntityPtr> objects,
            double _standing_duration,
            double _walking_duration,
            double _obstacle_margin,
            double _actor_margin,
            int start_mode = 2); //2 for random, 1 for walking, 0 for standing

    void OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects);
};

class Sitter : public Vehicle
{

protected:
    std::string chair_name;

    gazebo::physics::EntityPtr chair;

    void UpdateModel(double dt);

public:
    Sitter(gazebo::physics::ActorPtr _actor, std::string chair_name, std::vector<gazebo::physics::EntityPtr> objects, double height = 0.65);

    void OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects);
};

class Follower : public Vehicle
{

protected:
    double rand_angle_offset;

    double rand_radius = ignition::math::Rand::DblUniform(2, 2.5);

    gazebo::physics::EntityPtr leader;

    std::string leader_name;

    ignition::math::Pose3d last_leader_pose;

    bool blocking;

    void SetNextTarget(double dt);

public:
    Follower(gazebo::physics::ActorPtr _actor,
             double _mass,
             double _max_force,
             double _max_speed,
             ignition::math::Pose3d initial_pose,
             ignition::math::Vector3d initial_velocity,
             std::vector<gazebo::physics::EntityPtr> objects,
             std::string _leader_name,
             bool blocking,
             double _obstacle_margin,
             double _actor_margin);

    void OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects);

    void LoadLeader(gazebo::physics::EntityPtr leader);
};

class PathFollower : public Wanderer
{

protected:
    void Follow();

    boost::shared_ptr<Costmap> costmap;

    boost::shared_ptr<std::vector<ignition::math::Pose3d>> digits_coordinates;

    std::vector<ignition::math::Vector3d> curr_path;

    int path_ind;

    void RePath();

public:
    PathFollower(gazebo::physics::ActorPtr _actor,
                 double _mass,
                 double _max_force,
                 double _max_speed,
                 ignition::math::Pose3d initial_pose,
                 ignition::math::Vector3d initial_velocity,
                 std::vector<gazebo::physics::EntityPtr> objects,
                 boost::shared_ptr<Costmap> costmap,
                 boost::shared_ptr<std::vector<ignition::math::Pose3d>> digits_coordinates,
                 double _obstacle_margin,
                 double _actor_margin);

    void OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects);
};

////////////////////////////////////////////////////////////////////////////////////////////////////////
class ExtendedSocialForce_Actor : public Vehicle
{

protected:
    void SetNextTarget(std::vector<gazebo::physics::EntityPtr> objects);
    void ExtendedSFHuman(std::vector<boost::shared_ptr<Vehicle>> vehicles);
    void ExtendedSFRobot(std::vector<gazebo::physics::EntityPtr> objects);
    void ExtendedSFObstacle(std::vector<gazebo::physics::EntityPtr> objects);

public:
    using Vehicle::Vehicle;
    void OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects);
};

class Custom_Wanderer : public Vehicle
{

protected:
    std::map<std::string, double> custom_actor_goal;
    std::vector<std::string> vehicle_names;

    void SetNextTarget(std::vector<boost::shared_ptr<Vehicle>> vehicles);

public:
    Custom_Wanderer(gazebo::physics::ActorPtr _actor,
                    double _mass,
                    double _max_force,
                    double _max_speed,
                    ignition::math::Pose3d initial_pose,
                    ignition::math::Vector3d initial_velocity,
                    std::vector<gazebo::physics::EntityPtr> objects,
                    std::map<std::string, double> _custom_actor_goal,
                    std::vector<std::string> _vehicle_names,
                    double _obstacle_margin,
                    double _actor_margin);

    void OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects);
};

class FlowFollower : public Vehicle
{

public:
    // Variables
    std::vector<boost::shared_ptr<FlowField>> flow_fields;
    int current_flow;
    double distance_to_goal;
    double robot_margin;
    double robot_slow;

    // Methods
    FlowFollower(gazebo::physics::ActorPtr _actor,
                 double _mass,
                 double _max_force,
                 double _max_speed,
                 ignition::math::Pose3d initial_pose,
                 ignition::math::Vector3d initial_velocity,
                 std::vector<gazebo::physics::EntityPtr> objects,
                 std::vector<boost::shared_ptr<FlowField>> &flow_fields0,
                 double _obstacle_margin,
                 double _actor_margin,
                 double _robot_margin,
                 double _robot_slow);

    void CheckGoal();
    void UpdateDistance();
    void FlowForce();
    void UpdatePositionContactObstacles(std::vector<gazebo::physics::EntityPtr> &objects, double dt);
    void OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects);
    void OnPoseUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<gazebo::physics::EntityPtr> objects);
};



class Bouncer : public Vehicle
{

public:

    // Methods
    Bouncer(gazebo::physics::ActorPtr _actor,
                 double _mass,
                 double _max_force,
                 double _max_speed,
                 ignition::math::Pose3d initial_pose,
                 ignition::math::Vector3d initial_velocity,
                 std::vector<gazebo::physics::EntityPtr> objects,
                 double _obstacle_margin,
                 double _actor_margin);

    void OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects);
    void PlaneBounce(const ignition::math::Vector3d& normal, double dt);
};



class Reproducer : public Vehicle
{

protected:
    ignition::math::Pose3d new_pose;

public:
    using Vehicle::Vehicle;

    void OnUpdate(const gazebo::common::UpdateInfo &_info,
                  double dt,
                  std::vector<boost::shared_ptr<Vehicle>> vehicles,
                  std::vector<gazebo::physics::EntityPtr> objects,
                  ignition::math::Pose3d new_pose0);
    void UpdatePosition(double dt);
};