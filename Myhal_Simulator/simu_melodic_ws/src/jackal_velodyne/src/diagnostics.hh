#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <geometry_msgs/PoseStamped.h>
#include <actionlib/client/simple_action_client.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/Image.h"
#include "nav_msgs/Odometry.h"
#include "std_msgs/Bool.h"
#include <utility>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include <queue> 
#include <vector>
#include <map>

struct Stamp{
    ignition::math::Vector3d vel;
    ignition::math::Pose3d pose;
    double time;
    Stamp(ignition::math::Vector3d vel, ignition::math::Pose3d pose, double time): vel(vel), pose(pose), time(time){};
};

class Doctor{

    private:

        ros::NodeHandle nh;

        ros::Subscriber sub;

        ros::Publisher shutdown_pub;

        ros::Publisher failure_pub;

        std::string username;

        std::string filepath;


        std::string start_time = "ERROR SETTING START TIME";

        ignition::math::Pose3d last_pose;

        ignition::math::Vector3d lin_vel;

        double duration = 5;

        std::queue<Stamp> snapshots; // stores last this->duration seconds of pose 

        double running_sum = 0;

        double last_update;

        double last_status_update =0;

        std::vector<double> dists;

        std::ofstream log_file;

        std::string tour_name = "NONE";

        std::vector<std::map<std::string, std::string>> rooms;

        void GroundTruthCallback(const nav_msgs::Odometry::ConstPtr& msg);

        void ReadParams();

    public:

        Doctor();


};

Doctor::Doctor(){

    this->ReadParams();


    this->filepath = "/home/" + this->username + "/Myhal_Simulation/simulated_runs/" + start_time + "/";

    this->log_file.open(this->filepath + "/logs-"+start_time+"/log.txt", std::ios_base::app);

    ROS_INFO_STREAM("\nJACKAL DIAGNOSTICS RUNNING. OUTPUT CAN BE FOUND AT: " << this->filepath << "/logs-"+start_time+"/log.txt");

    this->log_file << "Tour Name: " << this->tour_name << std::endl;
    
    this->shutdown_pub = this->nh.advertise<std_msgs::Bool>("shutdown_signal", 1000);
    this->sub = this->nh.subscribe<nav_msgs::Odometry>("ground_truth/state", 1000, std::bind(&Doctor::GroundTruthCallback, this, std::placeholders::_1), ros::VoidConstPtr(), ros::TransportHints().tcpNoDelay(true));

    ros::spin();

}


void Doctor::GroundTruthCallback(const nav_msgs::Odometry::ConstPtr& msg){

    double time = msg->header.stamp.toSec();

    double dt = time-this->last_update;
    this->last_update = time;
    
    double x = msg->pose.pose.position.x;
	double y = msg->pose.pose.position.y;
	double z = msg->pose.pose.position.z;
	
	double qx = msg->pose.pose.orientation.x;
	double qy = msg->pose.pose.orientation.y;
	double qz= msg->pose.pose.orientation.z;
	double qw= msg->pose.pose.orientation.w;

    auto curr_pose =  ignition::math::Pose3d(x,y,z,qw,qx,qy,qz);
    this->lin_vel = (curr_pose.Pos()-this->last_pose.Pos())/dt;
    Stamp new_stamp = Stamp(lin_vel, curr_pose, time);
    if (this->lin_vel.Length() < 10e3){
        this->snapshots.push(new_stamp);
        this->running_sum+=lin_vel.Length();
    }
    
    if (time-this->snapshots.front().time > this->duration){
        this->running_sum-=this->snapshots.front().vel.Length();
        this->snapshots.pop();
    }
    this->last_pose = curr_pose;
    //std::cout << "hi\n";
    if ((time-this->last_status_update) >= this->duration){
        auto dist = (this->snapshots.front().pose.Pos()-curr_pose.Pos()).Length();

        double speed = this->running_sum/this->snapshots.size();
        ROS_INFO("\nCurrent robot pos: (%.1fm, %.1fm, %.1fm)\nDisplacement over last %.1fs: %.1fm\n%.1fs average speed: %.1fm/s\n\n",curr_pose.Pos().X(), curr_pose.Pos().Y(), curr_pose.Pos().Z(), this->duration, dist, this->duration, speed);
        
        this->last_status_update = time;

        int num_checks = 6;
     
        if (this->dists.size() == num_checks){
            this->dists.erase(this->dists.begin(), this->dists.begin()+1);
        }
        this->dists.push_back(dist);    

        double sum =0;
        for (double d: this->dists){
            sum+=d;
        }
        
        if (sum < 2 && sum > 1 && this->dists.size() == num_checks){
            
            ROS_WARN("\nROBOT MAY BE STUCK\n");
        }

        if (sum < 1 && this->dists.size() == num_checks){
            ROS_WARN("\nROBOT STUCK, STOPPING TOUR\n");

            this->log_file << "Tour failed: robot got stuck\n";

            this->log_file.close();
            std_msgs::Bool shutdown_msg;
            shutdown_msg.data = false; // because the robot was unsuccessful
            this->shutdown_pub.publish(shutdown_msg);
        }
       
        
    }

}

void Doctor::ReadParams(){

    this->username = "default";
    if (const char * user = std::getenv("USER")){
        this->username = user;
    } 

    if (!this->nh.getParam("start_time", this->start_time)){
        std::cout << "ERROR SETTING START TIME\n";
    }

    if (!this->nh.getParam("tour_name", this->tour_name)){
        std::cout << "ERROR FINDING TOUR NAME\n";
    }

    std::vector<std::string> room_names;
    if (!this->nh.getParam("room_names", room_names)){
        std::cout << "ERROR READING ROOM NAMES";
    }

    for (auto name: room_names){
        
        std::map<std::string, std::string> info;
        info["name"] = name;
        if (!nh.getParam(name, info)){
            std::cout << "ERROR READING ROOM PARAMS";
        }

        this->rooms.push_back(info);
        
    }

    
}
