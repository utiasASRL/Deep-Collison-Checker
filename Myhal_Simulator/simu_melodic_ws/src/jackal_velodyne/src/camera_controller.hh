#pragma once

#include <gazebo/plugins/CameraPlugin.hh>
#include <gazebo/common/Time.hh>
#include <gazebo_plugins/gazebo_ros_camera_utils.h>
#include <gazebo/physics/physics.hh>

#include "gazebo/msgs/msgs.hh"
#include "gazebo/gazebo.hh"

#include <std_srvs/Empty.h>
#include <ros/ros.h>

#include <string>

#define EMPHGREEN "\033[1;4;32m"
#define EMPHRED "\033[1;4;31m"
#define EMPHBLUE "\033[1;4;34m"

void print_color(std::string text, std::string color, bool newline = true){
    std::cout << color << text << "\033[0m";
    if (newline){
        std::cout << std::endl;
    }
}

double rolling_avg(double avg, double new_sample, double n){
    return (avg*(n-1)/n) + new_sample/n;
}


namespace gazebo {

class CameraController: public CameraPlugin {


    private: 
            
        ros::NodeHandle nh;

        std::string filepath;

		ros::Time last_update;

		physics::WorldPtr world;

        int save_count = 0;

        ros::Time last_update_time;

        physics::PhysicsEnginePtr p_eng;        

        double avg_dt = 0;

        double fps;

        double min_step;

    protected: 
            
        virtual void OnNewFrame(const unsigned char *_image, unsigned int _width, unsigned int _height, unsigned int _depth, const std::string &_format);

    public:
            
        CameraController();

        void Load(sensors::SensorPtr _parent, sdf::ElementPtr _sdf);

};


}
