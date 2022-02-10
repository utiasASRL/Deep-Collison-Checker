#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <ignition/math/Vector3.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Box.hh>
#include <ignition/math/Quaternion.hh>
#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include <ctype.h>
#include "costmap.hh"
#include "frame.hh"
#include <algorithm>

move_base_msgs::MoveBaseGoal PoseToGoal(ignition::math::Pose3d pose);
move_base_msgs::MoveBaseGoal PoseToHardcodedGoal(ignition::math::Pose3d pose, double x_target, double y_target);


class TourParser{

    private:

        std::string tour_path;
        
        bool parse_digits;

        std::string tour_name;

        std::vector<ignition::math::Pose3d> route;

        std::vector<ignition::math::Pose3d> route_digits;

        ignition::math::Box bounds = ignition::math::Box(ignition::math::Vector3d(-21.55, -21.4,0), ignition::math::Vector3d(21.55, 21.4,0));

        double resolution = 0.1; 

        void ReadTourParams();

        void ParseTour();

    public:

        TourParser(std::string name, bool parse_digits=false);

        std::vector<ignition::math::Pose3d> GetRoute();

        std::vector<ignition::math::Pose3d> GetDigitsCoordinates();
};

