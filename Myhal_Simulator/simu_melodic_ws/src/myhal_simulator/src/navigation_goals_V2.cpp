#include <actionlib/client/simple_action_client.h>
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/Image.h"
#include "parse_tour.hh"
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Int32.h>


typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

//int cycles = 0;
//int max_cycles = 10;

bool lidar = false;
bool camera = false;

void ImageCallback(const sensor_msgs::Image::ConstPtr& msg){
	
	if (!camera){
		if (!lidar){
			ROS_WARN("CAMERA RECEIVED, waiting for lidar\n");
		} else{
			ROS_WARN("CAMERA RECEIVED\n");
		}
		
	}
	camera = true;
}

void LidarCallback(const sensor_msgs::PointCloud2::ConstPtr& msg){

	// Point cloud are stored in msg
	//
	//  access header info like this msg->header.stamp
	//

	if (!lidar){
		if (!camera){
			ROS_WARN("LIDAR RECEIVED, waiting for camera\n");
		} else{
			ROS_WARN("LIDAR RECEIVED\n");
		}
		
	}
	lidar = true;
}


int main(int argc, char ** argv){

    // If a package is needed, add it into the cmake (find_package)
    // also add it in the xml file: build_depend build_export depend_exec_depend

    // IN CMAKELIST
    // To add additional cpp file use:
    // add_library

    // To add the main node file use:
    // add_executable
    

    ros::init(argc, argv, "simple_navigation_goals");
	ros::NodeHandle nh;
	ros::Publisher pub = nh.advertise<geometry_msgs::PoseStamped>("tour_data", 1000);
	ros::Publisher tot_pub = nh.advertise<std_msgs::Int32>("tour_length", 1000);
    ros::Publisher shutdown_pub = nh.advertise<std_msgs::Bool>("shutdown_signal", 1000);
	// Here we want to publish in another topicSubscribe to the topic "velodyne_points"

    std::string tour_name("test1");

    
    if (!nh.getParam("tour_name", tour_name)){
        std::cout << "ERROR READING TOUR NAME\n";
    }

    TourParser parser(tour_name);
    std::vector<ignition::math::Pose3d> route = parser.GetRoute();

    char temp[500];
    std::string current_path(getcwd(temp, sizeof(temp)));

    std::string start_time;
    if (!nh.getParam("start_time", start_time)){
        std::cout << "ERROR SETTING START TIME\n";
    }
    
    std::string filepath = current_path + "/../Data/Simulation_v2/simulated_runs/" + start_time + "/";
    
    std::ofstream log_file;
    log_file.open(filepath + "/logs-" + start_time + "/log.txt", std::ios_base::app);

    ROS_WARN("USING TOUR %s\n", tour_name.c_str());

	// Subscribe to the topic "velodyne_points".
	// Second number in a queue in which the messages are stored
	// Third argument is a function that is called on the message that the topic send

	// ros::spin() to block until receiving a new message (kind of while true loop)
	// Here the waiting is more advanced see ros doc

    ros::Subscriber lidar_sub = nh.subscribe("velodyne_points", 1000, LidarCallback);
    ros::Subscriber camera_sub = nh.subscribe("kinect_V2/depth/image_raw", 1000, ImageCallback); //nh.subscribe("no_gpu_points", 1000, LidarCallback);
    ros::Rate r(10);
    while (!lidar || !camera){ // wait until both lidar and camera have been recieved 
        
        ros::spinOnce();  
        r.sleep();
    }

    for (auto pose: route){
        geometry_msgs::PoseStamped msg;
		msg.pose.position.x = pose.Pos().X();
        msg.pose.position.y = pose.Pos().Y();
        msg.pose.position.z = pose.Pos().Z();
		pub.publish(msg);
    }

    std_msgs::Int32 msg;
    msg.data = (int) route.size();
    tot_pub.publish(msg);

    //tell the action client that we want to spin a thread by default
    MoveBaseClient ac("move_base", true);

    //wait for the action server to come up
    while(!ac.waitForServer(ros::Duration(5.0))){
        ROS_INFO("Waiting for the move_base action server to come up");
    }

    std::string filename;
    std::string shutdown_file = current_path + "/simu_melodic_ws/shutdown.sh";

    int count = 0;
    int use_custom_goal_bool(0); // 0 = False, other value = True
    if(!nh.getParam("/custom_robot_goal/use_custom_goal_bool", use_custom_goal_bool)){
        ROS_INFO("ERROR READING ROBOT CUSTOM PARAMS: use_custom_goal_bool");
    }

    log_file << "\nTour diagnostics: " << std::endl;
    if (use_custom_goal_bool){
        for (auto pose:route){
            ROS_WARN("SENDING CUSTOM GOAL TO THE ROBOT");
            double x_target(0), y_target(0);
            if(!nh.getParam("/custom_robot_goal/goal_x", x_target)){
                ROS_INFO("ERROR READING ROBOT GOAL_X");
            }
            if(!nh.getParam("/custom_robot_goal/goal_y", y_target)){
                ROS_INFO("ERROR READING ROBOT GOAL_Y");
            }
            ROS_WARN("Target -> (%f, %f, %f)", pose.Pos().X(), pose.Pos().Y(), pose.Pos().Z());
            auto goal = PoseToHardcodedGoal(pose, x_target, y_target);
            ac.sendGoal(goal);
            ac.waitForResult();
            if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED){
                ROS_INFO("Target Reached");
                
                log_file << "Reached target (" << count << "/" << route.size() << ") at position (" << std::fixed << std::setprecision(1) << pose.Pos().X() << "m, " << std::fixed << std::setprecision(1) << pose.Pos().Y()<< "m, " << std::fixed << std::setprecision(1) << pose.Pos().Z()<<  "m) at time: " << std::fixed << std::setprecision(1) << ros::Time::now().toSec() << "s" << std::endl;
            } else {
                ROS_INFO("Target Failed");
                log_file << "Failed to reach target (" << count << "/" << route.size() << ") at position (" << std::fixed << std::setprecision(1) << pose.Pos().X() << "m, " << std::fixed << std::setprecision(1) << pose.Pos().Y()<< "m, " << std::fixed << std::setprecision(1) << pose.Pos().Z()<<  "m) at time: " << std::fixed << std::setprecision(1) << ros::Time::now().toSec() << "s" << std::endl;
            }
        }
    }

    else{
        for (auto pose: route){
            count++;
            ROS_WARN("Sending Command (%d/%ld):", count, (long) route.size());
            ROS_WARN("Target -> (%f, %f, %f)", pose.Pos().X(), pose.Pos().Y(), pose.Pos().Z());

            auto goal = PoseToGoal(pose);

            ac.sendGoal(goal);
            ac.waitForResult();

            if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED){
                ROS_INFO("Target Reached");
                
                log_file << "Reached target (" << count << "/" << route.size() << ") at position (" << std::fixed << std::setprecision(1) << pose.Pos().X() << "m, " << std::fixed << std::setprecision(1) << pose.Pos().Y()<< "m, " << std::fixed << std::setprecision(1) << pose.Pos().Z()<<  "m) at time: " << std::fixed << std::setprecision(1) << ros::Time::now().toSec() << "s" << std::endl;
            } else {
                ROS_INFO("Target Failed");
                log_file << "Failed to reach target (" << count << "/" << route.size() << ") at position (" << std::fixed << std::setprecision(1) << pose.Pos().X() << "m, " << std::fixed << std::setprecision(1) << pose.Pos().Y()<< "m, " << std::fixed << std::setprecision(1) << pose.Pos().Z()<<  "m) at time: " << std::fixed << std::setprecision(1) << ros::Time::now().toSec() << "s" << std::endl;
            }
        }  
 
    }

    
    std_msgs::Bool shutdown_msg;
    shutdown_msg.data = true;
    shutdown_pub.publish(shutdown_msg);
    
    //const char *cstr = shutdown_file.c_str();
    //system(cstr);
    

    
    return 0;
}
