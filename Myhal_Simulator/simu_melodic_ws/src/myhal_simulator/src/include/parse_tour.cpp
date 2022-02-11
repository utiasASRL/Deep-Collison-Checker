#include "parse_tour.hh"

move_base_msgs::MoveBaseGoal PoseToGoal(ignition::math::Pose3d pose){
    move_base_msgs::MoveBaseGoal goal;
    goal.target_pose.header.frame_id = "map";
    goal.target_pose.header.stamp = ros::Time::now();
    goal.target_pose.pose.position.x = pose.Pos().X();
    goal.target_pose.pose.position.y = pose.Pos().Y();
    goal.target_pose.pose.position.z = pose.Pos().Z();
    goal.target_pose.pose.orientation.w = pose.Rot().W();
    goal.target_pose.pose.orientation.x = pose.Rot().X();
    goal.target_pose.pose.orientation.y = pose.Rot().Y();
    goal.target_pose.pose.orientation.z = pose.Rot().Z();
    return goal;
}

/*
* PoseToHardcodedGoal send a parametrized goal to the robot (x_target, y_target) to create simple scenario for testing purposes.
*/
move_base_msgs::MoveBaseGoal PoseToHardcodedGoal(ignition::math::Pose3d pose, double x_target, double y_target){

    move_base_msgs::MoveBaseGoal goal;
    goal.target_pose.header.frame_id = "map";
    goal.target_pose.header.stamp = ros::Time::now();
    goal.target_pose.pose.position.x = x_target;
    goal.target_pose.pose.position.y = y_target;
    goal.target_pose.pose.position.z = 0;
    goal.target_pose.pose.orientation.w = pose.Rot().W();
    goal.target_pose.pose.orientation.x = pose.Rot().X();
    goal.target_pose.pose.orientation.y = pose.Rot().Y();
    goal.target_pose.pose.orientation.z = pose.Rot().Z();
    return goal;
}

/*
* TourParser constructor. 
* @param name Tour name.
* @param parse_digits  Optional. Default value = false.
*/
TourParser::TourParser(std::string name, bool _parse_digits){

    std::string home_path = "/home/admin";
    if (const char *home_path0 = std::getenv("HOME"))
        home_path = home_path0;


    this->tour_name = name;
    this->tour_path = home_path + "/Deep-Collison-Checker/Myhal_Simulator/simu_melodic_ws/src/myhal_simulator/tours/" + name + "/";
    this->parse_digits = _parse_digits;
    this->ReadTourParams();
    this->ParseTour();
}

void TourParser::ReadTourParams(){
    int argc = 0;
    char **argv = NULL;
    ros::init(argc, argv, "TourParser");
    ros::NodeHandle nh;

    if (!nh.getParam("resolution", this->resolution)){
        std::cout << "ERROR READING RESOLUTION";
        this->resolution = 0.1;
    }

    std::vector<double> b;
    if (!nh.getParam("bounds", b)){
        b = {-21.55, -21.4, 21.55, 21.4};
        std::cout << "ERROR READING TOP LEFT CORNER";
    }

    this->bounds = ignition::math::Box(ignition::math::Vector3d(b[0], b[1],0), ignition::math::Vector3d(b[2], b[3],0));
}

void TourParser::ParseTour()
{
    /////////////////
    // Tour points //
    /////////////////

    // Load tour file
    std::ifstream tour_file(this->tour_path + "map.txt");
    std::string line;
    int row = 0;

    // Create a costmap for the lookup function
    Costmap map = Costmap(this->bounds, this->resolution);
    
    // Result container
    std::vector<TrajPoint> traj;

    // Get Tour trajectory points
    while (std::getline(tour_file, line)){
        //std::cout << line.size() << std::endl;
        for (int col =0; col < line.size(); col++){
           
            if (std::isalpha(line[col])){
                int order = (int)line[col];
                ignition::math::Vector3d loc;
                map.IndiciesToPos(loc, row, col);
                ignition::math::Pose3d pose = ignition::math::Pose3d(loc, ignition::math::Quaterniond(0,0,0,1));
                traj.push_back(TrajPoint(pose, (double) order));
            }
            
        }

        row++;
    }

    // Close files
    tour_file.close();

    std::sort(traj.begin(), traj.end());

    for (auto point: traj){
        //std::cout << point.pose << " time: " << point.time << std::endl;
        this->route.push_back(point.pose);
    }

    /////////////////
    // Flow points //
    /////////////////
    
    if(this->parse_digits)
    {

        // Load tour file
        std::ifstream tour_flow_file(this->tour_path + "map_flow.txt");
        row = 0;

        // Result container
        std::vector<TrajPoint> traj_digits;

        // Get Tour trajectory points
        while (std::getline(tour_flow_file, line)){
            //std::cout << line.size() << std::endl;
            for (int col=0; col < line.size(); col++)
            {
                if (std::isdigit(line[col])){
                    int order = (int)line[col];
                    ignition::math::Vector3d loc;
                    map.IndiciesToPos(loc, row, col);
                    ignition::math::Pose3d pose = ignition::math::Pose3d(loc, ignition::math::Quaterniond(0,0,0,1));
                    traj_digits.push_back(TrajPoint(pose, (double) order));
                }
                
            }
            row++;
        }

        // Close files
        tour_flow_file.close();
        for (auto point: traj_digits){
            this->route_digits.push_back(point.pose);
        }
    }

}

/*
* GetRoute return the parsed route from alphanumerical characters (A-Z) in /worlds/map.txt.
*/
std::vector<ignition::math::Pose3d> TourParser::GetRoute(){
    return this->route;
}

/*
* GetDigitsCoordinates return the parsed coordinates from digit characters (1-9) in /worlds/map.txt. 
* Coordinates are intended to be passed as randomly selected objectives for path_followers actor type (and not as an explicit route from a given start to a given finish).
*/
std::vector<ignition::math::Pose3d> TourParser::GetDigitsCoordinates(){
    return this->route_digits;
}
