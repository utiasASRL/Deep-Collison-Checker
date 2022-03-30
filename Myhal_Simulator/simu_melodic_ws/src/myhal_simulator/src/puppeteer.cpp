#include "puppeteer.hh"
#include "utilities.hh"
#include "parse_tour.hh"
#include <ros/forwards.h>

GZ_REGISTER_WORLD_PLUGIN(Puppeteer);

//PUPPETEER CLASS

void Puppeteer::Load(gazebo::physics::WorldPtr _world, sdf::ElementPtr _sdf)
{
    //note: world name is default

    this->world = _world;
    this->sdf = _sdf;
    this->update_connection = gazebo::event::Events::ConnectWorldUpdateBegin(std::bind(&Puppeteer::OnUpdate, this, std::placeholders::_1));

    this->ReadSDF();

    int argc = 0;
    char **argv = NULL;
    ros::init(argc, argv, "Puppeteer");

    this->ReadParams();

    this->path_pub = this->nh.advertise<geometry_msgs::PoseStamped>("optimal_path", 1000);

    auto building = this->world->ModelByName(this->building_name);

    this->building_box = building->BoundingBox();
    this->building_box.Min().X() -= 1;
    this->building_box.Min().Y() -= 1;
    this->building_box.Max().X() += 1;
    this->building_box.Max().Y() += 1;
    this->static_quadtree = boost::make_shared<QuadTree>(this->building_box);
    this->vehicle_quadtree = boost::make_shared<QuadTree>(this->building_box);
    this->costmap = boost::make_shared<Costmap>(this->building_box, 0.1);
    this->digits_coordinates = boost::make_shared<std::vector<ignition::math::Pose3d>>();

    // Parse digit coordinates in ./worlds/map.txt for path_followers (parameter set in common_vehicle_params)
    if (this->vehicle_params["parse_digits"])
    {
        TourParser parser_path_follower = TourParser(this->tour_name, this->vehicle_params["parse_digits"]);
        std::vector<ignition::math::Pose3d> _digits_coordinates = parser_path_follower.GetDigitsCoordinates();
        for (auto v : _digits_coordinates)
        {
            this->digits_coordinates->push_back(v);
        }
    }

    // Init flow fields
    for (unsigned int i = 0; i < digits_coordinates->size(); ++i)
    {
        auto goal0 = digits_coordinates->at(i).Pos();
        auto flow_field_ptr = boost::make_shared<FlowField>(costmap,
                                                            goal0,
                                                            this->vehicle_params["flow_obstacle_range"],
                                                            this->vehicle_params["flow_obstacle_strength"]);
        flow_fields.push_back(flow_field_ptr);
    }

    // Create objects and costmap
    for (unsigned int i = 0; i < world->ModelCount(); ++i)
    {
        auto model = world->ModelByIndex(i);
        auto act = boost::dynamic_pointer_cast<gazebo::physics::Actor>(model);

        if (act)
        {
            auto new_vehicle = this->CreateVehicle(act);
            this->vehicles_names.push_back(new_vehicle->GetName());
            this->vehicles.push_back(new_vehicle);
            auto min = ignition::math::Vector3d(new_vehicle->GetPose().Pos().X() - 0.4, new_vehicle->GetPose().Pos().Y() - 0.4, 0);
            auto max = ignition::math::Vector3d(new_vehicle->GetPose().Pos().X() + 0.4, new_vehicle->GetPose().Pos().Y() + 0.4, 0);
            auto box = ignition::math::Box(min, max);
            auto new_node = QTData(box, new_vehicle, vehicle_type);
            this->vehicle_quadtree->Insert(new_node);
        }

        if (model->GetName().substr(0, 3) == "CAM")
        {
            this->cams.push_back(this->CreateCamera(model));
            continue;
        }

        if (model->GetName() != "ground_plane")
        {

            auto links = model->GetLinks();
            for (gazebo::physics::LinkPtr link : links)
            {
                std::vector<gazebo::physics::CollisionPtr> collision_boxes = link->GetCollisions();
                for (gazebo::physics::CollisionPtr collision_box : collision_boxes)
                {

                    this->collision_entities.push_back(collision_box);
                    auto box = collision_box->BoundingBox();

                    box.Max().Z() = 0;
                    box.Min().Z() = 0;

                    auto new_node = QTData(box, collision_box, entity_type);
                    this->static_quadtree->Insert(new_node);

                    if (collision_box->BoundingBox().Min().Z() < 1.5 && collision_box->BoundingBox().Max().Z() > 10e-2)
                    {
                        box.Min().X() -= 0.2;
                        box.Min().Y() -= 0.2;
                        box.Max().X() += 0.2;
                        box.Max().Y() += 0.2;
                        this->costmap->AddObject(box);
                    }
                }
            }
        }
    }
    
    // In case of reproduction, init variables
    bool reproducing = (this->load_world.size() > 0) && (this->load_world != "none");
    if (reproducing)
        parseSavedVehicleTraj();


    // Compute a flow fields
    // *********************
    
    if (!reproducing)
    {
        // Timing variables
        std::cout << "\n\n--------------------> Computing flow fields" << std::endl;
        std::vector<clock_t> t_debug;
        t_debug.push_back(std::clock());

        // Compute flow fields
        for (unsigned int i = 0; i < digits_coordinates->size(); ++i)
        {
            flow_fields[i]->Compute(costmap->costmap);
            auto goal0 = digits_coordinates->at(i).Pos();
            std::cout << "Goal: (" << goal0.X() << ", " << goal0.Y() << ")" << std::endl;
            std::cout << "Size: " << flow_fields[i]->rows << " * " << flow_fields[i]->cols << std::endl;
            std::cout << "Reachability: " << flow_fields[i]->Reachability() * 100 << "% " << std::endl;
            std::cout << "***********************" << std::endl;
        }

        // Debug timing
        t_debug.push_back(std::clock());
        double duration = 1000 * (t_debug[1] - t_debug[0]) / (double)CLOCKS_PER_SEC;
        std::cout << "--------------------> Done in " << duration << " ms\n\n" << std::endl;
    }

    // Parse robot tour from ./worlds/map.txt
    if (this->tour_name != "")
    {
        TourParser parser = TourParser(this->tour_name);
        auto route = parser.GetRoute();
        route.insert(route.begin(), ignition::math::Pose3d(ignition::math::Vector3d(0, 0, 0), ignition::math::Quaterniond(0, 0, 0, 1)));

        for (int i = 0; i < route.size() - 1; i++)
        {
            auto start = route[i];
            auto end = route[i + 1];
            std::vector<ignition::math::Vector3d> path;
            this->costmap->ThetaStar(start.Pos(), end.Pos(), path);
            this->paths.push_back(path);
        }
    }

    if (this->viz_gaz)
    {
        this->global_path_viz = boost::make_shared<PathViz>("global_path_viz", 300, ignition::math::Vector4d(0, 1, 0, 1), this->world);
        this->local_path_viz = boost::make_shared<PathViz>("local_path_viz", 75, ignition::math::Vector4d(0, 0, 1, 1), this->world);
    }

    std::cout << "LOADED ALL VEHICLES\n";

    if (this->viz_gaz)
    {
        this->global_plan_sub = this->nh.subscribe("/move_base/NavfnROS/plan", 1, &Puppeteer::GlobalPlanCallback, this);
        this->local_plan_sub = this->nh.subscribe("/move_base/TrajectoryPlannerROS/local_plan", 1, &Puppeteer::LocalPlanCallback, this);
        this->nav_goal_sub = this->nh.subscribe("/move_base/goal", 1, &Puppeteer::NavGoalCallback, this);
        this->tf_sub = this->nh.subscribe("/tf", 1, &Puppeteer::TFCallback, this);
    }

    // Create a publisher for the flow field
    flow_pub = nh.advertise<geometry_msgs::PoseArray>("flow_field", 5);
    flow_v_pub = nh.advertise<nav_msgs::OccupancyGrid>("value_field", 5);

    // Create ublisher for the state of puppeteer
    state_pub = nh.advertise<std_msgs::String>("puppet_state", 5);
}

void Puppeteer::OnUpdate(const gazebo::common::UpdateInfo &_info)
{

    ////////////////////////
    // Puppeteer processings
    ////////////////////////

    ProcessUpdate(_info);


    ////////////////////////////////
    // Ros/Real Time synchronisation
    ////////////////////////////////

    // real_time_factor: max_step_size x real_time_update_rate
    // max_step_size = 0.001
    // real_time_factor = 0.2
    // real_time_update_rate = real_time_factor / max_step_size = 200
    // real_time_update_dt = max_step_size / real_time_factor = 0.005

    double aim_real_dt = this->max_step_size / this->aim_real_time_factor;
    std::chrono::duration<double> aim_duration(aim_real_dt);

    std::chrono::system_clock::time_point next_time;
    if (this->time_count > 0)
    {

        // Get variables
        next_time = this->last_real_time + std::chrono::duration_cast<std::chrono::system_clock::duration>(aim_duration);

        // Warning if we can't keep up with pace
        std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - this->last_real_time;
        if (elapsed_seconds.count() > aim_real_dt)
        {
            ROS_WARN_STREAM("Gazebo Camera real_time_factor control can't keep up with desired rate"
                            << std::endl << "current_dt = " << elapsed_seconds.count() << " / desired_dt = " << aim_real_dt);
        }

        // Wait until reaching desired rate
        while (elapsed_seconds.count() < aim_real_dt)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            elapsed_seconds = std::chrono::system_clock::now() - this->last_real_time;
        }
    }
    else
        next_time = std::chrono::system_clock::now();


    // Update last time variables
    this->last_real_time = next_time;
    this->time_count++;

}


void Puppeteer::ProcessUpdate(const gazebo::common::UpdateInfo &_info)
{

    /////////////////////
    // Function frequency
    /////////////////////

    // Get the time elapsed since the last call to this function
    double dt = (_info.simTime - this->last_update).Double();
    double dt_real = (_info.realTime - this->last_real).Double();

    // Only update stuff with a certain frequency
    if (dt < 1 / this->update_freq)
    {
        return;
    }
    this->last_update = _info.simTime;
    this->last_real = _info.realTime;

    ///////////////////
    // Timing variables
    ///////////////////

    // Visualization boolean
    bool publish_flow = false;
    bool show_flow_forces = false;
    bool cout_speeds = false;

    // Timing variables
    bool verbose = false;
    std::vector<clock_t> t_debug;
    std::vector<std::string> clock_str;
    if (verbose)
    {
        std::cout << "real time elapsed = " << dt_real << " sim time = " << dt << std::endl;
        clock_str.push_back("Robot Init ......... ");
        clock_str.push_back("Cam handle ......... ");
        clock_str.push_back("Vehicles quadtree .. ");
        clock_str.push_back("Vehicles update .... ");
        clock_str.push_back("Path update ........ ");
        clock_str.push_back("Flow Markers ....... ");
        clock_str.push_back("Goal Marker ........ ");
        clock_str.push_back("Pose Marker ........ ");
    }
    t_debug.push_back(std::clock());

    ///////////////////////
    // Robot initialization
    ///////////////////////

    // This part is called only once at the beginning
    if ((this->tour_name != "") && (this->robot_name != "") && this->robot == nullptr)
    {
        // Init the Robot model
        for (unsigned int i = 0; i < world->ModelCount(); ++i)
        {
            auto model = world->ModelByIndex(i);
            if (model->GetName() == this->robot_name)
            {

                std::cout << "LOADING MODEL: " << model->GetName() << std::endl;
                std::cout << "PLUGINS: " << model->GetPluginCount() << std::endl;
                std::cout << "SENSORS: " << model->GetSensorCount() << std::endl;

                this->robot = model;
                for (auto vehicle : this->follower_queue)
                {
                    vehicle->LoadLeader(this->robot);
                }
                this->robot_links = this->robot->GetLinks();
                std::cout << "ADDED ROBOT: " << this->robot->GetName() << std::endl;
            }
        }

        // Publish the robot path
        double i = 0;
        for (auto path : this->paths)
        {
            for (auto pose : path)
            {
                geometry_msgs::PoseStamped msg;
                msg.pose.position.x = pose.X();
                msg.pose.position.y = pose.Y();
                msg.pose.position.z = pose.Z();
                msg.header.stamp = ros::Time(i);
                path_pub.publish(msg);
            }
            i++;
        }

        return;
    }

    /////////////////
    // Handle Cameras
    /////////////////

    t_debug.push_back(std::clock());

    if (this->robot != nullptr)
    {
        this->robot_traj.push_back(this->robot->WorldPose().Pos());
        this->robot_traj.back().Z() = 0;
        for (auto cam : this->cams)
        {
            cam->OnUpdate(dt, this->robot_traj);
        }
    }

    //////////////////
    // Handle vehicles
    //////////////////

    t_debug.push_back(std::clock());

    // Place vehicles in a quadtree
    this->vehicle_quadtree = boost::make_shared<QuadTree>(this->building_box);
    for (auto vehicle : this->vehicles)
    {
        auto min = ignition::math::Vector3d(vehicle->GetPose().Pos().X() - 0.4, vehicle->GetPose().Pos().Y() - 0.4, 0);
        auto max = ignition::math::Vector3d(vehicle->GetPose().Pos().X() + 0.4, vehicle->GetPose().Pos().Y() + 0.4, 0);
        auto box = ignition::math::Box(min, max);
        auto new_node = QTData(box, vehicle, vehicle_type);
        this->vehicle_quadtree->Insert(new_node);
    }

    t_debug.push_back(std::clock());

    // Calling update for each vehicle
    for (auto vehicle : this->vehicles)
    {

        // Init containers for  close by vehicles
        std::vector<boost::shared_ptr<Vehicle>> near_vehicles;
        std::vector<gazebo::physics::EntityPtr> near_objects;

        // Init query range
        auto vehicle_pos = vehicle->GetPose();
        double quad_range = vehicle_params["actor_margin"] + 0.5;
        auto min = ignition::math::Vector3d(vehicle_pos.Pos().X() - quad_range, vehicle_pos.Pos().Y() - quad_range, 0);
        auto max = ignition::math::Vector3d(vehicle_pos.Pos().X() + quad_range, vehicle_pos.Pos().Y() + quad_range, 0);
        auto query_range = ignition::math::Box(min, max);

        // Query objects in quadtree
        std::vector<QTData> query_objects = this->static_quadtree->QueryRange(query_range);
        for (auto n : query_objects)
        {
            if (n.type == entity_type)
            {
                near_objects.push_back(boost::static_pointer_cast<gazebo::physics::Entity>(n.data));
            }
        }
        if (this->robot != nullptr && (vehicle_pos.Pos() - this->robot->WorldPose().Pos()).Length() < quad_range)
        {
            near_objects.push_back(this->robot);
        }

        // Query vehicles in quadtree
        std::vector<QTData> query_vehicles = this->vehicle_quadtree->QueryRange(query_range);
        for (auto n : query_vehicles)
        {
            if (n.type == vehicle_type)
            {
                near_vehicles.push_back(boost::static_pointer_cast<Vehicle>(n.data));
            }
        }

        // Update the forces on vehicle
        if (this->vehicle_reprod_poses.size() < 1)
            vehicle->OnUpdate(_info, dt, near_vehicles, near_objects);
    }

    // Get current pose index if reprod
    size_t pose_i = getCurrentPoseInd(_info.simTime.Double());

    // Calling update pose for each vehicle
    std::vector<double> speeds;
    std::vector<ignition::math::Pose3d> vehicle_poses;
    size_t veh_i = 0;
    for (auto vehicle : this->vehicles)
    {
        // Init containers for  close by vehicles
        std::vector<gazebo::physics::EntityPtr> near_objects;

        // Init query range
        auto vehicle_pos = vehicle->GetPose();
        double quad_range = vehicle_params["actor_margin"] + 0.5;
        auto min = ignition::math::Vector3d(vehicle_pos.Pos().X() - quad_range, vehicle_pos.Pos().Y() - quad_range, 0);
        auto max = ignition::math::Vector3d(vehicle_pos.Pos().X() + quad_range, vehicle_pos.Pos().Y() + quad_range, 0);
        auto query_range = ignition::math::Box(min, max);

        // Query objects in quadtree
        std::vector<QTData> query_objects = this->static_quadtree->QueryRange(query_range);
        for (auto n : query_objects)
        {
            if (n.type == entity_type)
            {
                near_objects.push_back(boost::static_pointer_cast<gazebo::physics::Entity>(n.data));
            }
        }
        if (this->robot != nullptr && (vehicle_pos.Pos() - this->robot->WorldPose().Pos()).Length() < quad_range)
        {
            near_objects.push_back(this->robot);
        }

        // Update the vehicle
        if (this->vehicle_reprod_poses.size() > 0)
        {
            // Get the pose at this time 
            ignition::math::Pose3d next_pose = this->vehicle_reprod_poses[pose_i][veh_i];
            vehicle->OnPoseReprod(_info, dt, near_objects, next_pose);
        }
        else
            vehicle->OnPoseUpdate(_info, dt, near_objects);

        if (cout_speeds)
            speeds.push_back(vehicle->GetVelocity().Length());
        veh_i++;

        // save vehicle poses
        vehicle_poses.push_back(vehicle->GetPose());
    }

    // save vehicle poses
    saveVehicleTraj(_info.simTime.Double(), vehicle_poses);

    if (cout_speeds)
    {
        std::cout.precision(2);
        std::cout << std::fixed << " ------- Speeds: ";
        for (auto veh_speed : speeds)
        {
            std::cout << veh_speed << " ";
        }
        std::cout << std::endl;
    }

    ////////////////////
    // Handle Robot path
    ////////////////////

    t_debug.push_back(std::clock());

    if (this->old_global_ind != this->new_global_ind)
    {
        this->global_path_viz->OnUpdate(this->global_plan);
    }

    if (this->old_local_ind != this->new_local_ind)
    {
        this->local_path_viz->OnUpdate(this->local_plan);
    }

    ///////////////////
    // Add visu markers
    ///////////////////

    t_debug.push_back(std::clock());

    // Publish the flow
    if (publish_flow)
    {
        PublishFlowMarkers();
        PublishIntegrationValue();
    }
    
    PublishPuppetState();

    if (show_flow_forces)
    {
        ShowFlowForces();
    }

    t_debug.push_back(std::clock());

    if (this->old_nav_ind != this->new_nav_ind)
    {
        std::string name = "nav_goal";
        if (this->old_nav_ind == 0)
        {
            this->AddGoalMarker(name, this->nav_goal, ignition::math::Vector4d(1, 0, 0, 1));
        }
        else
        {
            auto goal = this->world->EntityByName(name);
            auto p = this->nav_goal->goal.target_pose.pose.position;
            auto pos = ignition::math::Vector3d(p.x, p.y, p.z);
            goal->SetWorldPose(ignition::math::Pose3d(pos, ignition::math::Quaterniond(0, 0, 0, 1)));
        }
        this->old_nav_ind = this->new_nav_ind;
    }

    t_debug.push_back(std::clock());

    if (this->viz_gaz)
    {
        geometry_msgs::Pose est_pose;
        tf2::doTransform<geometry_msgs::Pose>(this->odom_to_base, est_pose, this->map_to_odom);
        this->ManagePoseEstimate(est_pose);

        ros::spinOnce();
    }

    t_debug.push_back(std::clock());

    if (verbose)
    {
        for (size_t i = 0; i < std::min(t_debug.size() - 1, clock_str.size()); i++)
        {
            double duration = 1000 * (t_debug[i + 1] - t_debug[i]) / (double)CLOCKS_PER_SEC;
            std::cout << clock_str[i] << duration << " ms" << std::endl;
        }
        std::cout << std::endl
                  << "***********************" << std::endl
                  << std::endl;
    }
}





void Puppeteer::ReadSDF()
{

    if (this->sdf->HasElement("building_name"))
    {
        this->building_name = this->sdf->GetElement("building_name")->Get<std::string>();
    }

    if (this->sdf->HasElement("robot_name"))
    {
        this->robot_name = this->sdf->GetElement("robot_name")->Get<std::string>();
    }
    else
    {
        this->robot_name = "";
    }
}

boost::shared_ptr<Vehicle> Puppeteer::CreateVehicle(gazebo::physics::ActorPtr actor)
{

    boost::shared_ptr<Vehicle> res;
    auto sdf = actor->GetSDF();
    std::map<std::string, std::string> actor_info;
    auto attribute = sdf->GetElement("plugin");
    while (attribute)
    {
        actor_info[attribute->GetAttribute("name")->GetAsString()] = attribute->GetAttribute("filename")->GetAsString();
        attribute = attribute->GetNextElement("plugin");
    }

    double max_speed = this->vehicle_params["max_speed"];

    if (actor_info.find("max_speed") != actor_info.end())
    {
        try
        {
            max_speed = std::stod(actor_info["max_speed"]);
        }
        catch (std::exception &e)
        {
            std::cout << "Error converting max_speed argument to double" << std::endl;
        }
    }

    // Randomize the max speed
    if (this->vehicle_params["max_speed_std"] > 0)
        max_speed = ignition::math::Rand::DblNormal(max_speed, this->vehicle_params["max_speed_std"]);

    

    if (actor_info.find("vehicle_type") != actor_info.end())
    {
        if (actor_info["vehicle_type"] == "wanderer")
        {
            res = boost::make_shared<Wanderer>(actor,
                                               this->vehicle_params["mass"],
                                               this->vehicle_params["max_force"],
                                               max_speed,
                                               actor->WorldPose(),
                                               ignition::math::Vector3d(0, 0, 0),
                                               this->collision_entities,
                                               vehicle_params["obstacle_margin"],
                                               vehicle_params["actor_margin"]);
        }
        else if (actor_info["vehicle_type"] == "custom_wanderer")
        {
            res = boost::make_shared<Custom_Wanderer>(actor,
                                                      this->vehicle_params["mass"],
                                                      this->vehicle_params["max_force"],
                                                      max_speed, actor->WorldPose(),
                                                      ignition::math::Vector3d(0, 0, 0),
                                                      this->collision_entities,
                                                      this->custom_actor_goal,
                                                      this->vehicles_names,
                                                      vehicle_params["obstacle_margin"],
                                                      vehicle_params["actor_margin"]);
        }
        else if (actor_info["vehicle_type"] == "random_walker")
        {
            res = boost::make_shared<RandomWalker>(actor,
                                                   this->vehicle_params["mass"],
                                                   this->vehicle_params["max_force"],
                                                   max_speed, actor->WorldPose(),
                                                   ignition::math::Vector3d(0, 0, 0),
                                                   this->collision_entities,
                                                   vehicle_params["obstacle_margin"],
                                                   vehicle_params["actor_margin"]);
        }
        else if (actor_info["vehicle_type"] == "extendedSF_actor")
        {
            res = boost::make_shared<ExtendedSocialForce_Actor>(actor,
                                                                this->vehicle_params["mass"],
                                                                this->vehicle_params["max_force"],
                                                                max_speed, actor->WorldPose(),
                                                                ignition::math::Vector3d(0, 0, 0),
                                                                this->collision_entities,
                                                                vehicle_params["obstacle_margin"],
                                                                vehicle_params["actor_margin"]);
        }
        else if (actor_info["vehicle_type"] == "boid")
        {
            auto random_vel = ignition::math::Vector3d(ignition::math::Rand::DblUniform(-1, 1), ignition::math::Rand::DblUniform(-1, 1), 0);
            random_vel.Normalize();
            random_vel *= 2;
            res = boost::make_shared<Boid>(actor,
                                           this->vehicle_params["mass"],
                                           this->vehicle_params["max_force"],
                                           max_speed, actor->WorldPose(),
                                           random_vel,
                                           this->collision_entities,
                                           this->boid_params["alignement"],
                                           this->boid_params["cohesion"],
                                           this->boid_params["separation"],
                                           this->boid_params["FOV_angle"],
                                           this->boid_params["FOV_radius"],
                                           vehicle_params["obstacle_margin"],
                                           vehicle_params["actor_margin"]); // read in as params
        }
        else if (actor_info["vehicle_type"] == "stander")
        {

            double standing_duration = 5;
            double walking_duration = 5;

            if (actor_info.find("standing_duration") != actor_info.end())
            {
                try
                {
                    standing_duration = std::stod(actor_info["standing_duration"]);
                }
                catch (std::exception &e)
                {
                    std::cout << "Error converting standing duration argument to double" << std::endl;
                }
            }

            if (actor_info.find("walking_duration") != actor_info.end())
            {
                try
                {
                    walking_duration = std::stod(actor_info["walking_duration"]);
                }
                catch (std::exception &e)
                {
                    std::cout << "Error converting walking duration argument to double" << std::endl;
                }
            }

            res = boost::make_shared<Stander>(actor,
                                              1,
                                              10,
                                              max_speed,
                                              actor->WorldPose(),
                                              ignition::math::Vector3d(0, 0, 0),
                                              this->collision_entities,
                                              standing_duration,
                                              walking_duration,
                                              this->vehicle_params["start_mode"],
                                              vehicle_params["obstacle_margin"],
                                              vehicle_params["actor_margin"]);
        }
        else if (actor_info["vehicle_type"] == "sitter")
        {

            std::string chair = "";

            if (actor_info.find("chair") != actor_info.end())
            {
                chair = actor_info["chair"];
            }

            res = boost::make_shared<Sitter>(actor, chair, this->collision_entities, actor->WorldPose().Pos().Z());
        }
        else if (actor_info["vehicle_type"] == "follower")
        {

            std::string leader_name = "";

            if (actor_info.find("leader") != actor_info.end())
            {
                leader_name = actor_info["leader"];

                res = boost::make_shared<Follower>(actor,
                                                   1,
                                                   10,
                                                   max_speed,
                                                   actor->WorldPose(),
                                                   ignition::math::Vector3d(0, 0, 0),
                                                   this->collision_entities,
                                                   leader_name,
                                                   (bool)this->vehicle_params["blocking"],
                                                   vehicle_params["obstacle_margin"],
                                                   vehicle_params["actor_margin"]);
                this->follower_queue.push_back(boost::dynamic_pointer_cast<Follower>(res));
            }
            else
            {
                std::cout << "leader name not found\n";
            }
        }
        else if (actor_info["vehicle_type"] == "path_follower")
        {

            res = boost::make_shared<PathFollower>(actor,
                                                   this->vehicle_params["mass"],
                                                   this->vehicle_params["max_force"],
                                                   max_speed,
                                                   actor->WorldPose(),
                                                   ignition::math::Vector3d(0, 0, 0),
                                                   this->collision_entities,
                                                   this->costmap,
                                                   this->digits_coordinates,
                                                   vehicle_params["obstacle_margin"],
                                                   vehicle_params["actor_margin"]);
        }
        else if (actor_info["vehicle_type"] == "flow_follower")
        {

            res = boost::make_shared<FlowFollower>(actor,
                                                   this->vehicle_params["mass"],
                                                   this->vehicle_params["max_force"],
                                                   max_speed, actor->WorldPose(),
                                                   ignition::math::Vector3d(0, 0, 0),
                                                   collision_entities,
                                                   flow_fields,
                                                   vehicle_params["obstacle_margin"],
                                                   vehicle_params["actor_margin"],
                                                   vehicle_params["robot_margin"],
                                                   vehicle_params["robot_slow_flow"]);
        }
        else if (actor_info["vehicle_type"] == "bouncer")
        {
            
            ignition::math::Vector3d init_vel;
            if (this->vehicle_params["custom_theta"] > -0.5)
            {
                init_vel = ignition::math::Vector3d(cos(this->vehicle_params["custom_theta"]),
                                                         sin(this->vehicle_params["custom_theta"]),
                                                         0) * max_speed;
            }
            else
                init_vel = ignition::math::Vector3d(0, 0, 0);

            res = boost::make_shared<Bouncer>(actor,
                                              this->vehicle_params["mass"],
                                              this->vehicle_params["max_force"],
                                              max_speed, actor->WorldPose(),
                                              init_vel,
                                              collision_entities,
                                              vehicle_params["obstacle_margin"],
                                              vehicle_params["actor_margin"]);
        }
        else
        {
            std::cout << "INVALID VEHICLE TYPE\n";
        }
    }

    return res;
}

void Puppeteer::ReadParams()
{

    if (!nh.getParam("start_time", this->start_time))
    {
        std::cout << "ERROR READING START TIME: ANY VIDEOS WILL BE SAVED TO /tmp/\n";
        this->start_time = "";
    }

    if (!nh.getParam("common_vehicle_params", this->vehicle_params))
    {
        ROS_ERROR("ERROR READING COMMON VEHICLE PARAMS");
        vehicle_params["mass"] = 1;
        vehicle_params["max_force"] = 10;
        vehicle_params["max_speed"] = 0.77;
        vehicle_params["max_speed_std"] = 0.2;
        vehicle_params["custom_theta"] = -1;
        vehicle_params["slowing_distance"] = 2;
        vehicle_params["arrival_distance"] = 0.5;
        vehicle_params["obstacle_margin"] = 0.4;
        vehicle_params["actor_margin"] = 1.0;
        vehicle_params["blocking"] = 0;
        vehicle_params["start_mode"] = 2;
        vehicle_params["parse_digits"] = 0;
        vehicle_params["flow_obstacle_range"] = 1.0;
        vehicle_params["flow_obstacle_strength"] = 1.0;
        vehicle_params["robot_margin"] = 1.0;
        vehicle_params["robot_slow_flow"] = 0;
    }

    if (!nh.getParam("custom_actor_goal", this->custom_actor_goal))
    {
        ROS_ERROR("ERROR READING CUSTOM ACTOR GOAL PARAMS");
    }

    if (!nh.getParam("common_boid_params", this->boid_params))
    {
        ROS_ERROR("ERROR READING COMMON BOID PARAMS");
        boid_params["alignement"] = 0.1;
        boid_params["cohesion"] = 0.01;
        boid_params["separation"] = 2;
        boid_params["FOV_angle"] = 4;
        boid_params["FOV_radius"] = 3;
    }

    if (!nh.getParam("gt_class", this->gt_class))
    {
        std::cout << "ERROR READING gt_class\n";
        this->gt_class = false;
    }

    if (!nh.getParam("filter_status", this->filter_status))
    {
        std::cout << "ERROR READING filter_status\n";
        this->filter_status = false;
    }

    // if (!nh.getParam("loc_method", this->loc_method))
    // {
    //     std::cout << "ERROR READING loc_method\n";
    //     this->loc_method = 0;
    // }

    if (!nh.getParam("viz_gaz", this->viz_gaz))
    {
        std::cout << "ERROR READING viz_gaz\n";
        this->viz_gaz = false;
    }

    if (!nh.getParam("tour_name", this->tour_name))
    {
        std::cout << "ERROR READING TOUR NAME\n";
        this->tour_name = "";
        return;
    }

    if (!nh.getParam("load_path", this->load_path))
    {
        this->load_path = "";
        return;
    }

    if (!nh.getParam("load_world", this->load_world))
    {
        this->load_world = "";
        return;
    }
    
    if (!nh.getParam("real_time_factor", this->aim_real_time_factor))
    {
        this->aim_real_time_factor = 0.1;
    }
    
    if (!nh.getParam("/gazebo/time_step", this->max_step_size))
    {
        this->max_step_size = 0.001;
    }
}

SmartCamPtr Puppeteer::CreateCamera(gazebo::physics::ModelPtr model)
{
    auto tokens = utilities::split(model->GetName(), '_');
    SmartCamPtr new_cam;
    if (tokens[1] == "0")
    {
        // sentry
        new_cam = boost::make_shared<Sentry>(model, model->WorldPose().Pos());
    }
    else if (tokens[1] == "1")
    {
        // hoverer
        double T = std::stod(tokens[2]);
        new_cam = boost::make_shared<Hoverer>(model, model->WorldPose().Pos(), T);
    }
    else if (tokens[1] == "2")
    {
        // path follower
        double dist = std::stod(tokens[3]);
        new_cam = boost::make_shared<Stalker>(model, model->WorldPose().Pos(), dist);
    }
    return new_cam;
}

void Puppeteer::TFCallback(const tf2_msgs::TFMessage::ConstPtr &msg)
{

    for (auto transform : msg->transforms)
    {
        if (transform.header.frame_id == "odom" && transform.child_frame_id == "base_link")
        {
            //std::cout << "GOT ODOM" << std::endl;
            auto translation = transform.transform.translation;
            auto rotation = transform.transform.rotation;
            this->odom_to_base.position.x = translation.x;
            this->odom_to_base.position.y = translation.y;
            this->odom_to_base.position.z = translation.z;
            this->odom_to_base.orientation.x = rotation.x;
            this->odom_to_base.orientation.y = rotation.y;
            this->odom_to_base.orientation.z = rotation.z;
            this->odom_to_base.orientation.w = rotation.w;
            //this->odom_to_base = transform;
        }
        else if (transform.header.frame_id == "map" && transform.child_frame_id == "odom")
        {
            //std::cout << "GOT MAP" << std::endl;
            this->map_to_odom = transform;
        }
    }
}

void Puppeteer::GlobalPlanCallback(const nav_msgs::Path::ConstPtr &path)
{
    if (path->poses.size() > 0)
    {
        std::cout << utilities::color_text("Global plan recieved by simulator", BLUE) << std::endl;
        this->global_plan = path;
        this->new_global_ind++;
    }
}

void Puppeteer::LocalPlanCallback(const nav_msgs::Path::ConstPtr &path)
{
    if (path->poses.size() > 0)
    {
        std::cout << utilities::color_text("Local plan recieved by simulator", YELLOW) << std::endl;
        this->local_plan = path;
        this->new_local_ind++;
    }
}

void Puppeteer::NavGoalCallback(const move_base_msgs::MoveBaseActionGoal::ConstPtr &goal)
{
    this->nav_goal = goal;
    this->new_nav_ind++;
}

void Puppeteer::AddPathMarkers(std::string name, const nav_msgs::Path::ConstPtr &plan, ignition::math::Vector4d color)
{

    boost::shared_ptr<sdf::SDF> sdf = boost::make_shared<sdf::SDF>();
    sdf->SetFromString(
        "<sdf version ='1.6'>\
          <model name ='path'>\
          </model>\
        </sdf>");

    auto model = sdf->Root()->GetElement("model");
    model->GetElement("static")->Set(true);
    model->GetAttribute("name")->SetFromString(name);

    int i = 0;
    for (auto pose : plan->poses)
    {
        auto p = pose.pose.position;
        auto pos = ignition::math::Vector3d(p.x, p.y, p.z);
        auto link = model->AddElement("link");
        link->GetElement("pose")->Set(pos);
        link->GetAttribute("name")->SetFromString("l_" + name + std::to_string(i));
        auto cylinder = link->GetElement("visual")->GetElement("geometry")->GetElement("cylinder");
        cylinder->GetElement("radius")->Set(0.03);
        cylinder->GetElement("length")->Set(0.001);
        auto mat = link->GetElement("visual")->GetElement("material");
        mat->GetElement("ambient")->Set(color);
        mat->GetElement("diffuse")->Set(color);
        mat->GetElement("specular")->Set(color);
        mat->GetElement("emissive")->Set(color);
        i++;
    }

    this->world->InsertModelSDF(*sdf);
}

void Puppeteer::AddGoalMarker(std::string name, const move_base_msgs::MoveBaseActionGoal::ConstPtr &goal, ignition::math::Vector4d color)
{

    auto p = goal->goal.target_pose.pose.position;
    auto pos = ignition::math::Vector3d(p.x, p.y, p.z);

    boost::shared_ptr<sdf::SDF> sdf = boost::make_shared<sdf::SDF>();
    sdf->SetFromString(
        "<sdf version ='1.6'>\
          <model name ='path'>\
          </model>\
        </sdf>");

    auto model = sdf->Root()->GetElement("model");
    model->GetElement("static")->Set(true);
    model->GetAttribute("name")->SetFromString(name);
    model->GetElement("pose")->Set(pos);

    auto link1 = model->AddElement("link");
    link1->GetAttribute("name")->SetFromString("l_1" + name);
    auto box1 = link1->GetElement("visual")->GetElement("geometry")->GetElement("box");
    box1->GetElement("size")->Set(ignition::math::Vector3d(0.5, 0.05, 0.001));
    auto mat1 = link1->GetElement("visual")->GetElement("material");
    mat1->GetElement("ambient")->Set(color);
    mat1->GetElement("diffuse")->Set(color);
    mat1->GetElement("specular")->Set(color);
    mat1->GetElement("emissive")->Set(color);

    auto link2 = model->AddElement("link");
    link2->GetAttribute("name")->SetFromString("l_2" + name);
    auto box2 = link2->GetElement("visual")->GetElement("geometry")->GetElement("box");
    box2->GetElement("size")->Set(ignition::math::Vector3d(0.05, 0.5, 0.001));
    auto mat2 = link2->GetElement("visual")->GetElement("material");
    mat2->GetElement("ambient")->Set(color);
    mat2->GetElement("diffuse")->Set(color);
    mat2->GetElement("specular")->Set(color);
    mat2->GetElement("emissive")->Set(color);

    this->world->InsertModelSDF(*sdf);
}

void Puppeteer::ManagePoseEstimate(geometry_msgs::Pose est_pose)
{
    auto pos = est_pose.position;
    auto ori = est_pose.orientation;
    if (std::isnan(pos.x) || std::isnan(ori.x))
    {
        return;
    }

    auto pose = ignition::math::Pose3d(pos.x, pos.y, 0, ori.w, ori.x, ori.y, ori.z);
    if (!this->added_est)
    {

        boost::shared_ptr<sdf::SDF> sdf = boost::make_shared<sdf::SDF>();
        sdf->SetFromString(
            "<sdf version ='1.6'>\
              <model name ='path'>\
              </model>\
            </sdf>");

        auto model = sdf->Root()->GetElement("model");
        model->GetElement("static")->Set(true);
        model->GetAttribute("name")->SetFromString("pose_estimate");
        model->GetElement("pose")->Set(pose);

        auto color = ignition::math::Vector4d(0, 1, 1, 1);

        auto link1 = model->AddElement("link");
        link1->GetAttribute("name")->SetFromString("pose_link");
        link1->GetElement("pose")->Set(ignition::math::Vector3d(-0.05, 0, 0.001));
        auto box1 = link1->GetElement("visual")->GetElement("geometry")->GetElement("box");
        box1->GetElement("size")->Set(ignition::math::Vector3d(0.42, 0.43, 0.002));
        auto mat1 = link1->GetElement("visual")->GetElement("material");
        mat1->GetElement("ambient")->Set(color);
        mat1->GetElement("diffuse")->Set(color);
        mat1->GetElement("specular")->Set(color);
        mat1->GetElement("emissive")->Set(color);

        color = ignition::math::Vector4d(1, 0, 1, 1);

        auto link2 = model->AddElement("link");
        link2->GetAttribute("name")->SetFromString("front_link");
        link2->GetElement("pose")->Set(ignition::math::Vector3d(0.21, 0, 0.001));
        auto box2 = link2->GetElement("visual")->GetElement("geometry")->GetElement("box");
        box2->GetElement("size")->Set(ignition::math::Vector3d(0.1, 0.43, 0.002));
        auto mat2 = link2->GetElement("visual")->GetElement("material");
        mat2->GetElement("ambient")->Set(color);
        mat2->GetElement("diffuse")->Set(color);
        mat2->GetElement("specular")->Set(color);
        mat2->GetElement("emissive")->Set(color);

        this->world->InsertModelSDF(*sdf);
        std::cout << utilities::color_text("Added pose estimate", TEAL) << std::endl;

        this->added_est = true;
    }
    else
    {

        if (!this->pose_est)
        {
            this->pose_est = this->world->ModelByName("pose_estimate");
        }
        else
        {
            this->pose_est->SetWorldPose(pose);
        }
    }
}

// Function that publishes the puppeteer state (simple string)
void Puppeteer::PublishPuppetState()
{

    std_msgs::String str_msg;
    str_msg.data = "Puppeteer running";
    state_pub.publish(str_msg);
}


// Function that publishes the flow field as a PoseArray
void Puppeteer::PublishFlowMarkers()
{

    geometry_msgs::PoseArray flow_msg;

    //make sure to set the header information
    flow_msg.header.stamp = ros::Time::now();
    flow_msg.header.frame_id = "map";

    // Initiate poses
    int rows = flow_fields[0]->rows;
    int cols = flow_fields[0]->cols;
    flow_msg.poses = std::vector<geometry_msgs::Pose>();

    for (int i = 0; i < rows * cols; i++)
    {
        // get indices of column and row
        int c = i % cols;
        int r = i / cols;

        // Get position of this grid cell
        ignition::math::Vector3d pos;
        flow_fields[0]->IndiciesToPos(pos, r, c);

        // Get flow field at this grid cell
        ignition::math::Vector2d flow = flow_fields[0]->field[r][c];

        // Only add pose if valid flow
        if (flow.SquaredLength() > 0)
        {
            // Get flow orientation
            double angle = std::atan2(flow.Y(), flow.X());
            //angle += M_PI;

            // Convert to quaternion
            ignition::math::Vector3d axis(0.0, 0.0, 1.0);
            ignition::math::Quaternion<double> quat(axis, angle);

            // Create pose object
            geometry_msgs::Pose pose;
            pose.position.x = pos.X() + 0.5 * flow_fields[0]->resolution;
            pose.position.y = pos.Y() - 0.5 * flow_fields[0]->resolution;
            pose.position.z = pos.Z();
            pose.orientation.x = quat.X();
            pose.orientation.y = quat.Y();
            pose.orientation.z = quat.Z();
            pose.orientation.w = quat.W();
            flow_msg.poses.push_back(pose);
        }
    }

    // Publish message
    flow_pub.publish(flow_msg);
}

// Function that publishes the value function for flow field as a a Map
void Puppeteer::PublishIntegrationValue()
{

    // Init occuppancy grid
    nav_msgs::OccupancyGrid map;

    // Set the header information
    map.header.stamp = ros::Time::now();
    map.header.frame_id = "map";

    // Set the map information
    int rows = flow_fields[0]->rows;
    int cols = flow_fields[0]->cols;

    map.info.width = cols;
    map.info.height = rows;
    map.info.origin.position.x = flow_fields[0]->boundary.Min().X();
    map.info.origin.position.y = flow_fields[0]->boundary.Min().Y();
    map.info.origin.position.z = 0;
    map.info.origin.orientation.x = 0.0;
    map.info.origin.orientation.y = 0.0;
    map.info.origin.orientation.z = 0.0;
    map.info.origin.orientation.w = 1.0;
    map.info.resolution = flow_fields[0]->resolution;

    // Get max integration field elem
    double min_v = flow_fields[0]->value_function[0][0];
    double max_v = -1;
    for (auto &integration_row : flow_fields[0]->value_function)
    {
        for (auto &value : integration_row)
        {
            if (value < 10e8 && max_v < value)
                max_v = value;
            if (min_v > value)
                min_v = value;
        }
    }
    double factor = 98.0 / max_v;
    factor = 98.0 / 9.80;

    // Fill the ROS map object
    map.data = std::vector<int8_t>(map.info.width * map.info.height, 0);
    int i = 0;
    for (auto &pix : map.data)
    {
        // get indices of column and row
        int c = i % cols;
        int r = (rows - 1) - i / cols;

        // Fill pixel
        double value = flow_fields[0]->value_function[r][c];
        if (value < 10e8)
        {
            double int8_value = value * factor;
            if (int8_value > 98)
                int8_value = 98;
            pix = (int8_t)floor(int8_value);
            if (pix == 0)
                pix = 110;
        }
        else
            pix = 99;

        i++;
    }

    // Publish map and map metadata
    flow_v_pub.publish(map);
}

// Function that publishes the value function for flow field as a a Map
void Puppeteer::ShowFlowForces()
{

    if (!added_forces)
    {

        int vehicle_i = 0;
        for (auto vehicle : this->vehicles)
        {
            // Create SDF object
            boost::shared_ptr<sdf::SDF> sdf = boost::make_shared<sdf::SDF>();
            sdf->SetFromString(
                "<sdf version ='1.6'>\
                <model name ='actor_force'>\
                </model>\
                </sdf>");

            // Get force pose (origin at the vehicle and orientation like the force itself)
            auto origin_pos = vehicle->GetPose().Pos();
            double angle = std::atan2(vehicle->showed_force.Y(), vehicle->showed_force.X());
            ignition::math::Vector3d axis(0.0, 0.0, 1.0);
            ignition::math::Quaternion<double> quat(axis, angle);
            ignition::math::Pose3d force_pose(origin_pos, quat);

            // Add origin element (only position, ignore pose)
            auto model = sdf->Root()->GetElement("model");
            model->GetElement("static")->Set(true);
            model->GetAttribute("name")->SetFromString("force_origin_" + std::to_string(vehicle_i));
            model->GetElement("pose")->Set(force_pose);

            // Add link (visual force object)
            auto link1 = model->AddElement("link");
            link1->GetAttribute("name")->SetFromString("force_link");

            // Set its pose according to the force
            link1->GetElement("pose")->Set(ignition::math::Vector3d(0.5, 0, 0));

            // Set its size according to the force
            auto box1 = link1->GetElement("visual")->GetElement("geometry")->GetElement("box");
            box1->GetElement("size")->Set(ignition::math::Vector3d(1.0, 0.02, 0.02));

            // Set link color
            auto color = ignition::math::Vector4d(1, 0, 0, 1);
            auto mat1 = link1->GetElement("visual")->GetElement("material");
            mat1->GetElement("ambient")->Set(color);
            mat1->GetElement("diffuse")->Set(color);
            mat1->GetElement("specular")->Set(color);
            mat1->GetElement("emissive")->Set(color);

            this->world->InsertModelSDF(*sdf);
            vehicle_i++;
        }

        added_forces = true;
    }
    else
    {
        if (showed_forces.size() == 0)
        {
            for (auto vehicle : this->vehicles)
                showed_forces.push_back(nullptr);
        }
        else
        {
            int vehicle_i = 0;
            for (auto vehicle : this->vehicles)
            {

                if (!showed_forces[vehicle_i])
                {
                    showed_forces[vehicle_i] = this->world->ModelByName("force_origin_" + std::to_string(vehicle_i));
                }
                else
                {
                    // Get force pose (origin at the vehicle and orientation like the force itself)
                    auto origin_pos = vehicle->GetPose().Pos();
                    origin_pos.Z() += 0.5;
                    double angle = std::atan2(vehicle->showed_force.Y(), vehicle->showed_force.X());
                    ignition::math::Vector3d axis(0.0, 0.0, 1.0);

                    // Init quaternion with euler angles (pitch for force strength)
                    double clipped_f = vehicle->showed_force.Length() * 2 / vehicle->max_force;
                    clipped_f = std::min(clipped_f, 1.0);
                    double pitch = std::acos(clipped_f);
                    ignition::math::Quaternion<double> quat(0.0, -pitch, angle);
                    ignition::math::Pose3d force_pose(origin_pos, quat);

                    // Set this new pose
                    showed_forces[vehicle_i]->SetWorldPose(force_pose);
                }
                vehicle_i++;
            }
        }
    }
}

// Function that save vehicle traj
void Puppeteer::saveVehicleTraj(double time, 
                                std::vector<ignition::math::Pose3d> &vehicle_poses)
{
    
    std::string save_path = this->load_path + "/" + this->start_time + "/vehicles.txt";

    // Init line with time
    char buffer0[100];
    sprintf(buffer0, "%.3f ", time);
	std::string line0(buffer0);

    // Add rest of the data
    for (auto &pose: vehicle_poses)
    {
        char buffer1[100];
        sprintf(buffer1, "%.3f %.3f %.3f %.3f %.3f %.3f %.3f ",
                pose.Pos().X(),
                pose.Pos().Y(),
                pose.Pos().Z(),
                pose.Rot().W(),
                pose.Rot().X(),
                pose.Rot().Y(),
                pose.Rot().Z());
        line0 = line0 + std::string(buffer1);
    }

    // Open file
    std::ofstream outfile;
    outfile.open(save_path, std::ios_base::app);

    // Save
    outfile << line0 << std::endl;
    outfile.close();

    return;
}

// Function that Loads vehicle traj
void Puppeteer::parseSavedVehicleTraj()
{

    std::cout << "\n\n--------------------> Loading previous trajectories" << std::endl;
    std::vector<clock_t> t_debug;
    t_debug.push_back(std::clock());

    std::string saved_traj_path = this->load_path + "/" + this->load_world + "/vehicles.txt";

    // Load tour file
    std::ifstream saved_traj_file(saved_traj_path);
    size_t row = 0;

    // Saved file format:
    //      TIME    x1  y1  a1  x2  y2  a2  ...
    //      0.1     .   .   .   .   .   .
    //      0.2     .   .   .   .   .   .
    //      0.3     .   .   .   .   .   .
    //      ...

    // Get Tour trajectory points
    std::string line;
    while (std::getline(saved_traj_file, line))
    {
        std::stringstream ss(line);

        // get first info: time
        double time0;
        ss >> time0;
        this->vehicle_reprod_times.push_back(time0);
        this->vehicle_reprod_poses.push_back(std::vector<ignition::math::Pose3d>());

        // Now get get vehicles
        double x, y, z, ax, ay, az, aw;
        int count = 0;
        while (ss >> x) 
        {
            ss >> y;
            ss >> z;
            ss >> aw;
            ss >> ax;
            ss >> ay;
            ss >> az;
            this->vehicle_reprod_poses.back().push_back(ignition::math::Pose3d(x, y, z, aw, ax, ay, az));
            count++;
        }
    }

    // Close files
    saved_traj_file.close();

    // Debug timing
    t_debug.push_back(std::clock());
    double duration = 1000 * (t_debug[1] - t_debug[0]) / (double)CLOCKS_PER_SEC;
    std::cout << "--------------------> Done in " << duration << " ms\n\n" << std::endl;
    
    return;
}


// Function to get the current pose index if we reproduce traj
size_t Puppeteer::getCurrentPoseInd(double current_time)
{
    size_t pose_i = 0;
    if (this->vehicle_reprod_times.size() > 0)
    {
        for (auto saved_time : this->vehicle_reprod_times)
        {
            if (saved_time >= current_time)
                break;
            ++pose_i;
        }
        if (pose_i >= this->vehicle_reprod_times.size())
        {
            ROS_WARN_STREAM("Current sim time greater than max saved sim time. Actors are now stopped");
            ROS_WARN_STREAM(current_time << " > " << this->vehicle_reprod_times[this->vehicle_reprod_times.size() - 1]);
            pose_i = this->vehicle_reprod_times.size() - 1;
        }
        if (pose_i > 0)
        {
            // Find closest
            double difft1 = std::abs(this->vehicle_reprod_times[pose_i - 1] - current_time);
            double difft2 = std::abs(this->vehicle_reprod_times[pose_i] - current_time);
            if (difft1 < difft2)
                pose_i--;
        }
    }

    return pose_i;
}

