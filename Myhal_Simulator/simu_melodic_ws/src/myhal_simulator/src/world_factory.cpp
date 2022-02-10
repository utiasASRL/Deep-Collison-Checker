#include "world_factory.hh"

#include "parse_tour.hh"


int main(int argc, char ** argv){

    ROS_WARN("WORLD GENERATION BEGINNING:");

    auto world_handler = WorldHandler();

    world_handler.Load();

    
    return 0;
}

void WorldHandler::Load(){

    std::cout << "Loading Parameters\n";

    this->LoadParams();

    std::cout << "Adding cameras\n";

    this->AddCameras();

    std::cout << "Done adding cameras\n";

    std::cout << "Filling Rooms\n";
    
    for (auto r_info: this->rooms)
    {

        this->FillRoom(r_info);
        
        r_info->room->AddToWorld(this->world_string);

    }

    std::cout << "Creating doors\n";

    for (auto door: this->doors){
        door->AddToWorld(this->world_string);
    }


    std::cout << "Writing to file\n";


    this->WriteToFile("myhal_sim.world");

    std::cout << "WORLD CREATED!\n";

}

void WorldHandler::AddCameras(){
    
    for (auto info: this->cam_info){
        std::string name = "CAM_";
        // we will encode camera information in it's name
        name += std::to_string(info->mode) + "_"; 
        name += std::to_string(info->period) + "_"; 
        name += std::to_string(info->dist) + "_"; 
        name += std::to_string(info->x) + "_"; 
        name += std::to_string(info->y) + "_"; 
        name += std::to_string(info->z);

        auto pose = ignition::math::Pose3d(info->x, info->y, info->z, 0,0,0,0);

        char temp[500];
        std::string current_path(getcwd(temp, sizeof(temp)));
        auto path = current_path + "/../Data/Simulation_v2/simulated_runs/" + this->start_time + "/logs-" + this->start_time + "/videos/";

        auto cam = myhal::Camera(name, pose, path);
        cam.AddToWorld(this->world_string);
    }

}

WorldHandler::WorldHandler(){
    this->world_string = "";
}

void WorldHandler::LoadParams(){

    //TODO: fix error handelling 

    int argc = 0;
    char **argv = NULL;
    ros::init(argc, argv, "WorldHandler");
    ros::NodeHandle nh;

    // READ camera info

    std::vector<std::string> camera_names;
    if (!nh.getParam("camera_names", camera_names)){
        std::cout << "ERROR READING CAMERA NAMES" << std::endl;
    }

    for (auto name: camera_names){
        std::vector<double> info;
        if (!nh.getParam(name, info)){
            std::cout << "MIS-NAMED CAMERA" << std::endl;
            continue;
        }
        if (info.size() < 5){
            std::cout << "Camera info is malformed" << std::endl;
        }
        auto cam = std::make_shared<CamInfo>(info[0], info[1], info[2], info[3], info[4], info[5]);
        this->cam_info.push_back(cam);
    }

    

    // READ BUILDING INFO

    if (!nh.getParam("start_time", this->start_time)){
        std::cout << "ERROR READING START TIME: ANY VIDEOS WILL BE SAVED TO /tmp/\n";
        this->start_time = "";
    }

    if (!nh.getParam("tour_name", this->tour_name)){
        std::cout << "ERROR READING TOUR NAME\n";
        this->tour_name = "A_tour";
    }

    double cam_frac;

    if (!nh.getParam("cam_frac", cam_frac)){
        std::cout << "ERROR READING CAM FRAC\n";
        cam_frac = 0;
    }

    std::vector<int> cam_rel_pos; 

    if (!nh.getParam("cam_rel_pos", cam_rel_pos)){
        std::cout << "ERROR READING CAM REL POS\n";
        cam_rel_pos = {0,0,0};
    }



    TourParser parser = TourParser(this->tour_name);


    this->route = parser.GetRoute();
    this->route.insert(this->route.begin(), ignition::math::Pose3d(ignition::math::Vector3d(0,0,0), ignition::math::Quaterniond(0,0,0,1)));
    

    int mod_cams;
    if (cam_frac > 0){
        mod_cams = (int) std::round(1/cam_frac);
    }

        
    this->costmap = std::make_shared<Costmap>(ignition::math::Box(ignition::math::Vector3d(-21.55,-21.4,0), ignition::math::Vector3d(21.55,21.4,0)), 0.2);


    char temp[500];

    std::string current_path(getcwd(temp, sizeof(temp)));

    happly::PLYData plyIn(current_path + "/simu_melodic_ws/src/myhal_simulator/params/default_params/myhal_walls.ply");

    auto static_objects = ReadObjects(plyIn);

    for (auto obj: static_objects){
        if (obj.MinZ() < 1.5 && obj.MaxZ() >10e-2){
            auto box = obj.Box();
            this->costmap->AddObject(box);
            box.Min().X()-=robot_radius;
            box.Min().Y()-=robot_radius;
            box.Max().X()+=robot_radius;
            box.Max().Y()+=robot_radius;

            this->walls.push_back(box);
        }
    }

    std::vector<ignition::math::Vector3d> paths;

    for (int i =0; i< route.size(); i++)
    {
        if ((cam_frac > 0) && (i % mod_cams == 0)){
            std::cout << utilities::color_text("Read Tour Camera", PURPLE) << std::endl; 
            auto cam = std::make_shared<CamInfo>(0, route[i].Pos().X() + cam_rel_pos[0], route[i].Pos().Y() + cam_rel_pos[1], route[i].Pos().Z() + cam_rel_pos[2], -1, -1);
            this->cam_info.push_back(cam);
        }

        if (i == (route.size()-1)){
            continue;
        }
        auto start = route[i];
        auto end = route[i+1];

        std::vector<ignition::math::Vector3d> path;
        this->costmap->AStar(start.Pos(), end.Pos(), path, false);
        paths.insert(paths.end(),path.begin(),path.end());
    }

    // Create doors

    for (auto obj: static_objects)
    {
        if (obj.MinZ()  >= (2 - 10e-3))
        {

            // Object box and position (object is the wall above the door)
            auto box = obj.Box();
            auto pos = (box.Min() + box.Max())/2;
            pos.Z() = 0;

            int open = ignition::math::Rand::IntUniform(0,1);

            auto door = std::make_shared<myhal::IncludeModel>("door", ignition::math::Pose3d(pos, ignition::math::Quaterniond(0,0,0,0)), "model://simple_door2", 0.9, 0.15);
            door->pose.Pos().Z() = 1;

        
            if (box.Max().X() - box.Min().X() > 0.2){
                // horizontal door
              
                door->pose.Pos().X() -= 0.45;
            } else{
                // vertical door
                door->RotateClockwise(1.571);
                door->pose.Pos().Y() -= 0.45;
            }

            bool intersected = false;
            bool near = false;

            for (int i =0; i<paths.size()-1; i++)
            {
                auto first = paths[i];
                first.Z() = 0;

                if ((first - pos).Length() < 1){
                    near = true;
                    auto second = paths[i+1];
                    second.Z() = 0;
                    auto line = ignition::math::Line3d(first, second);

                    auto door_box = box;
                    door_box.Min().Z() = 0;
                    door_box.Max().Z() = 0;

                    auto edges = utilities::get_box_edges(door_box);

                    for (auto edge: edges){
                        if (edge.Intersect(line) || utilities::inside_box(door_box, first, true)){
                            open = 1;
                            intersected = true;
                            break;
                        }
                    }
                }
            }

            //if (!intersected && near){
            //    open = 0;
            //}

            open = 1;

            auto yaw = door->pose.Rot().Yaw();
            
            door->pose.Rot() = ignition::math::Quaterniond(0,0,yaw+(open*1.571));

            if (!intersected){
                this->doors.push_back(door); //use this for now to eliminate failed tours 
            }
            

           
        }
    }

    /// READ PLUGIN INFO
    
    std::vector<std::string> plugin_names;
    if (!nh.getParam("plugin_names", plugin_names)){
        std::cout << "ERROR READING PLUGIN NAMES\n";
        return;
    }

    for (auto name: plugin_names){
        //std::cout << name << std::endl;
        
        std::map<std::string, std::string> info;
        if (!nh.getParam(name, info)){
            std::cout << "ERROR READING PLUGIN PARAMS\n";
            return;
        }

        //std::shared_ptr<SDFPlugin> plugin = std::make_shared<SDFPlugin>(info["name"], info["filename"]);

        std::map<std::string, std::string>::iterator it;

        for ( it = info.begin(); it != info.end(); it++ ){
            /*
            if (it->first == "name"){
                continue;
            }
            */

            auto new_plugin = std::make_shared<SDFPlugin>(it->first, it->second);
            
            /*
            if (it->first == "max_speed"){
                double speed = std::stod(info[it->first]);
                speed += ignition::math::Rand::DblUniform(-speed/5, speed/5);
                plugin->AddSubtag(it->first, std::to_string(speed));
            } else{
                plugin->AddSubtag(it->first, info[it->first]);
            }
            */

           this->vehicle_plugins[name].push_back(new_plugin);
            
        }
        //this->vehicle_plugins[name] = plugin;
    }

    /// READ ANIMATION INFO

    std::vector<std::string> animation_names;
    if (!nh.getParam("animation_names", animation_names)){
        std::cout << "ERROR READING ANIMATION NAMES\n";
        return;
    }

    for (auto name: animation_names){
        //std::cout << name << std::endl;
        
        std::map<std::string, std::string> info;
        if (!nh.getParam(name, info)){
            std::cout << "ERROR READING ANIMATION PARAMS\n";
            return;
        }

        std::shared_ptr<SDFAnimation> animation = std::make_shared<SDFAnimation>(name, info["filename"], true);

        this->animation_list.push_back(animation);
       
    }

    /// READ MODEL INFO

    std::vector<std::string> model_names;
    if (!nh.getParam("model_names", model_names)){
        std::cout << "ERROR READING MODEL NAMES\n";
        return;
    }

    for (auto name: model_names){
      
        
        std::map<std::string, std::string> info;
        if (!nh.getParam(name, info)){
            std::cout << "ERROR READING MODEL PARAMS\n";
            return;
        }
        std::shared_ptr<ModelInfo> m_info;
        if (info.find("height") != info.end()){
            m_info = std::make_shared<ModelInfo>(name, info["filename"], std::stod(info["width"]), std::stod(info["length"]), std::stod(info["height"]));
        } else{
            m_info = std::make_shared<ModelInfo>(name, info["filename"], std::stod(info["width"]), std::stod(info["length"]));
        }
        this->model_info[name] = m_info;
    }

    /// READ TABLE INFO

    std::vector<std::string> table_group_names;
    if (!nh.getParam("table_group_names", table_group_names)){
        std::cout << "ERROR READING TABLE GROUP NAMES\n";
        return;
    }

    for (auto name: table_group_names){
        std::map<std::string, std::string> info;
    

        if (!nh.getParam(name,info)){
            std::cout << "ERROR READING TABLE GROUP PARAMS\n";
            return;
        }

        std::shared_ptr<TableInfo> t_info = std::make_shared<TableInfo>(name, info["table"], info["chair"]);
        this->table_info[name] = t_info;
    }



    /// READ ACTOR INFO 

    std::vector<std::string> actor_names;
    if (!nh.getParam("actor_names", actor_names)){
        std::cout << "ERROR READING ACTOR NAMES\n";
        return;
    }

    for (auto name: actor_names){
        
        std::map<std::string, std::string> info;
        if (!nh.getParam(name, info)){
            std::cout << "ERROR READING ACTOR PARAMS\n";
            return;
        }

        std::shared_ptr<ActorInfo> a_info = std::make_shared<ActorInfo>(name, info["filename"], info["plugin"], std::stod(info["obstacle_margin"]));

        this->actor_info[name] = a_info;
    }

    /// READ SCENARIO INFO

    std::vector<std::string> scenario_names;
    if (!nh.getParam("scenario_names", scenario_names)){
        std::cout << "ERROR READING SCENARIO NAMES\n";
        return;
    }


    std::cout << "\n-----------------------------------------" << std::endl;
    std::cout << "Reading Scenarios: " << std::endl;
    for (auto name: scenario_names){

        
        std::map<std::string, std::string> info;
        if (!nh.getParam(name, info)){
            std::cout << "ERROR READING SCENARIO PARAMS\n";
            return;
        }

        int pop_num = -1;
        double pop_density = -1;
        double pop_rand_range = -1;
        if (info.count("pop_density"))
            pop_density = std::stod(info["pop_density"]);
        if (info.count("pop_num"))
            pop_num = std::stoi(info["pop_num"]);
        if (info.count("pop_rand_range"))
            pop_rand_range = std::stod(info["pop_rand_range"]);

        // Random population density
        if (pop_density > 0 && pop_rand_range > 0)
        {
            double min_density = std::max(0.0, pop_density - pop_rand_range);
            double max_density = pop_density + pop_rand_range;
            pop_density = ignition::math::Rand::DblUniform(min_density, max_density);
        }

        // Random table percentage
        double table_percentage = -1;
        double table_rand_range = -1;
        if (info.count("table_percentage"))
            table_percentage = std::stod(info["table_percentage"]);
        if (info.count("table_rand_range"))
            table_rand_range = std::stod(info["table_rand_range"]);
            
        if (table_percentage > 0 && table_rand_range > 0)
        {
            double min_percentage = std::max(0.0, table_percentage - table_rand_range);
            double max_percentage = std::min(1.0, table_percentage + table_rand_range);
            table_percentage = ignition::math::Rand::DblUniform(min_percentage, max_percentage);
        }

        std::cout << name << ": Tables = " << (int)(table_percentage* 100) << "%  |  Actors = " << (int)(pop_density* 100)<< "%" << std::endl;
        std::shared_ptr<Scenario> scenario = std::make_shared<Scenario>(pop_density, pop_num, table_percentage, info["actor"]);
        

        // Save info in log file
        std::ofstream log_file;
        std::string filepath = current_path + "/../Data/Simulation_v2/simulated_runs/" + start_time + "/";

        log_file.open(filepath + "/logs-" + start_time + "/log.txt", std::ios_base::app);
        log_file << "\nScenario: " << name << "\nTables: " << table_percentage << "\nActors: " << pop_density << std::endl;
        log_file.close();

        /*std::vector<std::string> model_list; 

        if (!nh.getParam(info["model_list"], model_list)){
            std::cout << "ERROR READING MODEL LIST\n";
            return;
        }


        for (auto model_name: model_list){
            scenario->AddModel(this->model_info[model_name]);
        }*/


        std::vector<std::string> table_group_list; 

        if (!nh.getParam(info["table_group_list"], table_group_list)){
            std::cout << "ERROR READING TABLE GROUP LIST\n";
            return;
        }


        for (auto table_group_name: table_group_list){
            scenario->AddTable(this->table_info[table_group_name]);
        }

        this->scenarios[name] = scenario;
    }
    std::cout << "-----------------------------------------" << std::endl;

    //READ CUSTOM SCENARIO INFO

    if (!nh.getParam("/custom_actor_spawn/use_custom_spawn_room", this->use_custom_spawn_room)){
        ROS_INFO("COULD NOT READ USE CUSTOM ACTOR SPAWN PARAMETER. SET TO FALSE.");
        this->use_custom_spawn_room = "";
    }

    if (!this->use_custom_spawn_room.empty()){
        if(!nh.getParam("/custom_actor_spawn_coordinates", this->custom_actor_spawn_parameters)){
            ROS_ERROR("COULD NOT READ ACTOR SPAWN COORDINATES WHILE CUSTOM ACTOR SPAWN IS ACTIVATED. CHECK FILE STRUCTURE: custom_simulation_params.yaml");
        }
    }

    /// READ ROOM INFO

    if (!nh.getParam("room_names", this->room_names)){
        std::cout << "ERROR READING ROOM NAMES\n";
        return;
    }

    for (auto name: this->room_names){
        //std::cout << name << std::endl;
        
        std::map<std::string, std::string> info;
        if (!nh.getParam(name, info)){
            std::cout << "ERROR READING ROOM PARAMS\n";
            return;
        }

        std::map<std::string, double> geometry; 

        if (!nh.getParam(info["geometry"], geometry)){
            std::cout << "ERROR READING ROOM GEOMETRY\n";
            return;
        }

        std::shared_ptr<myhal::Room> room = std::make_shared<myhal::Room>(geometry["x_min"], geometry["y_min"], geometry["x_max"], geometry["y_max"], this->walls, this->route, (bool) std::stoi(info["enclosed"]));

        std::vector<double> poses;

        if (!nh.getParam(info["positions"], poses)){
            std::cout << "ERROR READING POSITION PARAMS\n";
            return;
        }
        std::vector<std::vector<double>> positions;
        for (int j =0; j < (int) poses.size()-1; j+=2){
            positions.push_back({poses[j],poses[j+1]});
        }
  

        auto r_info = std::make_shared<RoomInfo>(room, info["scenario"], positions, name);
        this->rooms.push_back(r_info);
        
    }

}

void WorldHandler::FillRoom(std::shared_ptr<RoomInfo> room_info){

    if ( this->scenarios.find(room_info->scenario) == this->scenarios.end()){
        std::cout << "ERROR: YOU HAVE SPECIFIED A SCENARIO THAT DOESN'T EXIST" << std::endl;
        return;
    }

    // Get scenario and number of tables
    auto scenario = this->scenarios[room_info->scenario];
    int num_models = (int) (scenario->model_percentage*((room_info->positions.size())));
    
    // Shuffle the order of the possible table positions
    std::random_shuffle(room_info->positions.begin(), room_info->positions.end());
    
    // Add the tables
    // **************

    for (int i = 0; i < num_models; ++i)
    {

        auto t_info = scenario->GetRandomTable();

        auto random_pose = ignition::math::Pose3d(room_info->positions[i][0], room_info->positions[i][1], 0, 0, 0, 0); //TODO: specify randomization parameters in yaml
        if (t_info){
            //auto new_model = std::make_shared<myhal::IncludeModel>(m_info->name, random_pose, m_info->filename, m_info->width, m_info->length);
            if (!this->model_info.count(t_info->table_name)){
                std::cout << "TABLE NAME ERROR\n";
                return;
            } 

            if (!this->model_info.count(t_info->chair_name)){
                std::cout << "CHAIR NAME ERROR\n";
                return;
            } 

            
            auto t_model_info = this->model_info[t_info->table_name];
            auto c_model_info = this->model_info[t_info->chair_name];
            
            auto table_model = std::make_shared<myhal::IncludeModel>(t_model_info->name, random_pose, t_model_info->filename, t_model_info->width, t_model_info->length);
            auto chair_model = std::make_shared<myhal::IncludeModel>(c_model_info->name, random_pose, c_model_info->filename, c_model_info->width, c_model_info->length);

            // Add table and a random number of chairs
            double rotation = 1.5707 * ignition::math::Rand::IntUniform(0,1);
            auto table_group = std::make_shared<myhal::TableGroup>(table_model, chair_model, rotation); 
            room_info->room->AddModel(table_group->table_model);
            
            // Add Sitter on the chairs
            for (auto chair: table_group->chairs)
            {
                room_info->room->AddModel(chair);
                // 50% chance of someone sitting on the chair 
                if (ignition::math::Rand::IntUniform(0,1) == 1){
                   
                    //std::shared_ptr<SDFPlugin> plugin = std::make_shared<SDFPlugin>("sitter_plugin", "libvehicle_plugin.so");
                    auto sitter_plugin = std::make_shared<SDFPlugin>("vehicle_type", "sitter");
                    auto chair_plugin = std::make_shared<SDFPlugin>("chair", chair->name);
                    //plugin->AddSubtag("vehicle_type", "sitter");
                    //plugin->AddSubtag("chair", chair->name);
                    auto sit_pose = chair->pose;
                    sit_pose.Pos().Z() = c_model_info->height;
                    auto sitter = std::make_shared<myhal::Actor>("sitter", sit_pose, "sitting.dae", 0.5, 0.5);
                    for (auto animation: this->animation_list){
                        sitter->AddAnimation(animation);
                    }
                    //sitter->AddPlugin(plugin);
                    sitter->AddPlugin(sitter_plugin);
                    sitter->AddPlugin(chair_plugin);
                    room_info->room->models.push_back(sitter); 
                }

            }
        

            
        } 

         
    }
   
    if (this->use_custom_spawn_room.find(room_info->room_name) != std::string::npos)
    {
        int num_actors = scenario->pop_num;
        if (num_actors <= 0)
            num_actors = (int) floor((scenario->pop_density)*(room_info->room->Area()));

        // todo: replace by function that divide the density to get desired number of actors while avoiding empty rooms
        // To avoid empty rooms, only apply to room that have a non-zero density. Originally wanted to use room names but its not available here
        if (num_actors!=0){

            num_actors = this->custom_actor_spawn_parameters["num_actors"];
            auto a_info = this->actor_info[scenario->actor];

            for (int i =0; i<num_actors; i++){
                auto new_actor = std::make_shared<myhal::Actor>(a_info->name, ignition::math::Pose3d(0,0,1,0,0,ignition::math::Rand::DblUniform(0,6.28)), a_info->filename, a_info->width, a_info->length); //TODO randomize initial Rot

                for (auto animation: this->animation_list){
                    new_actor->AddAnimation(animation);
                    
                }
                auto plugin_list = this->vehicle_plugins[a_info->plugin];
            
                for (auto plugin: plugin_list){ 
                    new_actor->AddPlugin(plugin);
                }

                double x_spawn, y_spawn;
                x_spawn = (double) this->custom_actor_spawn_parameters["spawn_x_" + std::to_string(i)];
                y_spawn = (double) this->custom_actor_spawn_parameters["spawn_y_" + std::to_string(i)];
                room_info->room->AddModelSelectively(new_actor, x_spawn, y_spawn);

            }
        }
        return;
    }

    else{
        
        int num_actors = scenario->pop_num;
        if (num_actors <= 0)
            num_actors = (int) floor((scenario->pop_density)*(room_info->room->Area()));
            
        auto a_info = this->actor_info[scenario->actor];
        //auto plugin = this->vehicle_plugins[a_info->plugin];

        for (int i =0; i<num_actors; i++){
            auto new_actor = std::make_shared<myhal::Actor>(a_info->name, ignition::math::Pose3d(0,0,1,0,0,ignition::math::Rand::DblUniform(0,6.28)), a_info->filename, a_info->width, a_info->length); //TODO randomize initial Rot

            for (auto animation: this->animation_list){
                new_actor->AddAnimation(animation);
            }
            auto plugin_list = this->vehicle_plugins[a_info->plugin];
        
            for (auto plugin: plugin_list){
                new_actor->AddPlugin(plugin);
            }
            
            
            //new_actor->AddPlugin(plugin);
            
            
        
            room_info->room->AddModelRandomly(new_actor);
        }
        return;
    }
}


void WorldHandler::WriteToFile(std::string out_name){

    char temp[500];
    std::string current_path(getcwd(temp, sizeof(temp)));
    std::string in_string = current_path + "/simu_melodic_ws/src/myhal_simulator/worlds/myhal_template.txt";
    std::string out_string = current_path + "/simu_melodic_ws/src/myhal_simulator/worlds/" + out_name;

    std::ifstream in = std::ifstream(in_string);

    if (in){
        ROS_INFO("TEMPLATE FILE FOUND");
	} else{
        ROS_ERROR("TEMPLATE FILE NOT FOUND");
        return;
	}

    std::ofstream out;
    out.open(out_string);

    char str[255];
	int line =0;

	while(in) {
		in.getline(str, 255);  // delim defaults to '\n'
		if(in) {
			
			if (line == 112){
				// insert writing people and furnature here
				
				out << this->world_string;
			}

			out << str << std::endl;
			line++;

		}
	}	

    out.close();
    in.close();
}

Scenario::Scenario(double _pop_density, int _pop_num, double _model_percentage, std::string _actor){
    this->model_percentage = _model_percentage;
    this->actor = _actor;
    this->pop_density = _pop_density;
    this->pop_num = _pop_num;
}

void Scenario::AddModel(std::shared_ptr<ModelInfo> model){
    this->models.push_back(model);
}

void Scenario::AddTable(std::shared_ptr<TableInfo> table){
    this->tables.push_back(table);
}

std::shared_ptr<ModelInfo> Scenario::GetRandomModel(){
    if (this->models.size() <= 0){
        std::cout << "ERROR NO MODEL FOUND" << std::endl;
        return nullptr;
    }
   
    int i = ignition::math::Rand::IntUniform(0, this->models.size()-1);

    return this->models[i];
}

std::shared_ptr<TableInfo> Scenario::GetRandomTable(){
    if (this->tables.size() <= 0){
        std::cout << "ERROR NO TABLE FOUND" << std::endl;
        return nullptr;
    }
   
    int i = ignition::math::Rand::IntUniform(0, this->tables.size()-1);

    return this->tables[i];
}
