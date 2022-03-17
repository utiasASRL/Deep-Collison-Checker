#include "custom_velodyne.hh"

#include <algorithm>
#include <assert.h>

#include <gazebo/physics/World.hh>
#include <gazebo/sensors/Sensor.hh>
#include <sdf/sdf.hh>
#include <sdf/Param.hh>
#include <gazebo/common/Exception.hh>
#if GAZEBO_GPU_RAY
#include <gazebo/sensors/GpuRaySensor.hh>
#else
#include <gazebo/sensors/RaySensor.hh>
#endif
#include <gazebo/sensors/SensorTypes.hh>
#include <gazebo/transport/Node.hh>

#include <sensor_msgs/PointCloud2.h>

#include <tf/tf.h>

#if GAZEBO_GPU_RAY
#define RaySensor GpuRaySensor
#define STR_Gpu "Gpu"
#define STR_GPU_ "GPU "
#else
#define STR_Gpu ""
#define STR_GPU_ ""
#endif

namespace gazebo
{
    // Register this plugin with the simulator
    GZ_REGISTER_SENSOR_PLUGIN(GazeboRosVelodyneLaser)

    ////////////////////////////////////////////////////////////////////////////////
    // Constructor
    GazeboRosVelodyneLaser::GazeboRosVelodyneLaser() : nh_(NULL), gaussian_noise_(0), min_range_(0), max_range_(0){
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Destructor
    GazeboRosVelodyneLaser::~GazeboRosVelodyneLaser(){
        ////////////////////////////////////////////////////////////////////////////////
        // Finalize the controller / Custom Callback Queue
        laser_queue_.clear();
        laser_queue_.disable();
        if (nh_)
        {
            nh_->shutdown();
            delete nh_;
            nh_ = NULL;
        }
        callback_laser_queue_thread_.join();
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Load the controller
    void GazeboRosVelodyneLaser::Load(sensors::SensorPtr _parent, sdf::ElementPtr _sdf){

        // Load plugin
        RayPlugin::Load(_parent, _sdf);

        // Initialize Gazebo node
        gazebo_node_ = gazebo::transport::NodePtr(new gazebo::transport::Node());
        gazebo_node_->Init();

       
        // Get the parent ray sensor

        parent_ray_sensor_ = std::dynamic_pointer_cast<sensors::RaySensor>(_parent);

        if (!parent_ray_sensor_)
        {
            gzthrow("GazeboRosVelodyne" << STR_Gpu << "Laser controller requires a " << STR_Gpu << "Ray Sensor as its parent");
        }

        if (_sdf->HasElement("robot_name")){
            this->robot_name = _sdf->GetElement("robot_name")->Get<std::string>();
        }



        if (_sdf->HasElement("building_name")){
            this->building_name = _sdf->GetElement("building_name")->Get<std::string>();
        }

        robot_namespace_ = "/";
        if (_sdf->HasElement("robotNamespace")){
            robot_namespace_ = _sdf->GetElement("robotNamespace")->Get<std::string>();
        }

        if (!_sdf->HasElement("frameName")){
            ROS_INFO("Velodyne laser plugin missing <frameName>, defaults to /world");
            frame_name_ = "/world";
        }
        else{
            frame_name_ = _sdf->GetElement("frameName")->Get<std::string>();
        }

        if (!_sdf->HasElement("min_range"))
        {
            ROS_INFO("Velodyne laser plugin missing <min_range>, defaults to 0");
            min_range_ = 0;
        }
        else
        {
            min_range_ = _sdf->GetElement("min_range")->Get<double>();
        }

        if (!_sdf->HasElement("max_range"))
        {
            ROS_INFO("Velodyne laser plugin missing <max_range>, defaults to infinity");
            max_range_ = INFINITY;
        }
        else
        {
            max_range_ = _sdf->GetElement("max_range")->Get<double>();
        }

        min_intensity_ = std::numeric_limits<double>::lowest();
        if (!_sdf->HasElement("min_intensity"))
        {
            ROS_INFO("Velodyne laser plugin missing <min_intensity>, defaults to no clipping");
        }
        else
        {
            min_intensity_ = _sdf->GetElement("min_intensity")->Get<double>();
        }

        if (!_sdf->HasElement("topicName"))
        {
            ROS_INFO("Velodyne laser plugin missing <topicName>, defaults to /points");
            topic_name_ = "/points";
        }
        else
        {
            topic_name_ = _sdf->GetElement("topicName")->Get<std::string>();
        }

        if (!_sdf->HasElement("gaussianNoise"))
        {
            ROS_INFO("Velodyne laser plugin missing <gaussianNoise>, defaults to 0.0");
            gaussian_noise_ = 0;
        }
        else
        {
            gaussian_noise_ = _sdf->GetElement("gaussianNoise")->Get<double>();
        }

        // Make sure the ROS node for Gazebo has already been initialized
        if (!ros::isInitialized())
        {
            ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. "
                             << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
            return;
        }

        // Create node handle
        nh_ = new ros::NodeHandle(robot_namespace_);

        // Resolve tf prefix
        std::string prefix;
        nh_->getParam(std::string("tf_prefix"), prefix);
        if (robot_namespace_ != "/")
        {
            prefix = robot_namespace_;
        }
        boost::trim_right_if(prefix, boost::is_any_of("/"));
        frame_name_ = tf::resolve(prefix, frame_name_);

        // Advertise publisher with a custom callback queue
        if (topic_name_ != "")
        {
            ros::AdvertiseOptions ao = ros::AdvertiseOptions::create<sensor_msgs::PointCloud2>(
                topic_name_, 1,
                boost::bind(&GazeboRosVelodyneLaser::ConnectCb, this),
                boost::bind(&GazeboRosVelodyneLaser::ConnectCb, this),
                ros::VoidPtr(), &laser_queue_);
            pub_ = nh_->advertise(ao);
        }

        
        this->LoadWorld();
        
        int argc = 0;
        char **argv = NULL;
        ros::init(argc, argv, "CustomSensor");

        // Sensor generation off by default
        parent_ray_sensor_->SetActive(false);

        // Start custom queue for laser
        callback_laser_queue_thread_ = boost::thread(boost::bind(&GazeboRosVelodyneLaser::laserQueueThread, this));

        ROS_INFO("Velodyne %slaser plugin ready, %i lasers", STR_GPU_, parent_ray_sensor_->VerticalRangeCount());
    }

    void GazeboRosVelodyneLaser::ConnectCb()
    {
        boost::lock_guard<boost::mutex> lock(lock_);
        if (pub_.getNumSubscribers())
        {
            if (!sub_)
            {
                sub_ = gazebo_node_->Subscribe(this->parent_ray_sensor_->Topic(), &GazeboRosVelodyneLaser::OnScan, this);
            }
            parent_ray_sensor_->SetActive(true);
        }
        else
        {

            if (sub_)
            {
                sub_->Unsubscribe();
                sub_.reset();
            }

            parent_ray_sensor_->SetActive(false);
        }
    }

    void GazeboRosVelodyneLaser::OnScan(ConstLaserScanStampedPtr &_msg)
    {   
        
        if (!this->world->IsPaused()){
            this->world->SetPaused(true);
        }
        //load vehicle

        double dt = ros::Time::now().toSec() - this->last_update.toSec();
        this->last_update = ros::Time::now();

        if (this->robot == nullptr && (this->robot_name != "")){
            this->robot = this->world->ModelByName(this->robot_name);
            this->robot_links = this->robot->GetLinks();
        }

        auto sensor_pose = this->robot_links[0]->WorldPose();
        auto tf_trans = ignition::math::Pose3d(0,0,0.539,1,0,0,0) + ignition::math::Pose3d(0,0,0,1,0,0,0) + ignition::math::Pose3d(0,0,0.0377,1,0,0,0);
        sensor_pose+=tf_trans;

        // rebuild active quadtree:

        this->active_quadtree = boost::make_shared<QuadTree>(this->building_box);
        
        for (auto act: this->actors){
            if (dt > 0.05 && dt < 0.15){
                this->actor_speed[act->GetName()] = (act->WorldPose().Pos() - this->last_actor_pose[act->GetName()].Pos()).Length()/dt;
            }
            
            auto min = ignition::math::Vector3d(act->WorldPose().Pos().X() - 0.2, act->WorldPose().Pos().Y() - 0.2, 0);
            auto max = ignition::math::Vector3d(act->WorldPose().Pos().X() + 0.2, act->WorldPose().Pos().Y() + 0.2, 0);
            auto box = ignition::math::Box(min,max);
            auto new_node = QTData(box, act, vehicle_type);
            this->active_quadtree->Insert(new_node);

            this->last_actor_pose[act->GetName()] = act->WorldPose();
        }

        const ignition::math::Angle maxAngle = parent_ray_sensor_->AngleMax();
        const ignition::math::Angle minAngle = parent_ray_sensor_->AngleMin();

        const double maxRange = parent_ray_sensor_->RangeMax();
        const double minRange = parent_ray_sensor_->RangeMin();

        const int rayCount = parent_ray_sensor_->RayCount();
        const int rangeCount = parent_ray_sensor_->RangeCount();

        const int verticalRayCount = parent_ray_sensor_->VerticalRayCount();
        const int verticalRangeCount = parent_ray_sensor_->VerticalRangeCount();

        const ignition::math::Angle verticalMaxAngle = parent_ray_sensor_->VerticalAngleMax();
        const ignition::math::Angle verticalMinAngle = parent_ray_sensor_->VerticalAngleMin();

        const double yDiff = maxAngle.Radian() - minAngle.Radian();
        const double pDiff = verticalMaxAngle.Radian() - verticalMinAngle.Radian();

        const double MIN_RANGE = std::max(min_range_, minRange);
        const double MAX_RANGE = std::min(max_range_, maxRange);
        const double MIN_INTENSITY = min_intensity_;

        // Populate message fields
        const uint32_t POINT_STEP = 22;
        sensor_msgs::PointCloud2 msg;
        msg.header.frame_id = frame_name_;
        msg.header.stamp = ros::Time(_msg->time().sec(), _msg->time().nsec());
        msg.fields.resize(6);
        msg.fields[0].name = "x";
        msg.fields[0].offset = 0;
        msg.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
        msg.fields[0].count = 1;
        msg.fields[1].name = "y";
        msg.fields[1].offset = 4;
        msg.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
        msg.fields[1].count = 1;
        msg.fields[2].name = "z";
        msg.fields[2].offset = 8;
        msg.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
        msg.fields[2].count = 1;
        msg.fields[3].name = "time";
        msg.fields[3].offset = 12;
        msg.fields[3].datatype = sensor_msgs::PointField::FLOAT32;
        msg.fields[3].count = 1;
        msg.fields[4].name = "intensity";
        msg.fields[4].offset = 16;
        msg.fields[4].datatype = sensor_msgs::PointField::FLOAT32;
        msg.fields[4].count = 1;
        msg.fields[5].name = "ring";
        msg.fields[5].offset = 20;
        msg.fields[5].datatype = sensor_msgs::PointField::UINT16;
        msg.fields[5].count = 1;
        msg.data.resize(verticalRangeCount * rangeCount * POINT_STEP);

        int i, j;
        uint8_t *ptr = msg.data.data();
        for (i = 0; i < rangeCount; i++)
        {
            for (j = 0; j < verticalRangeCount; j++)
            {

                // Range
                double r = _msg->scan().ranges(i + j * rangeCount);
                // Intensity
                double intensity = _msg->scan().intensities(i + j * rangeCount);
                int catagory;
                float time0 = 0;
                // Ignore points that lay outside range bands or optionally, beneath a
                // minimum intensity level.
                if ((MIN_RANGE >= r) || (r >= MAX_RANGE) || (intensity < MIN_INTENSITY))
                {
                    continue;
                }

                // Noise
                if (gaussian_noise_ != 0.0)
                {
                    r += gaussianKernel(0, gaussian_noise_);
                }

                // Get angles of ray to get xyz for point
                double yAngle;
                double pAngle;

                if (rangeCount > 1)
                {
                    yAngle = i * yDiff / (rangeCount - 1) + minAngle.Radian();
                }
                else
                {
                    yAngle = minAngle.Radian();
                }

                if (verticalRayCount > 1)
                {
                    pAngle = j * pDiff / (verticalRangeCount - 1) + verticalMinAngle.Radian();
                }
                else
                {
                    pAngle = verticalMinAngle.Radian();
                }

                // pAngle is rotated by yAngle:
                if ((MIN_RANGE < r) && (r < MAX_RANGE))
                {
                    float x = r * cos(pAngle) * cos(yAngle);
                    float y = r * cos(pAngle) * sin(yAngle);
                    float z = r * sin(pAngle);
                    *((float *)(ptr + 0)) = x; //x
                    *((float *)(ptr + 4)) = y; //y
                    *((float *)(ptr + 8)) = z; //z

                    
                    ignition::math::Vector3d point = sensor_pose.CoordPositionAdd(ignition::math::Vector3d(x,y,z));

                    if (point.Z() < 0.05){
                        catagory = 0; //ground
                        *((float *)(ptr + 12)) = time0;
                        *((float *)(ptr + 16)) = catagory;
                        *((uint16_t *)(ptr + 20)) = j; // ring

                        ptr += POINT_STEP;
                        continue;
                    }

                    // query quadtrees

                    std::vector<gazebo::physics::ActorPtr> near_actors;
                    std::vector<gazebo::physics::EntityPtr> near_objects;

                    double resolution = 0.7;
                    auto min = ignition::math::Vector3d(point.X() - resolution, point.Y() - resolution, 0);
                    auto max = ignition::math::Vector3d(point.X() + resolution, point.Y() + resolution, 0);
                    auto query_range = ignition::math::Box(min,max);

                    std::vector<CollObj> check_objects;
                    
                    std::vector<QTData> query_objects = this->static_quadtree->QueryRange(query_range);
                    for (auto n: query_objects){
                        if (n.type == entity_type){
                            
                            auto entity = boost::static_pointer_cast<gazebo::physics::Entity>(n.data);
                            int cat;
                            std::string name = entity->GetParent()->GetParent()->GetName();
                           
                            if (name == this->building_name){
                                cat = 5; //wall
                            } else if (name.substr(0,5) == "table"){
                                cat = 4; //table
                            } else if (name.substr(0,5) == "chair"){
                                cat = 1; //chair
                            } else if (name.substr(0,4) == "door"){
                                cat = 6;
                                
                            } else {
                                cat = 0; //ground
                            }
                            check_objects.push_back(CollObj(cat, entity->BoundingBox()));
                            //near_objects.push_back(boost::static_pointer_cast<gazebo::physics::Entity>(n.data));
                        }
                    }

                    std::vector<QTData> query_actors = this->active_quadtree->QueryRange(query_range);
                    for (auto n: query_actors){
                        if (n.type == vehicle_type){
                            auto actor = boost::static_pointer_cast<gazebo::physics::Actor>(n.data);
                            int cat;
                            if (actor_speed[actor->GetName()] < 10e-5){
                                cat = 3; // stationary actors 
                            } else{
                                cat = 2; // moving actors 
                            }
                            
                            auto box = ignition::math::Box(ignition::math::Vector3d(actor->WorldPose().Pos().X()-0.2, actor->WorldPose().Pos().Y()-0.2,0), ignition::math::Vector3d(actor->WorldPose().Pos().X()+0.2, actor->WorldPose().Pos().Y()+0.2, 1));
                            check_objects.push_back(CollObj(cat, box));
                            //near_actors.push_back(boost::static_pointer_cast<gazebo::physics::Actor>(n.data));
                        }
                    }
                    
                   

                    //collision checks:

                    
                    if (check_objects.size() == 0){
                        catagory = 5;
                    } else{

                        
                        catagory = 0;
                        double min_dist = point.Z();
                        if (min_dist > 0.05){
                            catagory = 5;
                            min_dist = 10e9;
                        }

                        for (auto n: check_objects){
                            double dist;
                            if (n.cat == 2 || n.cat ==3){
                                //if we a checking a person, treat them as a cylindar with a radius that is equal to box width/2
                                double r = n.box.Max().X() - n.box.Min().X();
                                auto other = (n.box.Min()+n.box.Max())/2; // the center of the box is the position of the person
                                dist = (point-other).Length()-r;
                                if (dist < 0){
                                    dist = 0;
                                }
                            } else{
                                dist = utilities::dist_to_box(point, n.box);
                                
                            }

                            if (n.cat == 5 && dist <0.05){
                                catagory = 5; 
                                break;
                            }
                            
                            if (dist <= min_dist){
                                min_dist = dist;
                                catagory = n.cat;
                            }
                        }
                    }
                  
                    

                    *((float *)(ptr + 12)) = time0;
                    *((float *)(ptr + 16)) = catagory;
                    *((uint16_t *)(ptr + 20)) = j; // ring

                    ptr += POINT_STEP;
                }
            }
        }

        // Populate message with number of valid points
        msg.point_step = POINT_STEP;
        msg.row_step = ptr - msg.data.data();
        msg.height = 1;
        msg.width = msg.row_step / POINT_STEP;
        msg.is_bigendian = false;
        msg.is_dense = true;
        msg.data.resize(msg.row_step); // Shrink to actual size

        // Publish output
        pub_.publish(msg);

        if (this->world->IsPaused()){
            this->world->SetPaused(false);
        }
    }

    void GazeboRosVelodyneLaser::laserQueueThread()
    {
        while (nh_->ok())
        {
            laser_queue_.callAvailable(ros::WallDuration(0.01));
        }
    }

    void GazeboRosVelodyneLaser::LoadWorld(){
        this->last_update = ros::Time::now();
        if (gazebo::physics::has_world("default")){
            this->world = gazebo::physics::get_world("default");
            std::cout << "SENSOR FOUND WORLD, proceeding\n";
        } else {
            std::cout << "FAILED TO FIND WORLD\n";
            return;
        }
    
        gazebo::physics::ModelPtr building;
        //auto building = this->world->ModelByName(this->building_name);
        // there is some error on the above line when including the viewbots
        for (unsigned int i = 0; i < world->ModelCount(); ++i) {
            auto model = world->ModelByIndex(i);
            if (model->GetName() == this->building_name){
                building = model;
                std::cout << "Added building to sensor plugin\n";
            }
        }
        
        this->building_box = building->BoundingBox();
        this->building_box.Min().X()-=1;
        this->building_box.Min().Y()-=1;
        this->building_box.Max().X()+=1;
        this->building_box.Max().Y()+=1;
        this->static_quadtree = boost::make_shared<QuadTree>(this->building_box);
        this->active_quadtree = boost::make_shared<QuadTree>(this->building_box);

        
        for (unsigned int i = 0; i < this->world->ModelCount(); ++i) {
            auto model = this->world->ModelByIndex(i);
            if (model->GetName() == "" || model == nullptr){
                std::cout << "NULL model found\n";
            }
                
            auto act = boost::dynamic_pointer_cast<gazebo::physics::Actor>(model);

            if (act){
                this->actors.push_back(act);
                this->last_actor_pose[act->GetName()] = act->WorldPose();
                this->actor_speed[act->GetName()] = 0;
                auto min = ignition::math::Vector3d(act->WorldPose().Pos().X() - 0.2, act->WorldPose().Pos().Y() - 0.2, 0);
                auto max = ignition::math::Vector3d(act->WorldPose().Pos().X() + 0.2, act->WorldPose().Pos().Y() + 0.2, 0);
                auto box = ignition::math::Box(min,max);
                auto new_node = QTData(box, act, vehicle_type);
                this->active_quadtree->Insert(new_node);
                continue;
            } 
            
            if (model->GetName() != "ground_plane" && model->GetName() != this->robot_name){
                if (model->GetName().substr(0,11) == "global_plan" || model->GetName().substr(0,10) == "local_plan"){
                    continue;
                }
               
                auto links = model->GetLinks();
                for (gazebo::physics::LinkPtr link: links){
                    std::vector<gazebo::physics::CollisionPtr> collision_boxes = link->GetCollisions();
                    for (gazebo::physics::CollisionPtr collision_box: collision_boxes){
                        this->collision_entities.push_back(collision_box);
                        auto box = collision_box->BoundingBox();
                       
                        box.Max().Z() = 0;
                        box.Min().Z() = 0;
                        auto new_node = QTData(box, collision_box, entity_type);
                        this->static_quadtree->Insert(new_node);
                        
                    }
                        
                }
            }
        
           
        
        
        }
        
    }
} // namespace gazebo
