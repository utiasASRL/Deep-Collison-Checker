#include "vehicles.hh"
#include "Perlin.h"
#include <thread>

PathViz::PathViz(std::string name, int num_dots, ignition::math::Vector4d color, gazebo::physics::WorldPtr world) : name(name), num_dots(num_dots), color(color), world(world)
{

    boost::shared_ptr<sdf::SDF> sdf = boost::make_shared<sdf::SDF>();
    sdf->SetFromString(
        "<sdf version ='1.6'>\
          <model name ='path'>\
          </model>\
        </sdf>");

    auto model = sdf->Root()->GetElement("model");
    model->GetElement("static")->Set(true);
    model->GetAttribute("name")->SetFromString(this->name);

    for (int i = 0; i < this->num_dots; i++)
    {
        auto pos = ignition::math::Vector3d(50, 50, 0); //put them offscreen to start
        auto link = model->AddElement("link");
        link->GetElement("pose")->Set(pos);
        link->GetAttribute("name")->SetFromString(name + "_" + std::to_string(i));
        auto cylinder = link->GetElement("visual")->GetElement("geometry")->GetElement("cylinder");
        cylinder->GetElement("radius")->Set(0.03);
        cylinder->GetElement("length")->Set(0.001);
        auto mat = link->GetElement("visual")->GetElement("material");
        mat->GetElement("ambient")->Set(this->color);
        mat->GetElement("diffuse")->Set(this->color);
        mat->GetElement("specular")->Set(this->color);
        mat->GetElement("emissive")->Set(this->color);
    }

    this->world->InsertModelSDF(*sdf);
}

void PathViz::OnUpdate(const nav_msgs::Path::ConstPtr &plan)
{
    if (this->model == nullptr)
    {
        this->model = this->world->ModelByName(this->name);
        this->dots = this->model->GetLinks();
    }

    auto path = plan->poses;
    if (path.size() < 0)
    {
        return;
    }
    double frac = std::min(1.0, ((double)this->num_dots) / ((double)path.size()));
    int mod = (int)std::ceil(1 / frac);

    int link_count = 0;
    for (int i = 0; i < path.size(); i++)
    {
        if ((i % mod) != 0)
        {
            continue;
        }
        auto pose = path[i];
        auto p = pose.pose.position;
        auto world_pose = ignition::math::Pose3d(p.x, p.y, p.z, 0, 0, 0);
        if (link_count < this->dots.size())
        {
            this->dots[link_count++]->SetWorldPose(world_pose);
        }
    }
    // reset unused dots
    for (int i = link_count; i < this->dots.size(); i++)
    {
        auto world_pose = ignition::math::Pose3d(50, 50, 0, 0, 0, 0);
        this->dots[link_count++]->SetWorldPose(world_pose);
    }
}

SmartCam::SmartCam(gazebo::physics::ModelPtr self, ignition::math::Vector3d initial_pos) : self(self), updated_pos(initial_pos) {}

void SmartCam::UpdateModel()
{
    this->heading.Normalize();
    double yaw = std::atan2(this->heading.Y(), this->heading.X());
    double pitch = -std::atan2(heading.Z(), std::hypot(heading.X(), heading.Y()));
    self->SetWorldPose(ignition::math::Pose3d(this->updated_pos, ignition::math::Quaterniond(0, pitch, yaw)));
}

void Sentry::OnUpdate(double dt, std::vector<ignition::math::Vector3d> &robot_traj)
{
    if (robot_traj.size() == 0)
    {
        return;
    }
    auto target = robot_traj.back();
    this->heading = target - this->updated_pos;
    this->UpdateModel();
}

Hoverer::Hoverer(gazebo::physics::ModelPtr self, ignition::math::Vector3d initial_pos, double T) : SmartCam(self, initial_pos), T(T), relative_pos(initial_pos) {}

void Hoverer::OnUpdate(double dt, std::vector<ignition::math::Vector3d> &robot_traj)
{

    if (robot_traj.size() == 0)
    {
        return;
    }
    auto target = robot_traj.back();

    if (this->T > 0)
    {
        auto dyaw = (dt / this->T) * 6.28;
        auto rt = ignition::math::Quaterniond(0, 0, dyaw);
        this->relative_pos = rt.RotateVector(this->relative_pos);
    }
    this->updated_pos = target + this->relative_pos;
    this->heading = target - this->updated_pos;
    this->UpdateModel();
}

Stalker::Stalker(gazebo::physics::ModelPtr self, ignition::math::Vector3d initial_pos, double dist) : SmartCam(self, initial_pos), dist(dist) {}

void Stalker::OnUpdate(double dt, std::vector<ignition::math::Vector3d> &robot_traj)
{
    // maintain this->dist distance from the robot along it's path

    if (robot_traj.size() == 0)
    {
        return;
    }

    auto target = robot_traj.back();
    if (robot_traj.size() > 1)
    {
        this->curr_dist += (target - robot_traj[robot_traj.size() - 2]).Length();
    }
    else
    {
        this->curr_dist += target.Length();
    }

    while (this->curr_ind < (robot_traj.size() - 1) && this->curr_dist > dist)
    {
        this->curr_dist -= (robot_traj[this->curr_ind + 1] - robot_traj[this->curr_ind]).Length();
        this->curr_ind++;
    }

    this->updated_pos.X() = robot_traj[this->curr_ind].X();
    this->updated_pos.Y() = robot_traj[this->curr_ind].Y();
    this->heading = target - this->updated_pos;
    this->UpdateModel();
}

//VEHICLE CLASS

Vehicle::Vehicle(gazebo::physics::ActorPtr _actor,
                 double _mass,
                 double _max_force,
                 double _max_speed,
                 ignition::math::Pose3d initial_pose,
                 ignition::math::Vector3d initial_velocity,
                 std::vector<gazebo::physics::EntityPtr> objects,
                 double _obstacle_margin,
                 double _actor_margin)
{

    this->actor = _actor;
    this->mass = _mass;
    this->max_force = _max_force;
    this->max_speed = _max_speed;
    this->slow_factor = 1.0;
    this->pose = initial_pose;
    this->curr_target = initial_pose.Pos();
    this->velocity = initial_velocity;
    this->acceleration = 0;
    this->all_objects = objects;
    this->height = initial_pose.Pos().Z();
    this->obstacle_margin = _obstacle_margin;
    this->actor_margin = _actor_margin;

    this->flow_force = ignition::math::Vector3d(0, 0, 0);

    std::map<std::string, gazebo::common::SkeletonAnimation *>::iterator it;
    std::map<std::string, gazebo::common::SkeletonAnimation *> skel_anims = this->actor->SkeletonAnimations();

    for (it = skel_anims.begin(); it != skel_anims.end(); it++)
    {
        this->trajectories[it->first] = std::make_shared<gazebo::physics::TrajectoryInfo>();
        this->trajectories[it->first]->type = it->first;
        this->trajectories[it->first]->duration = 1.0;
    }

    this->actor->SetCustomTrajectory(this->trajectories["walking"]);
}

void Vehicle::OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects)
{
    this->Seek(this->curr_target);
}

void Vehicle::OnPoseUpdate(const gazebo::common::UpdateInfo &_info,
                           double dt,
                           std::vector<gazebo::physics::EntityPtr> objects)
{
    // Update
    UpdatePosition(dt);
    // UpdatePositionContactObstacles(objects, dt);
    UpdateModel();
}

void Vehicle::OnPoseReprod(const gazebo::common::UpdateInfo &_info,
                           double dt,
                           std::vector<gazebo::physics::EntityPtr> objects,
                           ignition::math::Pose3d new_pose0)
{
    // Set the new pose
    this->pose.Pos() = new_pose0.Pos();
    this->pose.Rot() = new_pose0.Rot();

    // Set the velocity (in case any other part of the code needs it)
    this->velocity = (new_pose0.Pos() - this->pose.Pos()) / dt;

    UpdateModel();
}

gazebo::physics::ActorPtr Vehicle::GetActor()
{
    return this->actor;
}

void Vehicle::UpdateModel()
{
    double distance_travelled = (this->pose.Pos() - this->actor->WorldPose().Pos()).Length();
    this->actor->SetWorldPose(this->pose, true, true);
    this->actor->SetScriptTime(this->actor->ScriptTime() + (distance_travelled * this->animation_factor));
}

void Vehicle::ApplyForce(ignition::math::Vector3d force)
{
    this->acceleration += (force / this->mass);
}

void Vehicle::UpdatePosition(double dt)
{
    this->velocity += this->acceleration * dt;

    double current_max_speed = this->max_speed * this->slow_factor;

    if (this->velocity.Length() > current_max_speed)
    {
        this->velocity.Normalize();
        this->velocity *= current_max_speed;
    }

    ignition::math::Vector3d direction = this->velocity;

    direction.Normalize();

    double dir_yaw = atan2(direction.Y(), direction.X());

    double current_yaw = this->pose.Rot().Yaw();

    ignition::math::Angle yaw_diff = dir_yaw - current_yaw + IGN_PI_2;

    yaw_diff.Normalize();

    if (this->velocity.Length() < 10e-2)
    {
        yaw_diff = 0;
    }

    this->pose.Pos() += this->velocity * dt;

    this->pose.Rot() = ignition::math::Quaterniond(IGN_PI_2, 0, current_yaw + yaw_diff.Radian() * 0.1);

    this->acceleration = 0;
}

/*
*   CustomAvoidObstacles: FLowFollowers can use the flow value function to know where obstacles are and avoid them
*   This function uses the current acceleration to determine if a collision is going to happen and correct it accordingly
*/
void Vehicle::UpdatePositionContactObstacles(std::vector<gazebo::physics::EntityPtr> &objects, double dt)
{

    // First predict future to see if there is contact
    // ***********************************************

    double current_max_speed = max_speed * slow_factor;

    // Predict the velocity and next position of the actor
    auto predicted_velocity = this->velocity + acceleration * dt;
    if (predicted_velocity.Length() > current_max_speed)
    {
        predicted_velocity.Normalize();
        predicted_velocity *= current_max_speed;
    }

    // Predict next position
    auto next_pos = this->pose.Pos() + predicted_velocity * dt;

    // Correct the next position to avoid getting into an obstacle
    // ***********************************************************

    // Find the closest obstacle
    bool inside_obstacle = false;
    ignition::math::Vector3d min_normal(0, 0, 0);
    double min_dist = 10e9;
    for (gazebo::physics::EntityPtr object : objects)
    {

        // Get object bounding box
        ignition::math::Box box = object->BoundingBox();

        // Check height for doors
        double min_z = std::min(box.Min().Z(), box.Max().Z());
        if (min_z > 1.5)
            continue;

        // Get the vector from closest box point to the current pos
        auto next_normal = utilities::min_box_repulsive_vector(next_pos, box);

        // Check if the person would arrive inside an object
        if (utilities::inside_box(box, next_pos))
        {
            inside_obstacle = true;
            break;
        }

        // Otherwise consider the closest object the person would arrive to
        if (next_normal.Length() < min_dist)
        {
            min_normal = next_normal;
            min_dist = next_normal.Length();
        }
    }

    // Get the actor at a distance of obstacle_margin from the nearest obstacle
    double dist = min_normal.Length();
    min_normal.Normalize();
    if (inside_obstacle)
        next_pos += min_normal * (-obstacle_margin - dist);
    else if (dist < obstacle_margin)
        next_pos += min_normal * (obstacle_margin - dist);

    // Back propgate this new position to speed
    // ****************************************

    this->velocity = (next_pos - this->pose.Pos()) / dt;
    this->pose.Pos() = next_pos;

    // Finish with the orientation update
    // **********************************

    // Get new direction
    ignition::math::Vector3d direction = this->velocity;
    direction.Normalize();

    // Get the diff in yaw
    double dir_yaw = atan2(direction.Y(), direction.X());
    double current_yaw = this->pose.Rot().Yaw();
    ignition::math::Angle yaw_diff = dir_yaw - current_yaw + IGN_PI_2;

    // Safe check yaw diff
    yaw_diff.Normalize();
    if (this->velocity.Length() < 10e-2)
        yaw_diff = 0;

    // Update pose
    this->pose.Rot() = ignition::math::Quaterniond(IGN_PI_2, 0, current_yaw + yaw_diff.Radian() * 0.1);

    // Reset acceleration
    this->acceleration = 0;
}

void Vehicle::Seek(ignition::math::Vector3d target, double weight)
{

    ignition::math::Vector3d desired_v = target - this->pose.Pos();
    desired_v.Normalize();
    desired_v *= this->max_speed * this->slow_factor;

    ignition::math::Vector3d steer = desired_v - this->velocity;

    steer *= weight;
    if (steer.Length() > this->max_force)
    {
        steer.Normalize();
        steer *= this->max_force;
    }

    ApplyForce(steer);
    // if(steer.Length() != 0){
    //     std::cout << this->GetName() << " steer force: " << steer << std::endl;
    // }
}

void Vehicle::Arrival(ignition::math::Vector3d target, double weight)
{

    double current_max_speed = this->max_speed * this->slow_factor;

    ignition::math::Vector3d desired_v = target - this->pose.Pos();
    double dist = desired_v.Length();
    desired_v.Normalize();

    if (dist < this->slowing_distance)
    {
        desired_v *= (dist / this->slowing_distance) * current_max_speed;
    }
    else
    {
        desired_v *= current_max_speed;
    }

    ignition::math::Vector3d steer = desired_v - this->velocity;

    steer *= weight;

    if (steer.Length() > this->max_force)
    {
        steer.Normalize();
        steer *= this->max_force;
    }

    ApplyForce(steer);
}

void Vehicle::AvoidObstacles(std::vector<gazebo::physics::EntityPtr> objects)
{
    ignition::math::Vector3d boundary_force = ignition::math::Vector3d(0, 0, 0);

    for (gazebo::physics::EntityPtr object : objects)
    {
        // Ignore robot
        if (object->GetName() == "jackal")
            continue;

        // Get object bounding box
        ignition::math::Box box = object->BoundingBox();

        // // inflate the box slightly => Useless as we crop dist after
        // double inflate = 0.1;
        // box.Min().X() -= inflate;
        // box.Max().X() += inflate;
        // box.Min().Y() -= inflate;
        // box.Max().Y() += inflate;

        // Check height for doors
        double min_z = std::min(box.Min().Z(), box.Max().Z());
        if (min_z > 1.5)
            continue;

        ignition::math::Vector3d min_normal;
        double dist;

        // If the person has somehow arrived inside an object, force to leave
        if (utilities::inside_box(box, this->pose.Pos()))
        {
            // Apply the max force to repel from obstacle center
            min_normal = this->pose.Pos() - box.Center();
            min_normal.Z() = 0;
            dist = 0.0;
            //std::cout << " ----> WARNING: " << this->GetName()  << " is in an obstacle" << std::endl;
        }
        else
        {
            // Get the vector from closest box point to the current pos
            min_normal = utilities::min_box_repulsive_vector(this->pose.Pos(), box);
            min_normal.Z() = 0;
            dist = min_normal.Length();
        }

        // Compute boundary force
        dist = std::max(0.0, dist - this->obstacle_margin); //acounting for the radius of the person
        if (dist < this->obstacle_margin)
        {
            min_normal.Normalize();
            double exp_term = dist / (obstacle_margin / 2);
            exp_term *= exp_term;
            boundary_force += min_normal * (0.8 * max_force * exp(-exp_term));
        }
    }

    // Clip max force
    if (boundary_force.Length() > this->max_force)
    {
        boundary_force.Normalize();
        boundary_force *= this->max_force;
    }

    // Apply force
    boundary_force.Z() = 0;
    this->ApplyForce(boundary_force);
}

void Vehicle::AvoidRobot(std::vector<gazebo::physics::EntityPtr> objects, double robot_margin_factor, bool slowing)
{
    double margin = robot_margin_factor * this->actor_margin;

    ignition::math::Vector3d boundary_force = ignition::math::Vector3d(0, 0, 0);

    for (gazebo::physics::EntityPtr object : objects)
    {
        if (object->GetName() == "jackal")
        {
            ignition::math::Box box = object->BoundingBox();

            ignition::math::Vector3d rad = this->pose.Pos() - box.Center();
            rad.Z() = 0;
            double dist = rad.Length();
            dist = std::max(0.0, dist - 0.2); // acounting for the radius of the person

            // Update slowing factor and corresponding force if slowing happens
            double force_factor = 0.4;
            if (slowing)
            {
                slow_factor = std::max(0.5, std::min(0.4 + dist / margin, 1.0));
                force_factor = 0.3;
            }

            // Compute boundary force
            if (dist < margin)
            {
                rad.Normalize();
                double exp_term = dist / (margin / 2);
                exp_term *= exp_term;
                boundary_force += rad * (force_factor * max_force * exp(-exp_term));
            }
        }
    }

    // Clip max force
    if (boundary_force.Length() > this->max_force)
    {
        boundary_force.Normalize();
        boundary_force *= this->max_force;
    }

    // Apply force
    this->ApplyForce(boundary_force);
}

void Vehicle::AvoidActors(std::vector<boost::shared_ptr<Vehicle>> vehicles)
{

    ignition::math::Vector3d steer = ignition::math::Vector3d(0, 0, 0);
    for (int i = 0; i < (int)vehicles.size(); i++)
    {
        auto other = vehicles[i]->GetActor();
        if (other == this->actor)
        {
            continue;
        }

        ignition::math::Vector3d this_pos = this->pose.Pos();
        this_pos.Z() = 0;
        ignition::math::Vector3d other_pos = other->WorldPose().Pos();
        other_pos.Z() = 0;

        ignition::math::Vector3d diff_pos = this_pos - other_pos;
        double dist = diff_pos.Length();
        double dist_cropped = std::max(0.0, dist - 0.5); //acounting for the radius of the two person

        ignition::math::Vector3d rep_force(0, 0, 0);
        if (1e-6 < dist && dist < this->actor_margin)
        {
            // Normalize force vector
            ignition::math::Vector3d rad(diff_pos);
            rad.Normalize();

            // Force value defined by gaussian
            double exp_term = dist_cropped / (actor_margin / 4);
            exp_term *= exp_term;
            rep_force = rad * exp(-exp_term) * 0.4 * max_force;

            ignition::math::Vector3d other_flow_force = vehicles[i]->flow_force;
            if (flow_force.Length() > 0 && other_flow_force.Length() > 0)
            {
                auto self_direction(flow_force);
                self_direction.Normalize();
                double self_dist = self_direction.Dot(-diff_pos);
                double self_term = (self_dist - actor_margin / 4) / (actor_margin / 3);
                self_term *= self_term;

                auto other_direction(other_flow_force);
                other_direction.Normalize();
                double other_dist = other_direction.Dot(diff_pos);
                double other_term = (other_dist - actor_margin / 4) / (actor_margin / 3);
                other_term *= other_term;

                // Ensure right/left orientation of the tangential force
                auto self_angle = acos(ignition::math::clamp(self_dist / diff_pos.Length(), -1.0, 1.0));
                auto other_angle = acos(ignition::math::clamp(other_dist / diff_pos.Length(), -1.0, 1.0));
                ignition::math::Vector3d self_tan(diff_pos[1], -diff_pos[0], 0);
                self_tan.Normalize();
                if (self_tan.Dot(self_direction) < 0)
                    self_tan *= -1;
                ignition::math::Vector3d other_tan(self_tan);
                if (other_tan.Dot(other_direction) < 0)
                    other_tan *= -1;
                if (self_tan.Dot(other_tan) > 0)
                {
                    if (self_angle < other_angle)
                        self_tan *= -1;
                }

                // Tangential force value
                double tangential_v = 0.5 * flow_force.Length() * exp(-other_term) * exp(-self_term);
                rep_force += self_tan * tangential_v;
            }
        }

        steer += rep_force;
    }

    // Clamp the value
    if (steer.Length() > this->max_force)
    {
        steer.Normalize();
        steer *= this->max_force;
    }
    steer.Z() = 0;

    showed_force = steer;
    this->ApplyForce(steer);
    // if(steer.Length() != 0){
    //     std::cout << this->GetName() << " avoid actor force: " << steer << std::endl;
    // }
}

void Vehicle::SetAllObjects(std::vector<gazebo::physics::EntityPtr> objects)
{
    this->all_objects = objects;
}

ignition::math::Pose3d Vehicle::GetPose()
{
    return this->pose;
}

ignition::math::Vector3d Vehicle::GetVelocity()
{
    return this->velocity;
}

std::string Vehicle::GetName()
{
    return this->actor->GetName();
}

bool Vehicle::IsStill()
{
    return this->still;
}

//----------------------------------------------------------------------------------------------------------------------------------------------
// WANDERER

void Wanderer::OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects)
{

    this->SetNextTarget();
    this->Seek(this->curr_target);
    this->AvoidActors(vehicles);
    this->AvoidObstacles(objects);
    this->AvoidRobot(objects);
}

void Wanderer::SetNextTarget()
{
    this->curr_theta += ignition::math::Rand::DblUniform(-this->rand_amp, this->rand_amp); //TODO: perlin noise

    auto dir = this->velocity;

    dir.Normalize();

    dir *= 2;

    auto offset = ignition::math::Vector3d(1, 0, 0);

    auto rotation = ignition::math::Quaterniond(0, 0, this->curr_theta);

    offset = rotation.RotateVector(offset);

    this->curr_target = this->pose.Pos() + dir + offset;
    this->curr_target.Z() = this->height;
}

//----------------------------------------------------------------------------------------------------------------------------------------------
// RANDOM WALKER

void RandomWalker::OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects)
{

    if ((this->pose.Pos() - this->curr_target).Length() < this->arrival_distance)
    {
        this->SetNextTarget(this->all_objects);
    }

    this->Seek(this->curr_target);
    this->AvoidActors(vehicles);
    this->AvoidObstacles(objects); //TODO: make sure this is safe here
    this->AvoidRobot(objects);
}

void RandomWalker::SetNextTarget(std::vector<gazebo::physics::EntityPtr> objects)
{
    bool target_found = false;

    while (!target_found)
    {

        auto dir = this->velocity;
        if (dir.Length() < 1e-6)
        {
            dir = ignition::math::Vector3d(ignition::math::Rand::DblUniform(-1, 1), ignition::math::Rand::DblUniform(-1, 1), 0);
        }
        dir.Normalize();
        auto rotation = ignition::math::Quaterniond::EulerToQuaternion(0, 0, ignition::math::Rand::DblUniform(-3, 3));
        dir = rotation.RotateVector(dir);
        dir *= 10000;
        auto ray = ignition::math::Line3d(this->pose.Pos().X(), this->pose.Pos().Y(), this->pose.Pos().X() + dir.X(), this->pose.Pos().Y() + dir.Y());
        ignition::math::Vector3d closest_intersection;
        ignition::math::Line3d closest_edge;
        double min_dist = 100000;

        for (auto object : objects)
        {

            auto box = object->BoundingBox();

            std::vector<ignition::math::Line3d> edges = utilities::get_box_edges(box);

            for (ignition::math::Line3d edge : edges)
            {

                ignition::math::Vector3d test_intersection;

                if (ray.Intersect(edge, test_intersection))
                { //if the ray intersects the boundary
                    ignition::math::Vector3d zero_z = this->pose.Pos();
                    zero_z.Z() = 0;
                    double dist_to_int = (test_intersection - zero_z).Length();
                    if (dist_to_int < min_dist)
                    {
                        min_dist = dist_to_int;
                        closest_intersection = test_intersection;
                        closest_edge = edge;
                    }
                }
            }
        }

        auto zero_z = this->pose.Pos();
        zero_z.Z() = 0;
        auto final_ray = closest_intersection - zero_z;
        auto v_to_add = final_ray * ignition::math::Rand::DblUniform(0.1, 0.9);

        ignition::math::Vector3d normal;
        if (utilities::get_normal_to_edge(this->pose.Pos(), closest_edge, normal) && (normal.Length() < this->obstacle_margin))
        {
            continue;
        }

        if ((final_ray - v_to_add).Length() < (this->obstacle_margin))
        {
            v_to_add.Normalize();
            auto small_subtraction = (v_to_add * this->obstacle_margin) * 2;
            v_to_add = final_ray - small_subtraction;
            if (small_subtraction.Length() > final_ray.Length())
            {
                v_to_add *= 0;
            }
        }

        this->curr_target = v_to_add + this->pose.Pos();
        this->curr_target.Z() = this->height;
        target_found = true;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------------------
// BOID

Boid::Boid(gazebo::physics::ActorPtr _actor,
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
           double _actor_margin)
    : Vehicle(_actor, _mass, _max_force, _max_speed, initial_pose, initial_velocity, objects, _obstacle_margin, _actor_margin)
{
    this->weights[ALI] = _alignement;
    this->weights[COH] = _cohesion;
    this->weights[SEP] = _separation;

    this->FOV_angle = angle;
    this->FOV_radius = radius;
}

void Boid::OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects)
{

    this->Alignement(dt, vehicles);
    this->Cohesion(vehicles);
    this->Separation(vehicles);
    this->AvoidObstacles(objects);
}

void Boid::Separation(std::vector<boost::shared_ptr<Vehicle>> vehicles)
{
    ignition::math::Vector3d steer = ignition::math::Vector3d(0, 0, 0);
    for (auto other : vehicles)
    {
        if (other->GetName() == this->GetName())
        {
            continue;
        }
        auto this_pos = this->pose.Pos();
        this_pos.Z() = 0;
        auto other_pos = other->GetPose().Pos();
        other_pos.Z() = 0;
        auto rad = this_pos - other_pos;
        double dist = rad.Length();

        auto dir = this->velocity;
        dir.Normalize();
        rad.Normalize();
        double angle = std::acos(rad.Dot(dir));

        if (dist < this->obstacle_margin && dist > 0)
        {
            rad.Normalize();
            rad /= dist;
            steer += rad;
        }
    }

    if ((steer * this->weights[SEP]).Length() > this->max_force)
    {
        steer.Normalize();
        steer *= this->max_force;
    }
    else
    {
        steer *= this->weights[SEP];
    }
    steer.Z() = 0;
    this->ApplyForce(steer);
}

void Boid::Alignement(double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles)
{
    auto vel_sum = ignition::math::Vector3d(0, 0, 0);
    int count = 0;

    for (auto other : vehicles)
    {
        if (other->GetName() == this->GetName())
        {
            continue;
        }
        auto this_pos = this->pose.Pos();
        this_pos.Z() = 0;
        auto other_pos = other->GetPose().Pos();
        other_pos.Z() = 0;
        auto r = this_pos - other_pos;

        if (r.Length() < this->FOV_radius)
        {

            auto dir = this->velocity;
            dir.Normalize();
            r.Normalize();
            double angle = std::acos(r.Dot(dir));

            if (angle < this->FOV_angle / 2)
            {
                vel_sum += other->GetVelocity();
                count++;
            }
        }
    }

    if (count > 0)
    {

        // Implement Reynolds: Steering = Desired - Velocity
        vel_sum.Normalize();
        vel_sum *= this->max_speed;
        vel_sum -= this->velocity;

        if ((vel_sum * this->weights[SEP]).Length() > this->max_force)
        {
            vel_sum.Normalize();
            vel_sum *= this->max_force;
        }
        else
        {
            vel_sum *= this->weights[SEP];
        }
        vel_sum.Z() = 0;
        this->ApplyForce(vel_sum);
    }
}

void Boid::Cohesion(std::vector<boost::shared_ptr<Vehicle>> vehicles)
{
    auto sum_pos = ignition::math::Vector3d(0, 0, 0);
    int count = 0;
    auto this_pos = this->pose.Pos();
    this_pos.Z() = 0;

    for (auto other : vehicles)
    {
        if (other->GetName() == this->GetName())
        {
            continue;
        }
        auto other_pos = other->GetPose().Pos();
        other_pos.Z() = 0;
        auto rad = this_pos - other_pos;
        double dist = rad.Length();
        auto dir = this->velocity;
        dir.Normalize();
        rad.Normalize();
        double angle = std::acos(rad.Dot(dir));
        if (dist < this->FOV_radius && angle < this->FOV_angle / 2)
        {
            sum_pos += other_pos;
            count++;
        }
    }

    if (count > 0)
    {
        sum_pos.Z() = this->height;
        this->Seek(sum_pos, this->weights[COH]);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------------------
/// STANDER

Stander::Stander(gazebo::physics::ActorPtr _actor,
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
                 int start_mode)
    : Wanderer(_actor, _mass, _max_force, _max_speed, initial_pose, initial_velocity, objects, _obstacle_margin, _actor_margin)
{

    this->standing_duration = std::max(0.0, _standing_duration + ignition::math::Rand::DblUniform(-0.5, 0.5));
    this->walking_duration = std::max(0.0, _walking_duration + ignition::math::Rand::DblUniform(-0.5, 0.5));

    if (start_mode == 2)
    {
        this->standing = (bool)ignition::math::Rand::IntUniform(0, 1);
    }
    else if (start_mode == 1)
    {
        this->standing = false;
    }
    else
    {
        this->standing = true;
    }

    if (walking_duration <= 0)
    {
        this->never_walk = true;
    }
    if (standing)
    {
        this->actor->SetCustomTrajectory(this->trajectories["standing"]);
    }
    else
    {
        this->actor->SetCustomTrajectory(this->trajectories["walking"]);
    }
    this->UpdatePosition(0.1);
    this->actor->SetWorldPose(this->pose, true, true);
    this->actor->SetScriptTime(this->actor->ScriptTime());
}

void Stander::UpdateModel(double dt)
{

    if (this->standing)
    {
        this->actor->SetWorldPose(this->pose, true, true);
        this->actor->SetScriptTime(this->actor->ScriptTime() + dt * this->animation_factor);
    }
    else
    {
        double distance_travelled = (this->pose.Pos() - this->actor->WorldPose().Pos()).Length();
        this->actor->SetWorldPose(this->pose, true, true);
        this->actor->SetScriptTime(this->actor->ScriptTime() + (distance_travelled * this->animation_factor));
    }
}

void Stander::OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects)
{

    if (this->standing)
    {
        this->still = true;
        if (!this->never_walk && (_info.simTime - this->standing_start).Double() >= this->standing_duration)
        {
            this->standing = false;
            this->walking_start = _info.simTime;
            this->actor->SetCustomTrajectory(this->trajectories["walking"]);
            this->velocity = 0;
            this->standing_duration += ignition::math::Rand::DblUniform(-0.5, 0.5);
            if (this->standing_duration <= 0)
            {
                this->standing_duration = 0;
            }
        }
    }
    else
    {
        this->still = false;
        this->SetNextTarget();
        this->Seek(this->curr_target);
        this->AvoidActors(vehicles);
        this->AvoidObstacles(objects);
        this->AvoidRobot(objects);

        if ((_info.simTime - this->walking_start).Double() >= this->walking_duration)
        {
            this->standing = true;
            this->standing_start = _info.simTime;
            this->actor->SetCustomTrajectory(this->trajectories["standing"]);
            this->walking_duration += ignition::math::Rand::DblUniform(-0.5, 0.5);
            if (this->walking_duration <= 0)
            {
                this->walking_duration = 0;
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------------------

Sitter::Sitter(gazebo::physics::ActorPtr _actor,
               std::string _chair_name,
               std::vector<gazebo::physics::EntityPtr> objects,
               double height)
    : Vehicle(_actor, 1, 1, 1, ignition::math::Pose3d(100, 100, 0.5, 0, 0, 0), ignition::math::Vector3d(0, 0, 0), objects, 0.2, 1.0)
{
    this->chair_name = _chair_name;
    bool found = false;
    this->still = true;
    for (auto model : this->all_objects)
    {
        if (model->GetParent()->GetParent()->GetName() == this->chair_name)
        {
            this->chair = boost::static_pointer_cast<gazebo::physics::Entity>(model->GetParent()->GetParent());
            found = true;
            break;
        }
    }

    if (found)
    {

        this->pose = this->chair->WorldPose();
        this->pose.Pos().Z() = height;
        this->pose.Rot() = ignition::math::Quaterniond(1.15, 0, this->chair->WorldPose().Rot().Yaw());
        this->actor->SetCustomTrajectory(this->trajectories["sitting"]);
        this->actor->SetWorldPose(this->pose, true, true);
        this->actor->SetScriptTime(this->actor->ScriptTime());
    }
}

void Sitter::OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects)
{
    return;
}

void Sitter::UpdateModel(double dt)
{

    this->actor->SetWorldPose(this->pose, true, true);
    this->actor->SetScriptTime(this->actor->ScriptTime() + dt * this->animation_factor);
}

//----------------------------------------------------------------------------------------------------------------------------------------------

Follower::Follower(gazebo::physics::ActorPtr _actor,
                   double _mass,
                   double _max_force,
                   double _max_speed,
                   ignition::math::Pose3d initial_pose,
                   ignition::math::Vector3d initial_velocity,
                   std::vector<gazebo::physics::EntityPtr> objects,
                   std::string _leader_name, bool blocking,
                   double _obstacle_margin,
                   double _actor_margin)
    : Vehicle(_actor, _mass, _max_force, _max_speed, initial_pose, initial_velocity, objects, _obstacle_margin, _actor_margin)
{

    this->leader_name = _leader_name;
    this->blocking = blocking;
    if (this->blocking)
    {
        this->rand_angle_offset = ignition::math::Rand::DblUniform(-1, 1);
    }
    else
    {
        this->rand_angle_offset = ignition::math::Rand::DblUniform(-2.5, 2.5);
    }
}

void Follower::LoadLeader(gazebo::physics::EntityPtr leader)
{
    this->leader = leader;
    this->last_leader_pose = this->leader->WorldPose();
}

void Follower::SetNextTarget(double dt)
{

    if (this->leader == nullptr)
    {
        this->curr_target = this->pose.Pos();
        return;
    }
    //auto leader_dir = this->leader->GetVelocity();
    auto leader_pose = this->leader->WorldPose();
    auto leader_dir = leader_pose.Pos() - this->last_leader_pose.Pos();

    auto rotation = ignition::math::Quaterniond(0, 0, leader_pose.Rot().Yaw() + this->rand_angle_offset);

    if (leader_dir.Length() < 10e-6)
    {

        auto offset = rotation.RotateVector(ignition::math::Vector3d(this->rand_radius, 0, 0));
        this->curr_target = leader_pose.Pos();

        if (this->blocking)
        {
            this->curr_target += offset;
        }
        else
        {
            this->curr_target -= offset;
        }

        this->curr_target.Z() = this->pose.Pos().Z();

        return;
    }

    leader_dir.Normalize();

    // if we find ourselves in front of the leader, steer laterally away from the leaders path

    if (!this->blocking)
    {
        auto front_edge = ignition::math::Line3d(leader_pose.Pos(), leader_pose.Pos() + leader_dir);

        ignition::math::Vector3d normal;

        if (utilities::get_normal_to_edge(this->pose.Pos(), front_edge, normal))
        {
            if (normal.Length() < this->obstacle_margin)
            {
                auto mag = normal.Length();
                if (mag == 0)
                {
                    mag = 10e-9;
                }
                normal.Normalize();

                normal *= 1 / (mag * mag);
                if (normal.Length() > this->max_force)
                {
                    normal.Normalize();
                    normal *= this->max_force;
                }
                this->ApplyForce(normal);
            }
        }
    }

    leader_dir *= this->rand_radius;
    rotation = ignition::math::Quaterniond(0, 0, this->rand_angle_offset);
    leader_dir = rotation.RotateVector(leader_dir);
    this->curr_target = leader_pose.Pos();

    if (this->blocking)
    {
        this->curr_target += leader_dir / 2;
    }
    else
    {
        this->curr_target -= leader_dir / 2;
    }

    this->curr_target.Z() = this->pose.Pos().Z();
    this->last_leader_pose = leader_pose;
}

void Follower::OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects)
{

    this->SetNextTarget(dt);
    this->Arrival(this->curr_target);

    this->AvoidActors(vehicles);
    this->AvoidObstacles(objects);
    this->AvoidRobot(objects);
}

//----------------------------------------------------------------------------------------------------------------------------------------------

void PathFollower::Follow()
{

    if ((this->pose.Pos() - this->curr_target).Length() < this->arrival_distance)
    {

        this->path_ind++;

        if (this->path_ind >= this->curr_path.size())
        {

            this->RePath();
        }

        this->curr_target = this->curr_path[this->path_ind];
        this->curr_target.Z() = this->height;
        //std::cout << "TARGET: " << this->curr_target << std::endl;
    }

    this->Seek(this->curr_target, 7);
}

PathFollower::PathFollower(gazebo::physics::ActorPtr _actor,
                           double _mass,
                           double _max_force,
                           double _max_speed,
                           ignition::math::Pose3d initial_pose,
                           ignition::math::Vector3d initial_velocity,
                           std::vector<gazebo::physics::EntityPtr> objects,
                           boost::shared_ptr<Costmap> costmap,
                           boost::shared_ptr<std::vector<ignition::math::Pose3d>> digits_coordinates,
                           double _obstacle_margin,
                           double _actor_margin)
    : Wanderer(_actor, _mass, _max_force, _max_speed, initial_pose, initial_velocity, objects, _obstacle_margin, _actor_margin)
{
    this->costmap = costmap;
    this->digits_coordinates = digits_coordinates;
    this->path_ind = 0;

    this->RePath();
    //this->costmap->AStar(this->pose.Pos(), ignition::math::Vector3d(0,-10,this->height), this->curr_path);
    //this->curr_target = this->curr_path[this->path_ind];
    //std::cout << "size: " << this->curr_path.size() << std::endl;
}

void PathFollower::OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects)
{

    this->Follow();
    this->AvoidActors(vehicles);
    this->AvoidObstacles(objects);
    this->AvoidRobot(objects);
}

void PathFollower::RePath()
{
    this->path_ind = 1;

    ignition::math::Vector3d next_goal;
    this->curr_path.clear();
    do
    {
        if (this->digits_coordinates->size() == 0)
        {
            next_goal = this->costmap->RandPos();
        }
        else
        {
            auto &digits_ref = *this->digits_coordinates;
            next_goal = digits_ref[rand() % this->digits_coordinates->size()].Pos();
        }
        // std::cout << "GOAL: " << next_goal << std::endl;
    } while (!this->costmap->AStar(this->pose.Pos(), next_goal, this->curr_path, false));
    //
    //while (!this->costmap->FindPath(this->pose.Pos(), next_goal, this->curr_path));
    // std::string costmap_string;
    // costmap_string = this->costmap->ToString();
    // std::ofstream out("output.txt");
    // out << costmap_string;
}

//----------------------------------------------------------------------------------------------------------------------------------------------

////////////////////////////////////
// ExtendedSocialForce_Actor
void ExtendedSocialForce_Actor::OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects)
{

    if ((this->pose.Pos() - this->curr_target).Length() < this->arrival_distance)
    {
        this->SetNextTarget(this->all_objects); //TODO: Include the relaxation time
    }

    this->Seek(this->curr_target); //TODO: include weight to see the effect
    this->ExtendedSFHuman(vehicles);
    this->ExtendedSFRobot(objects);
    this->ExtendedSFObstacle(objects);
}

void ExtendedSocialForce_Actor::SetNextTarget(std::vector<gazebo::physics::EntityPtr> objects)
{
    bool target_found = false;

    while (!target_found)
    {

        auto dir = this->velocity;
        if (dir.Length() < 1e-6)
        {
            dir = ignition::math::Vector3d(ignition::math::Rand::DblUniform(-1, 1), ignition::math::Rand::DblUniform(-1, 1), 0);
        }
        dir.Normalize();
        auto rotation = ignition::math::Quaterniond::EulerToQuaternion(0, 0, ignition::math::Rand::DblUniform(-3, 3));
        dir = rotation.RotateVector(dir);
        dir *= 10000;
        auto ray = ignition::math::Line3d(this->pose.Pos().X(), this->pose.Pos().Y(), this->pose.Pos().X() + dir.X(), this->pose.Pos().Y() + dir.Y());
        ignition::math::Vector3d closest_intersection;
        ignition::math::Line3d closest_edge;
        double min_dist = 100000;

        for (auto object : objects)
        {

            auto box = object->BoundingBox();

            std::vector<ignition::math::Line3d> edges = utilities::get_box_edges(box);

            for (ignition::math::Line3d edge : edges)
            {

                ignition::math::Vector3d test_intersection;

                if (ray.Intersect(edge, test_intersection))
                { //if the ray intersects the boundary
                    ignition::math::Vector3d zero_z = this->pose.Pos();
                    zero_z.Z() = 0;
                    double dist_to_int = (test_intersection - zero_z).Length();
                    if (dist_to_int < min_dist)
                    {
                        min_dist = dist_to_int;
                        closest_intersection = test_intersection;
                        closest_edge = edge;
                    }
                }
            }
        }

        auto zero_z = this->pose.Pos();
        zero_z.Z() = 0;
        auto final_ray = closest_intersection - zero_z;
        auto v_to_add = final_ray * ignition::math::Rand::DblUniform(0.1, 0.9);

        ignition::math::Vector3d normal;
        if (utilities::get_normal_to_edge(this->pose.Pos(), closest_edge, normal) && (normal.Length() < this->obstacle_margin))
        {
            continue;
        }

        if ((final_ray - v_to_add).Length() < (this->obstacle_margin))
        {
            v_to_add.Normalize();
            auto small_subtraction = (v_to_add * this->obstacle_margin) * 2;
            v_to_add = final_ray - small_subtraction;
            if (small_subtraction.Length() > final_ray.Length())
            {
                v_to_add *= 0;
            }
        }

        this->curr_target = v_to_add + this->pose.Pos();
        this->curr_target.Z() = this->height;
        target_found = true;
    }
}

/**
 * Define the social interaction forces (social and physical) for human interactions according to 
 *      Anvari et al., IROS 2020. All the constant parameter used in calculation are calculated using
 *      the paper recommended values. They should be calibrated for different interaction types, 
 *      and represent a starting point for this analysis.
 * 
 * @param vehicles pointer to all actors in the scene 
 * 
 */
void ExtendedSocialForce_Actor::ExtendedSFHuman(std::vector<boost::shared_ptr<Vehicle>> vehicles)
{
    //ignition::math::Rand::Seed(42);
    ignition::math::Vector3d steer = ignition::math::Vector3d(0, 0, 0);
    double interaction_strength = 0.8 + ignition::math::Rand::DblUniform(-0.1, 0.1);
    double interaction_range = 1 + ignition::math::Rand::DblUniform(-0.25, 0.25);

    for (int i = 0; i < (int)vehicles.size(); i++)
    {
        auto other = vehicles[i]->GetActor();
        if (other == this->actor)
        {
            continue;
        }
        ignition::math::Vector3d this_pos = this->pose.Pos();
        this_pos.Z() = 0;
        ignition::math::Vector3d other_pos = other->WorldPose().Pos();
        other_pos.Z() = 0;
        ignition::math::Vector3d rad = this_pos - other_pos;
        double dist = rad.Length();
        dist = std::max(0.0, dist - 0.2); //acounting for the radius of the person

        auto dir = this->curr_target;
        dir.Normalize();
        rad.Normalize();
        double angle_rad_dir = std::acos(dir.Dot(rad));
        double SF_form_factor = 0.2 + 0.4 * (1 + std::cos(angle_rad_dir));
        rad *= 5 * interaction_strength * std::exp(-dist / interaction_range) * SF_form_factor;
        steer += rad;

        if (dist < this->actor_margin && dist > 0)
        { //Add physical force for when there is a collision
            rad /= dist;
            steer += rad;
        }

        if (steer.Length() > this->max_force)
        {
            steer.Normalize();
            steer *= this->max_force;
        }
        steer.Z() = 0;

        this->ApplyForce(steer);
    }
}

/*
SEE https://pdfs.semanticscholar.org/9fe0/8e8f61e1a173e7487ebc712bf429e3e9f213.pdf?_ga=2.196497262.372224845.1607942871-148799776.1607942871
FOR FORCES DESCRIPTION

WORK ON CONTACT FORCE
*/
void ExtendedSocialForce_Actor::ExtendedSFRobot(std::vector<gazebo::physics::EntityPtr> objects)
{
    //ignition::math::Rand::Seed(42);
    ignition::math::Vector3d boundary_force = ignition::math::Vector3d(0, 0, 0);
    double interaction_strength = 1.2;
    double interaction_range = 2.6;

    for (gazebo::physics::EntityPtr object : objects)
    {
        if (object->GetName() == "jackal")
        {

            //std::cout<<"applying robot force"<< std::endl;
            ignition::math::Box box = object->BoundingBox();

            ignition::math::Vector3d rad = this->pose.Pos() - box.Center();
            double dist = rad.Length();
            dist = std::max(0.0, dist - 0.2); //acounting for the radius of the person

            auto dir = this->curr_target;
            dir.Normalize();
            rad.Normalize();
            double angle_rad_dir = std::acos(dir.Dot(rad));
            double SF_form_factor = 0.2 + 0.4 * (1 + std::cos(angle_rad_dir));
            rad *= interaction_strength * std::exp(-dist / interaction_range) * SF_form_factor;

            if (dist < this->obstacle_margin)
            {
                rad *= 100; //multiply force if we're too close
                            // std::cout<<"Too close to the robot: " << this->GetName() <<std::endl;
            }
            boundary_force += rad;
        }
    }
    boundary_force.Z() = 0;
    this->ApplyForce(boundary_force);
}

/**
* Define the obstacle avoidance forces that act on humans in the scene according to the Extended
*       Social Force Model of Anvari et al., IROS 2020. All the constant parameter used in calculation 
*       are calculated using the paper recommended values. They should be calibrated for different 
*       interaction types, and represent a starting point for this analysis.
*
* @param objects pointer to all objects in the scene. 
**/

void ExtendedSocialForce_Actor::ExtendedSFObstacle(std::vector<gazebo::physics::EntityPtr> objects)
{
    ignition::math::Vector3d boundary_force = ignition::math::Vector3d(0, 0, 0);
    double interaction_strength = 0.8 + ignition::math::Rand::DblUniform(-0.1, 0.1);
    double interaction_range = 1 + ignition::math::Rand::DblUniform(-0.25, 0.25);

    for (gazebo::physics::EntityPtr object : objects)
    {
        ignition::math::Box box = object->BoundingBox();

        ignition::math::Vector3d rad = this->pose.Pos() - box.Center();
        double dist = rad.Length();
        dist = std::max(0.0, dist - 0.2); //acounting for the radius of the person

        auto dir = this->curr_target;
        dir.Normalize();
        rad.Normalize();
        double angle_rad_dir = std::acos(dir.Dot(rad));
        double SF_form_factor = 0.2 + 0.4 * (1 + std::cos(angle_rad_dir));
        rad *= interaction_strength * std::exp(-dist / interaction_range) * SF_form_factor;

        if (dist < this->obstacle_margin)
        {
            rad *= 100; //multiply force if we're too close
            std::cout << "Too close to the obstacle : " << this->GetName() << std::endl;
        }
        boundary_force += rad;
    }
    boundary_force.Z() = 0;
    this->ApplyForce(boundary_force);
}

//----------------------------------------------------------------------------------------------------------------------------------------------

// CUSTOM_WANDERER
/*
* Custom_Wanderer will follow the goals defined in custom_simulation_params.yaml. 
* They have the same behavior as a Wanderer otherwise. 
* TODO: Add options to change the type of behavior: sitter, extendedSF, custom: yield/no-yield behavior etc...
*/

Custom_Wanderer::Custom_Wanderer(gazebo::physics::ActorPtr _actor,
                                 double _mass,
                                 double _max_force,
                                 double _max_speed,
                                 ignition::math::Pose3d initial_pose,
                                 ignition::math::Vector3d initial_velocity,
                                 std::vector<gazebo::physics::EntityPtr> objects,
                                 std::map<std::string, double> _custom_actor_goal,
                                 std::vector<std::string> _vehicle_names,
                                 double _obstacle_margin,
                                 double _actor_margin)
    : Vehicle(_actor, _mass, _max_force, _max_speed, initial_pose, initial_velocity, objects, _obstacle_margin, _actor_margin)
{
    this->custom_actor_goal = _custom_actor_goal;
    this->vehicle_names = _vehicle_names;
}

void Custom_Wanderer::OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects)
{

    this->SetNextTarget(vehicles);
    this->Seek(this->curr_target);
    this->AvoidActors(vehicles);
    this->AvoidObstacles(objects);
    this->AvoidRobot(objects);
}

/*
* SetNextTarget sort all actor names, and send a custom goal to one Custom_Wanderer agent following the spawning order.
* The objective is to keep the same order as in params/custom_simulation_params.yaml; in order to send the corresponding
* goal_x, goal_y to each actor. 
*
* Using the custom_wanderer type implies that a custom_simulation_params file exists with the right structure.
* If there are more custom_wanderers than goal specified, the first goal (x_0,y_0) is passed to all the undefined actors. 
*/
void Custom_Wanderer::SetNextTarget(std::vector<boost::shared_ptr<Vehicle>> vehicles)
{
    int count(0);
    for (auto name : this->vehicle_names)
    {
        if (this->GetName() == name)
        {
            break;
        }
        if (name.find("actor_custom_") != std::string::npos)
        {
            count += 1;
        }
    }

    ignition::math::v4::Vector3d goal;
    if (this->custom_actor_goal.find("goal_x_" + std::to_string(count)) == custom_actor_goal.end())
    {
        goal.Set(this->custom_actor_goal["goal_x_0"], this->custom_actor_goal["goal_y_0"], 0);
    }
    else
    {
        goal.Set(this->custom_actor_goal["goal_x_" + std::to_string(count)], this->custom_actor_goal["goal_y_" + std::to_string(count)], 0);
    }

    this->curr_target = this->pose.Pos() + goal;
    this->curr_target.Z() = this->height;
}

//----------------------------------------------------------------------------------------------------------------------------------------------

FlowFollower::FlowFollower(gazebo::physics::ActorPtr _actor,
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
                           double _robot_slow)
    : Vehicle(_actor, _mass, _max_force, _max_speed, initial_pose, initial_velocity, objects, _obstacle_margin, _actor_margin)
{

    // Init variables
    // **************

    robot_margin = _robot_margin;
    robot_slow = _robot_slow;

    flow_fields = flow_fields0;
    current_flow = 0;
    distance_to_goal = 0;

    // Choose a valid initial flow to follow
    // *************************************

    // Get a random permutation of the flow indices (except current one)
    std::vector<int> rand_order;
    for (int i = 0; i < flow_fields.size(); ++i)
    {
        if (i != current_flow)
            rand_order.push_back(i);
    }
    std::random_shuffle(rand_order.begin(), rand_order.end());

    // Search for the first new flow that is reachable
    for (auto rand_i : rand_order)
    {
        current_flow = rand_i;
        UpdateDistance();
        if (distance_to_goal < 10e8)
            break;
    }
}

void FlowFollower::OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects)
{
    // 1. Check if we reached the objective and change it in that case
    CheckGoal();

    // 2. Apply a force on the actor according to the flow field
    FlowForce();

    // 3. Apply force from other actors
    AvoidActors(vehicles);

    // 4. Apply force from robot
    double robot_margin_factor = robot_margin / actor_margin;
    AvoidRobot(objects, robot_margin_factor, robot_slow > 0);

    // Get total applied force from the summed acceleration
    //showed_force = acceleration * mass;
}

void FlowFollower::OnPoseUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<gazebo::physics::EntityPtr> objects)
{
    // Update
    UpdatePositionContactObstacles(objects, dt);
    UpdateModel();
}

void FlowFollower::CheckGoal()
{
    // Update distance
    UpdateDistance();

    // Set new goal
    if (distance_to_goal < 1.0 || distance_to_goal > 10e8)
    {

        // Get a random permutation of the flow indices (except current one)
        std::vector<int> rand_order;
        for (int i = 0; i < flow_fields.size(); ++i)
        {
            if (i != current_flow)
                rand_order.push_back(i);
        }
        std::random_shuffle(rand_order.begin(), rand_order.end());

        // Search for the first new flow that is reachable
        for (auto rand_i : rand_order)
        {
            current_flow = rand_i;
            UpdateDistance();
            if (distance_to_goal < 10e8)
                break;
        }
    }

    // if (distance_to_goal > 10e8)
    //     ROS_WARN_STREAM(this->GetName() << " at an unreachable point of the flow map #" << current_flow);
}

void FlowFollower::UpdateDistance()
{

    distance_to_goal = flow_fields[current_flow]->SmoothValueLookup(this->pose.Pos());

    // int r, c;
    // if (!flow_fields[current_flow]->PosToIndicies(this->pose.Pos(), r, c))
    //     throw std::out_of_range("FlowFollower position outside the flow map");
    // distance_to_goal = flow_fields[current_flow]->value_function[r][c];
}

void FlowFollower::FlowForce()
{
    // Get flow at current position
    ignition::math::Vector2d flow;
    if (!flow_fields[current_flow]->SmoothFlowLookup(this->pose.Pos(), flow))
    {
        //throw std::out_of_range("FlowFollower position outside the flow map");
        flow.X() = - this->pose.Pos().X();
        flow.Y() = - this->pose.Pos().Y();
        flow.Normalize();
    }

    // Apply flow directly as a force. the flow length on a standard pixel = resolution
    ignition::math::Vector3d steer(flow.X(), flow.Y(), 0);

    // Set the standard flow force to 30% of the max force (can be higher close to obstacles)
    steer *= 0.2 * max_force / flow_fields[current_flow]->resolution;

    // Flow is the direction of the desired speed
    //ignition::math::Vector3d desired_v(flow.X(), flow.Y(), 0);
    //.Normalize();
    //desired_v *= this->max_speed;

    // Clamp force
    if (steer.Length() > this->max_force)
    {
        steer.Normalize();
        steer *= this->max_force;
    }

    // Save flow force
    flow_force = steer;

    ApplyForce(steer);
}


//----------------------------------------------------------------------------------------------------------------------------------------------

Bouncer::Bouncer(gazebo::physics::ActorPtr _actor,
                           double _mass,
                           double _max_force,
                           double _max_speed,
                           ignition::math::Pose3d initial_pose,
                           ignition::math::Vector3d initial_velocity,
                           std::vector<gazebo::physics::EntityPtr> objects,
                           double _obstacle_margin,
                           double _actor_margin)
    : Vehicle(_actor, _mass, _max_force, _max_speed, initial_pose, initial_velocity, objects, _obstacle_margin, _actor_margin)
{

    // Choose an initial direction
    // ***************************

    if (initial_velocity.Length() > 0.000001)
    {
        this->velocity = initial_velocity;
    }
    else
    {
        // Random orientation
        double theta = ignition::math::Rand::DblUniform(-IGN_PI, IGN_PI);

        // Corresponding speed
        this->velocity = ignition::math::Vector3d(cos(theta), sin(theta), 0) * _max_speed;
    }


    return;
}

void Bouncer::OnUpdate(const gazebo::common::UpdateInfo &_info, double dt, std::vector<boost::shared_ptr<Vehicle>> vehicles, std::vector<gazebo::physics::EntityPtr> objects)
{

    // Handle collisions with obstacles
    // ********************************

    // Predict next position
    auto next_pos = this->pose.Pos() + this->velocity * dt;

    // Find the closest obstacle
    bool inside_obstacle = false;
    ignition::math::Vector3d min_normal(0, 0, 0);
    double min_dist = 10e9;
    for (gazebo::physics::EntityPtr object : objects)
    {
        
        // Special case of the robot (handle it like )
        if (object->GetName() == "jackal")
        {
            ignition::math::Box box = object->BoundingBox();
            ignition::math::Vector3d rad = this->pose.Pos() - box.Center();
            rad.Z() = 0;
            double dist = rad.Length();
            rad.Normalize();
            
            // Bounce if we are too close 3 times obstacle margin for robot 
            if (dist < 3 * obstacle_margin)
                PlaneBounce(rad, dt);
        }
        else
        {
            // Get object bounding box
            ignition::math::Box box = object->BoundingBox();

            // Check height for doors
            double min_z = std::min(box.Min().Z(), box.Max().Z());
            if (min_z > 1.5)
                continue;

            // Get the vector from closest box point to the current pos
            auto next_normal = utilities::min_box_repulsive_vector(next_pos, box);
            next_normal.Z() = 0;
            
            // Correct the normal direction in case the next position is inside and obstacle
            double dist;
            if (utilities::inside_box(box, next_pos))
            {
                next_normal *= -1;
                dist = 0;
            }
            else
            {
                dist = next_normal.Length();
            }
            next_normal.Normalize();

            // Bounce if we are too close (obstacle margin is the limit)
            if (dist < obstacle_margin)
                PlaneBounce(next_normal, dt);
        }
    }

    // Handle collisions with vehicles
    // *******************************
    
    for (int i = 0; i < (int)vehicles.size(); i++)
    {
        auto other = vehicles[i]->GetActor();
        if (other == this->actor)
            continue;

        ignition::math::Vector3d diff_pos = this->pose.Pos() - other->WorldPose().Pos();
        diff_pos.Z() = 0;
        double dist = diff_pos.Length();
        diff_pos.Normalize();
        
        // Bounce if we are too close 3 times obstacle margin for robot 
        if (dist < 3 * obstacle_margin)
            PlaneBounce(diff_pos, dt);
    }


}

void Bouncer::PlaneBounce(const ignition::math::Vector3d& normal, double dt)
{
    // Flip speed in the direction of the normal vector
    double correction = normal.Dot(this->velocity);
    if (correction < 0)
    {
        // we want: velocity += - 2 * correction * normal
        this->acceleration += normal * (-2 * correction / dt);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------------------

void Reproducer::OnUpdate(const gazebo::common::UpdateInfo &_info,
                          double dt,
                          std::vector<boost::shared_ptr<Vehicle>> vehicles,
                          std::vector<gazebo::physics::EntityPtr> objects,
                          ignition::math::Pose3d new_pose0)
{

    // Get next pos
    this->new_pose = new_pose0;

    // // Predict next position
    // auto next_pos = this->pose.Pos() + this->velocity * dt;
}

void Reproducer::UpdatePosition(double dt)
{
    // Set the new pose
    this->pose.Pos() = this->new_pose.Pos();
    this->pose.Rot() = this->new_pose.Rot();


    // Set the velocity (in case any other part of the code needs it)
    this->velocity = (this->new_pose.Pos() - this->pose.Pos()) / dt;


    // // Handle yaw
    // ignition::math::Vector3d direction = this->velocity;
    // direction.Normalize();
    // double dir_yaw = atan2(direction.Y(), direction.X());
    // double current_yaw = this->pose.Rot().Yaw();
    // ignition::math::Angle yaw_diff = dir_yaw - current_yaw + IGN_PI_2;
    // yaw_diff.Normalize();
    // if (this->velocity.Length() < 10e-2)
    // {
    //     yaw_diff = 0;
    // }

    // // Update pose
    // this->pose.Pos() += this->velocity * dt;
    // this->pose.Rot() = ignition::math::Quaterniond(IGN_PI_2, 0, current_yaw + yaw_diff.Radian() * 0.1);
    // this->acceleration = 0;
}