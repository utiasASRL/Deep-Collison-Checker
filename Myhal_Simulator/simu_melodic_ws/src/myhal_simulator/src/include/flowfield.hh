#ifndef FLOW_FIELD_HH
#define FLOW_FIELD_HH

#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include <ignition/math/Vector2.hh>
#include <ignition/math/Box.hh>
#include "gazebo/physics/physics.hh"
#include "gazebo/common/common.hh"
#include "gazebo/gazebo.hh"
#include <vector>
#include "utilities.hh"
#include "costmap.hh"
#include <algorithm>

class FlowField{

    public:

        std::vector<std::vector<ignition::math::Vector2d>> field;

        std::vector<std::vector<double>> obstacle_map;

        std::vector<std::vector<double>> value_function;

        ignition::math::Vector2d goal;

        int rows;

        int cols;

        double resolution;

        double obstacle_range;

        double obstacle_strength;

        ignition::math::Box boundary;

        FlowField();
        FlowField(boost::shared_ptr<Costmap> costmap0, ignition::math::Vector3d goal0, double obstacle_range0, double obstacle_strength0);

        bool PosToIndicies(ignition::math::Vector3d pos, int &r, int &c);

        bool IndiciesToPos(ignition::math::Vector3d &pos, int r, int c);

        std::vector<std::vector<int>> GetNeighbours(std::vector<int> curr_ind, bool diag = true);

        void ObstacleMap(std::vector<std::vector<int>>& costmap);

        bool Integrate(std::vector<std::vector<int>>& costmap);

        void Compute(std::vector<std::vector<int>>& costmap);

        bool Lookup(ignition::math::Vector3d pos, ignition::math::Vector2d &res);
        bool SmoothFlowLookup(ignition::math::Vector3d pos, ignition::math::Vector2d &res);
        double SmoothValueLookup(ignition::math::Vector3d pos);


        double Linear(const double &t, const double &a, const double &b);
        double Bilinear(const double &tx, const double &ty, const double &c00, const double &c10, const double &c01, const double &c11);
        double BilinearValueLookup(ignition::math::Vector3d pos);

        double Reachability();

};

#endif