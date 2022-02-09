#ifndef QUADTREE_HH
#define QUADTREE_HH

#include <utility>
#include <memory>
#include <vector>
#include <ignition/math/Box.hh>
#include <algorithm>
#include "gazebo/physics/physics.hh"
#include "gazebo/common/common.hh"

enum types {vehicle_type, entity_type, box_type};


struct QTData{

    ignition::math::Box box;
    boost::shared_ptr<void> data;
    types type;

    QTData(ignition::math::Box box, boost::shared_ptr<void> data, types type);
};

class QuadTree{


    private:

        const int capacity = 1;

        ignition::math::Box boundary;

        std::vector<QTData> objects;

        std::shared_ptr<QuadTree> top_right;
        std::shared_ptr<QuadTree> top_left;
        std::shared_ptr<QuadTree> bot_right;
        std::shared_ptr<QuadTree> bot_left;

    public:

        QuadTree(ignition::math::Box _boundary);

        bool Insert(QTData data);

        void Subdivide();

        std::vector<QTData> QueryRange(ignition::math::Box range);

        void Print();
};


#endif