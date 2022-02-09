#ifndef WORLD_ENTITIES_HH
#define WORLD_ENTITIES_HH

#include <vector>
#include <string>
#include "sdfstring.hh"
#include <ignition/math/Pose3.hh>
#include <ignition/math/Box.hh>

namespace myhal{


    class Model{

        protected:

            static int num_models;
            
            std::vector<std::shared_ptr<SDFPlugin>> plugins;
            

        public:

            std::string model_file;

            std::vector<ignition::math::Vector3d> corners; 

            std::string name;

            ignition::math::Pose3d pose;

            Model(std::string _name, ignition::math::Pose3d _pose, std::string _model_file, double _width, double _length);

            void AddPlugin(std::shared_ptr<SDFPlugin> plugin);

            virtual std::string CreateSDF();

            void AddToWorld(std::string &world_string);

            ignition::math::Box GetCollisionBox();

            bool DoesCollide(std::shared_ptr<Model> other);

            void Reposition(double x_shift, double y_shift);

            void RotateClockwise(double angle);

            double GetWidth();

            double GetLength();

    };

    class Actor: public Model{

        private:
            
            std::vector<std::shared_ptr<SDFAnimation>> animations;

        public:

            using Model::Model;
            
            void AddAnimation(std::shared_ptr<SDFAnimation> animation);

            std::string CreateSDF();

    };

    class IncludeModel: public Model{

        public:

            using Model::Model;

            std::string CreateSDF();

    };


    class BoundaryBox: public Model{

        public:

            double width,length;
            
            BoundaryBox(double _x, double _y, double _width, double _length);

            std::string CreateSDF();
    };

    class Camera: public Model{

        public:
            
            std::string filepath;
            bool save;

            Camera(std::string name, ignition::math::Pose3d _pose, std::string path = "");

            std::string CreateSDF();


    };

    class TableGroup{

        public:

            std::vector<std::shared_ptr<Model>> chairs;
            int num_chairs;
            double rotation_angle;
            std::shared_ptr<Model> table_model;
            std::shared_ptr<Model> chair_model;

            TableGroup(std::shared_ptr<Model> _table_model, std::shared_ptr<Model> _chair_model, double _rotation_angle);

    };

    class Room{

        protected:

            std::string building_name;
            bool enclosed;

        public: 

            ignition::math::Box boundary; 
            
            std::vector<ignition::math::Box> walls;

            std::vector<ignition::math::Pose3d> route;

            std::vector<std::shared_ptr<Model>> models;

            Room(double x_min, double y_min, double x_max, double y_max, std::vector<ignition::math::Box> walls, std::vector<ignition::math::Pose3d> route,  bool _enclosed);

            bool AddModel(std::shared_ptr<Model> model); //if safety is true, it prevents model collisions 

            bool AddModelRandomly(std::shared_ptr<Model> model); 

            bool AddModelSelectively(std::shared_ptr<Model> model, double x, double y); 

            void AddToWorld(std::string &world_string);

            double Area();
    };

}

#endif
