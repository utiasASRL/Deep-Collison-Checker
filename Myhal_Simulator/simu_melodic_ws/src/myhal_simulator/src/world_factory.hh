#ifndef WORLD_FACTORY_HH
#define  WORLD_FACTORY_HH

#include <functional>
#include "world_entities.hh"
#include <utility>
#include <ros/ros.h>
#include "frame.hh"
#include "costmap.hh"
#include "utilities.hh"
#include <iterator>
#include <algorithm>   
#include <ctime>        
#include <cstdlib> 
#include <iostream>
#include <fstream>

class CamInfo{
    
    public:
        int mode; // (0 - sentry), (1 - hoverer), (2 - stalker) 
        double x,y,z,period,dist;

        CamInfo(int mode, double x, double y, double z, double T, double dist): mode(mode), x(x), y(y), z(z), period(T), dist(dist){}

};

class RoomInfo{

    public:
        std::shared_ptr<myhal::Room> room;
        std::string scenario;
        std::vector<std::vector<double>> positions;
        std::string room_name;

        RoomInfo(std::shared_ptr<myhal::Room> _room, std::string _scenario, std::vector<std::vector<double>> _positions, std::string _room_name):
        room(_room), scenario(_scenario), positions(_positions), room_name(_room_name){}

};

class TableInfo{

    public: 
        std::string name;
        std::string table_name;
        std::string chair_name;

        TableInfo(std::string _name, std::string _table_name, std::string _chair_name):
        name(_name), table_name(_table_name), chair_name(_chair_name){}
    
};

class ModelInfo{

    public: 
        std::string name;
        std::string filename;
        double width;
        double length;
        double height;

        ModelInfo(std::string _name, std::string _filename, double _width, double _length, double _height = -1) : name(_name), filename(_filename), width(_width), length(_length), height(_height)
        {}
};

class ActorInfo: public ModelInfo{

    public: 
        std::string plugin;

        ActorInfo(std::string _name, std::string _filename, std::string _plugin, double _obstacle_margin):
        ModelInfo(_name,_filename, _obstacle_margin, _obstacle_margin), plugin(_plugin){}
};

class Scenario{

    protected:

        std::vector<std::shared_ptr<ModelInfo>> models;
        std::vector<std::shared_ptr<TableInfo>> tables;

    public:

        double pop_density;
        int pop_num;
        double model_percentage;
        std::string actor;
        

        Scenario(double _pop_denisty, int _pop_num, double _model_percentage, std::string _actor);

        void AddModel(std::shared_ptr<ModelInfo> model);
        void AddTable(std::shared_ptr<TableInfo> table);

        std::shared_ptr<ModelInfo> GetRandomModel();
        std::shared_ptr<TableInfo> GetRandomTable();
};

class WorldHandler{

    public: 
        WorldHandler();

        void Load();

        void LoadParams();

        void WriteToFile(std::string out_name);

        void FillRoom(std::shared_ptr<RoomInfo> room_info);

        void AddCameras();

        std::string world_string;

        std::vector<std::shared_ptr<myhal::Model>> world_models;

        //std::vector<bool> camera_pos;

        std::string start_time;

        std::vector<std::string> room_names; //to name the cameras effectivly

        std::vector<std::shared_ptr<CamInfo>> cam_info;

        //TODO: change these to not be pointers
        //std::map<std::string, std::shared_ptr<SDFPlugin>> vehicle_plugins; //one per actor

        std::map<std::string, std::vector<std::shared_ptr<SDFPlugin>>> vehicle_plugins; //one per actor
        std::map<std::string, double> custom_actor_spawn_parameters;
        std::string use_custom_spawn_room;

        std::vector<std::shared_ptr<SDFAnimation>> animation_list; //added to all actors 
        std::map<std::string, std::shared_ptr<ModelInfo>> model_info;
        std::map<std::string, std::shared_ptr<Scenario>> scenarios;
        std::map<std::string , std::shared_ptr<ActorInfo>> actor_info;
        std::map<std::string, std::shared_ptr<TableInfo>> table_info;

        std::vector<std::shared_ptr<RoomInfo>> rooms;
        //std::map<std::string , std::shared_ptr<myhal::Room>> rooms;

        std::vector<ignition::math::Box> walls;

        double robot_radius = std::sqrt((0.21*0.21) + (0.165*0.165));

        std::string tour_name;

        std::vector<ignition::math::Pose3d> route;

        std::vector<std::shared_ptr<myhal::Model>> doors; 

        std::shared_ptr<Costmap> costmap;

};


#endif
