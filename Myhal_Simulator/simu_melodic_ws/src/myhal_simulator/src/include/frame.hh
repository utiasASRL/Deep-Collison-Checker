#pragma once 

#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include <ignition/math/Box.hh>
#include <vector>
#include "happily.h"

class Point{

    private:

        ignition::math::Vector3d pos;
        int cat;

    public:

        double X(){
            return this->pos.X();
        }

        double Y(){
            return this->pos.Y();
        }

        double Z(){
            return this->pos.Z();
        }

        int Cat(){
            return this->cat;
        }

        void SetCat(int cat){
            this->cat = cat;
        }

        Point(ignition::math::Vector3d pos, int cat): pos(pos), cat(cat){};
};

void addPoints(happly::PLYData &plyOut, std::vector<Point>& vertexPositions, bool color = false){

    std::string vertexName = "vertex";
    size_t N = vertexPositions.size();

    // Create the element
    if (!plyOut.hasElement(vertexName)) {
    plyOut.addElement(vertexName, N);
    }

    bool catagory = true;

    // De-interleave
    std::vector<float> xPos(N);
    std::vector<float> yPos(N);
    std::vector<float> zPos(N);
    std::vector<int> cat(N);
    std::vector<int> r(N);
    std::vector<int> g(N);
    std::vector<int> b(N);
    for (size_t i = 0; i < vertexPositions.size(); i++) {
        xPos[i] = (float) vertexPositions[i].X();
        yPos[i] = (float) vertexPositions[i].Y();
        zPos[i] = (float) vertexPositions[i].Z();
        if (catagory){
            cat[i] = vertexPositions[i].Cat();
            if (color){
                if (cat[i] == 0){ //ground
                    r[i] = 244;
                    g[i] = 244;
                    b[i] = 237;
                } else if (cat[i] == 1){ // chair
                    r[i] = 0;
                    g[i] = 240;
                    b[i] = 181;
                } else if (cat[i] == 2){ // mover
                    r[i] = 255;
                    g[i] = 189;
                    b[i] = 0;
                } else if (cat[i] == 3){ // sitter
                    r[i] = 246;
                    g[i] = 16;
                    b[i] = 103;
                } else if (cat[i] == 4){ // table
                    r[i] = 94;
                    g[i] = 35;   
                    b[i] = 157;
                } else if (cat[i] == 5){ // wall
                    r[i] = 196;
                    g[i] = 215;
                    b[i] = 242;
                }
            }
            if (cat[i] < 0){

                catagory = false;
            }
        }
        
    }

    // Store
    plyOut.getElement(vertexName).addProperty<float>("x", xPos);
    plyOut.getElement(vertexName).addProperty<float>("y", yPos);
    plyOut.getElement(vertexName).addProperty<float>("z", zPos);
    if (catagory){
        if (color){
            plyOut.getElement(vertexName).addProperty<int>("red", r);
            plyOut.getElement(vertexName).addProperty<int>("green", g);
            plyOut.getElement(vertexName).addProperty<int>("blue", b);
        } else{
            plyOut.getElement(vertexName).addProperty<int>("category", cat);
        }
        
    }
}

void addPose(happly::PLYData &plyOut, ignition::math::Pose3d pose){

    plyOut.addElement("gt_pose", 1);
    plyOut.getElement("gt_pose").addProperty<float>("pos.x", {(float) pose.Pos().X()});
    plyOut.getElement("gt_pose").addProperty<float>("pos.y", {(float) pose.Pos().Y()});
    plyOut.getElement("gt_pose").addProperty<float>("pos.z", {(float) pose.Pos().Z()});
    plyOut.getElement("gt_pose").addProperty<double>("rot.x", {pose.Rot().X()});
    plyOut.getElement("gt_pose").addProperty<double>("rot.y", {pose.Rot().Y()});
    plyOut.getElement("gt_pose").addProperty<double>("rot.z", {pose.Rot().Z()});
    plyOut.getElement("gt_pose").addProperty<double>("rot.w", {pose.Rot().W()});
}

class Frame{

    private:

       
        ignition::math::Pose3d gt_pose;
        double time;
        bool has_pose = true;

    public:

        std::vector<Point> points;

        Frame(bool has_pose): has_pose(has_pose) {};

        Frame(ignition::math::Pose3d gt_pose, double time): gt_pose(gt_pose), time(time){};

        void AddPoint(ignition::math::Vector3d pos, int cat = -1){
            this->points.push_back(Point(pos,cat));
        }

        void WriteToFile(std::string path, bool color = false){
            happly::PLYData plyOut;
            if (this->has_pose){
                addPose(plyOut, this->gt_pose);
            }
            addPoints(plyOut, this->points, color);
            plyOut.write(path + std::to_string(this->time) + ".ply", happly::DataFormat::Binary);
        }

        void SetPose(ignition::math::Pose3d gt_pose){
            this->has_pose = true;
            this->gt_pose = gt_pose;
        }

        void SetTime(double time){
            this->time = time;
        }

        double Time(){
            return this->time;
        }

        std::vector<Point> Points(){
            return this->points;
        }
    
};

Frame ReadFrame(std::string filepath, bool gt = false){
    happly::PLYData plyIn(filepath);
    Frame res = Frame(false);

    std::vector<double> xPos = plyIn.getElement("vertex").getProperty<double>("x");
    std::vector<double> yPos = plyIn.getElement("vertex").getProperty<double>("y");
    std::vector<double> zPos = plyIn.getElement("vertex").getProperty<double>("z");
    std::vector<int> classif = plyIn.getElement("vertex").getProperty<int>("classif");
    std::vector<int> labels = plyIn.getElement("vertex").getProperty<int>("labels");

    for (size_t i = 0; i < xPos.size(); i++) {
        if (gt){
            res.AddPoint(ignition::math::Vector3d(xPos[i], yPos[i], zPos[i]), labels[i]);
        } else{
            res.AddPoint(ignition::math::Vector3d(xPos[i], yPos[i], zPos[i]), classif[i]);
        }
        
    }

    return res;
}


class BoxObject{

    private:

        ignition::math::Box box;

        int cat;

    public:

        BoxObject(ignition::math::Box box, int cat): box(box), cat(cat) {};

        double MinX(){
            return this->box.Min().X();
        }

        double MinY(){
            return this->box.Min().Y();
        }

        double MinZ(){
            return this->box.Min().Z();
        }

        double MaxX(){
            return this->box.Max().X();
        }

        double MaxY(){
            return this->box.Max().Y();
        }

        double MaxZ(){
            return this->box.Max().Z();
        }

        int Cat(){
            return this->cat;
        }

        ignition::math::Box Box(){
            return this->box;
        }

};

void AddBoxes(happly::PLYData &plyOut, std::vector<BoxObject> boxes){
   
    size_t N = boxes.size();

    // Create the element
    if (!plyOut.hasElement("box")) {
        plyOut.addElement("box", N);
    }

    std::vector<double> min_x(N);
    std::vector<double> min_y(N);
    std::vector<double> min_z(N);
    std::vector<double> max_x(N);
    std::vector<double> max_y(N);
    std::vector<double> max_z(N);
    std::vector<int> cat(N);
    for (size_t i = 0; i < boxes.size(); i++) {
        min_x[i] = boxes[i].MinX();
        min_y[i] = boxes[i].MinY();
        min_z[i] = boxes[i].MinZ();
        max_x[i] = boxes[i].MaxX();
        max_y[i] = boxes[i].MaxY();
        max_z[i] = boxes[i].MaxZ();
        cat[i] = boxes[i].Cat();
    }

    // Store
    plyOut.getElement("box").addProperty<double>("min_x", min_x);
    plyOut.getElement("box").addProperty<double>("min_y", min_y);
    plyOut.getElement("box").addProperty<double>("min_z", min_z);
    plyOut.getElement("box").addProperty<double>("max_x", max_x);
    plyOut.getElement("box").addProperty<double>("max_y", max_y);
    plyOut.getElement("box").addProperty<double>("max_z", max_z);
    plyOut.getElement("box").addProperty<int>("category", cat);
}

std::vector<BoxObject> ReadObjects(happly::PLYData &plyIn, std::string element_name = "box"){

    std::vector<double> min_x = plyIn.getElement(element_name).getProperty<double>("min_x");
    std::vector<double> min_y = plyIn.getElement(element_name).getProperty<double>("min_y");
    std::vector<double> min_z = plyIn.getElement(element_name).getProperty<double>("min_z");
    std::vector<double> max_x = plyIn.getElement(element_name).getProperty<double>("max_x");
    std::vector<double> max_y = plyIn.getElement(element_name).getProperty<double>("max_y");
    std::vector<double> max_z = plyIn.getElement(element_name).getProperty<double>("max_z");
    std::vector<int> cat = plyIn.getElement(element_name).getProperty<int>("category");
   
    std::vector<BoxObject> boxes;

    for (int i = 0; i< min_x.size(); i++){
        boxes.push_back(BoxObject(ignition::math::Box(min_x[i], min_y[i], min_z[i], max_x[i], max_y[i], max_z[i]), cat[i]));
    }

    return boxes;
}

struct TrajPoint{

    ignition::math::Pose3d pose;
    double time;

    TrajPoint(ignition::math::Pose3d pose, double time): pose(pose), time(time){};

    bool operator<(const TrajPoint &b){
        return this->time < b.time;
    }
};

std::vector<TrajPoint> ReadTrajectory(happly::PLYData &plyIn, std::string element_name = "trajectory"){

    std::vector<double> pos_x = plyIn.getElement(element_name).getProperty<double>("pos_x");
    std::vector<double> pos_y = plyIn.getElement(element_name).getProperty<double>("pos_y");
    std::vector<double> pos_z = plyIn.getElement(element_name).getProperty<double>("pos_z");
    std::vector<double> rot_x = plyIn.getElement(element_name).getProperty<double>("rot_x");
    std::vector<double> rot_y = plyIn.getElement(element_name).getProperty<double>("rot_y");
    std::vector<double> rot_z = plyIn.getElement(element_name).getProperty<double>("rot_z");
    std::vector<double> rot_w = plyIn.getElement(element_name).getProperty<double>("rot_w");
    std::vector<double> time = plyIn.getElement(element_name).getProperty<double>("time");

    std::vector<TrajPoint> trajectory;

    for (int i = 0; i< pos_x.size(); i++){
        trajectory.push_back(TrajPoint(ignition::math::Pose3d(pos_x[i], pos_y[i], pos_z[i], rot_x[i], rot_y[i], rot_z[i], rot_w[i]), time[i]));
    }

    return trajectory;
}

void AddTrajectory(happly::PLYData &plyOut, std::vector<TrajPoint> trajectory){
       
    size_t N = trajectory.size();

    // Create the element
    if (!plyOut.hasElement("trajectory")) {
        plyOut.addElement("trajectory", N);
    }
    std::vector<double> pos_x(N);
    std::vector<double> pos_y(N);
    std::vector<double> pos_z(N);
    std::vector<double> rot_x(N);
    std::vector<double> rot_y(N);
    std::vector<double> rot_z(N);
    std::vector<double> rot_w(N);
    std::vector<double> time(N);

    for (size_t i = 0; i < trajectory.size(); i++) {
        pos_x[i] = trajectory[i].pose.Pos().X();
        pos_y[i] = trajectory[i].pose.Pos().Y();
        pos_z[i] = trajectory[i].pose.Pos().Z();
        rot_x[i] = trajectory[i].pose.Rot().X();
        rot_y[i] = trajectory[i].pose.Rot().Y();
        rot_z[i] = trajectory[i].pose.Rot().Z();
        rot_w[i] = trajectory[i].pose.Rot().W();
        time[i] = trajectory[i].time;
    }

    plyOut.getElement("trajectory").addProperty<double>("pos_x", pos_x);
    plyOut.getElement("trajectory").addProperty<double>("pos_y", pos_y);
    plyOut.getElement("trajectory").addProperty<double>("pos_z", pos_z);
    plyOut.getElement("trajectory").addProperty<double>("rot_x", rot_x);
    plyOut.getElement("trajectory").addProperty<double>("rot_y", rot_y);
    plyOut.getElement("trajectory").addProperty<double>("rot_z", rot_z);
    plyOut.getElement("trajectory").addProperty<double>("rot_w", rot_w);
    plyOut.getElement("trajectory").addProperty<double>("time", time);
}