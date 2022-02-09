#include "world_entities.hh"
#include "quadtree.hh"
#include "puppeteer.hh"
#include <iterator>
#include <algorithm>   
#include <ctime>        
#include <cstdlib> 
#include <iostream>
#include <fstream>
#include "vehicles.hh"
#include "utilities.hh"
#include "costmap.hh"


int main(int argc, char ** argv){

    auto camera = myhal::Camera("camera", ignition::math::Pose3d(0,0,0,0,0,0), "/tmp/");
    std::cout << camera.CreateSDF();
    return 0;
}
