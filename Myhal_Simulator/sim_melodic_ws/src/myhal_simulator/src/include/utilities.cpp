#include "utilities.hh"

std::string utilities::color_text(std::string text, std::vector<int> rgb){
    auto reset = "\033[0m";
    auto leading = "\033[38;2;" + std::to_string(rgb[0]) + ";" + std::to_string(rgb[1]) + ";" + std::to_string(rgb[2]) + "m";
    return leading + text + reset;
}

std::vector<ignition::math::Line3d> utilities::get_box_edges(ignition::math::Box box){
	ignition::math::Vector3d min_corner = box.Min();
	ignition::math::Vector3d max_corner = box.Max();
	min_corner.Z() = 0;
	max_corner.Z() = 0;


	//TODO: ensure that these methods work using Line3d
	ignition::math::Line3d left = ignition::math::Line3d(min_corner.X(),min_corner.Y(),min_corner.X(), max_corner.Y());
	ignition::math::Line3d right = ignition::math::Line3d(max_corner.X(),min_corner.Y(),max_corner.X(), max_corner.Y());
	ignition::math::Line3d top = ignition::math::Line3d(min_corner.X(),max_corner.Y(),max_corner.X(), max_corner.Y());
	ignition::math::Line3d bot = ignition::math::Line3d(min_corner.X(),min_corner.Y(),max_corner.X(), min_corner.Y());
				
	std::vector<ignition::math::Line3d> edges = {left, right, top, bot}; // store all edges of link bounding box
	
	return edges;
}

std::vector<ignition::math::Line3d> utilities::get_edges(gazebo::physics::EntityPtr entity){
	ignition::math::Box box = entity->BoundingBox();
	return utilities::get_box_edges(box);
}

std::vector<ignition::math::Vector3d> utilities::get_corners(gazebo::physics::EntityPtr entity){
	ignition::math::Box box = entity->BoundingBox();
	ignition::math::Vector3d min_corner = box.Min();
	ignition::math::Vector3d max_corner = box.Max();
	min_corner.Z() = 0;
	max_corner.Z() = 0;


	//TODO: ensure that these methods work using Line3d
	ignition::math::Vector3d bot_l = min_corner;
	ignition::math::Vector3d bot_r = ignition::math::Vector3d(max_corner.X(),min_corner.Y(),0);
	ignition::math::Vector3d top_l = ignition::math::Vector3d(min_corner.X(),max_corner.Y(),0);
	ignition::math::Vector3d top_r = max_corner;
				
	std::vector<ignition::math::Vector3d> corners = {bot_l, bot_r, top_l, top_r}; // store all edges of link bounding box

	return corners;
}

std::vector<ignition::math::Vector3d> utilities::get_box_corners(ignition::math::Box box){
	ignition::math::Vector3d min_corner = box.Min();
	ignition::math::Vector3d max_corner = box.Max();
	min_corner.Z() = 0;
	max_corner.Z() = 0;

	//TODO: ensure that these methods work using Line3d
	ignition::math::Vector3d bot_l = min_corner;
	ignition::math::Vector3d bot_r = ignition::math::Vector3d(max_corner.X(),min_corner.Y(),0);
	ignition::math::Vector3d top_l = ignition::math::Vector3d(min_corner.X(),max_corner.Y(),0);
	ignition::math::Vector3d top_r = max_corner;
				
	std::vector<ignition::math::Vector3d> corners = {bot_l, bot_r, top_l, top_r}; // store all edges of link bounding box

	return corners;
}

//returns true if the projection of pos falls within the bounds of edge. If it does, it stores the normal vector between the edge and pos in normal (pointing from edge to point)

bool utilities::get_normal_to_edge(ignition::math::Vector3d pos, ignition::math::Line3d edge, ignition::math::Vector3d &normal){
    ignition::math::Vector3d edge_vector = edge.Direction(); 
					
	ignition::math::Vector3d pos_vector = ignition::math::Vector3d(pos.X()-edge[0].X(), pos.Y()-edge[0].Y(), 0);
	ignition::math::Vector3d proj = ((pos_vector.Dot(edge_vector))/(edge_vector.Dot(edge_vector)))*edge_vector; 

	if (edge.Within(proj+edge[0])){
		normal = pos_vector-proj;
        return true;
	} else{
        return false;
    }
}

bool utilities::inside_box(ignition::math::Box box, ignition::math::Vector3d point, bool edge){
	ignition::math::Vector3d min_corner = box.Min();
	ignition::math::Vector3d max_corner = box.Max();

	if (edge){
		return (point.X() <= std::max(min_corner.X(), max_corner.X())
		&& point.X() >= std::min(min_corner.X(), max_corner.X())
		&& point.Y() <= std::max(min_corner.Y(), max_corner.Y())
		&& point.Y() >= std::min(min_corner.Y(), max_corner.Y()));
	} else{
		return (point.X() < std::max(min_corner.X(), max_corner.X())
		&& point.X() > std::min(min_corner.X(), max_corner.X())
		&& point.Y() < std::max(min_corner.Y(), max_corner.Y())
		&& point.Y() > std::min(min_corner.Y(), max_corner.Y()));
	}
}



double utilities::width(ignition::math::Box box){
	return std::abs(box.Max().X() - box.Min().X());
}

double utilities::height(ignition::math::Box box){
	return std::abs(box.Max().Y() - box.Min().Y());
}

double utilities::map(double val, double from_min, double from_max, double to_min, double to_max){
	double frac = (val - from_min)/(from_max-from_min);
	return to_min + frac*(to_max-to_min);
}

void utilities::print_vector(ignition::math::Vector3d vec, bool newline){
	std::printf("(%.2f, %.2f, %.2f)", vec.X(), vec.Y(), vec.Z());
	if (newline){
		std::printf("\n");
	}
}

//returns the shortest normal vector between pos and one of the edges on the bounding box of entity
// will return the shortest corner distance if the normal does not exist 
ignition::math::Vector3d utilities::min_box_repulsive_vector(ignition::math::Vector3d pos, ignition::math::Box box){
    auto edges = utilities::get_box_edges(box);
	ignition::math::Vector3d min_normal;
	double min_mag = 10e9;
	bool found = false;

	for (ignition::math::Line3d edge: edges){

					
		ignition::math::Vector3d edge_vector = edge.Direction(); // vector in direction of edge 
					
		ignition::math::Vector3d pos_vector = ignition::math::Vector3d(pos.X()-edge[0].X(), pos.Y()-edge[0].Y(), 0);// vector from edge corner to actor pos
					
		ignition::math::Vector3d proj = ((pos_vector.Dot(edge_vector))/(edge_vector.Dot(edge_vector)))*edge_vector; // project pos_vector onto edge_vector
			
		//check if the projected point is within the edge
		if (edge.Within(proj+edge[0])){
			//compute normal
			ignition::math::Vector3d normal = pos_vector-proj;
						
			if (normal.Length() < min_mag){
				min_normal = normal;
				min_mag = normal.Length();
				found = true;
			}

		}
				
	}

	if (!found){ // iterate over all corners and find the closest one 
		min_mag = 10e9;
		auto corners = utilities::get_box_corners(box);
		for (auto corner: corners){
			if (pos.Distance(corner) < min_mag){
				min_mag = pos.Distance(corner);
				min_normal = pos-corner;
			}
		}
	}

	return min_normal;
}


ignition::math::Vector3d utilities::min_repulsive_vector(ignition::math::Vector3d pos, gazebo::physics::EntityPtr entity){
    return utilities::min_box_repulsive_vector(pos, entity->BoundingBox());
    /*
	std::vector<ignition::math::Line3d> edges = utilities::get_edges(entity);

	ignition::math::Vector3d min_normal;
	double min_mag = 1000000;
	bool found = false;

	for (ignition::math::Line3d edge: edges){

					
		ignition::math::Vector3d edge_vector = edge.Direction(); // vector in direction of edge 
					
		ignition::math::Vector3d pos_vector = ignition::math::Vector3d(pos.X()-edge[0].X(), pos.Y()-edge[0].Y(), 0);// vector from edge corner to actor pos
					
		ignition::math::Vector3d proj = ((pos_vector.Dot(edge_vector))/(edge_vector.Dot(edge_vector)))*edge_vector; // project pos_vector onto edge_vector
			
		//check if the projected point is within the edge
		if (edge.Within(proj+edge[0])){
			//compute normal
			ignition::math::Vector3d normal = pos_vector-proj;
						
			if (normal.Length() < min_mag){
				min_normal = normal;
				min_mag = normal.Length();
				found = true;
			}

		}
				
	}

	if (!found){ // iterate over all corners and find the closest one 
		min_mag = 1000000;
		auto corners = utilities::get_corners(entity);
		for (auto corner: corners){
			if (pos.Distance(corner) < min_mag){
				min_mag = pos.Distance(corner);
				min_normal = pos-corner;
			}
		}
	}

	return min_normal;
    */
}


bool utilities::contains(ignition::math::Box b1, ignition::math::Box b2){

	ignition::math::Vector3d min_corner = b2.Min();
	ignition::math::Vector3d max_corner = b2.Max();
	min_corner.Z() = 0;
	max_corner.Z() = 0;

	ignition::math::Vector3d bot_l = min_corner;
	ignition::math::Vector3d bot_r = ignition::math::Vector3d(max_corner.X(),min_corner.Y(),0);
	ignition::math::Vector3d top_l = ignition::math::Vector3d(min_corner.X(),max_corner.Y(),0);
	ignition::math::Vector3d top_r = max_corner;

	// check if each point lies within b1:

	if (utilities::inside_box(b1, bot_l) && utilities::inside_box(b1, bot_r) && utilities::inside_box(b1, top_l) && utilities::inside_box(b1, top_r)){
		return true;
	}

	return false;

}

/*
input: a point and a bounding box
output: the vector that is the shortest between the point and some point on the box

if it cannot be projected onto a side: returns min corner distance 
*/
double utilities::dist_to_box(ignition::math::Vector3d pos, ignition::math::Box box){
	// get all planes that make up the box 

	auto min = ignition::math::Vector3d(std::min(box.Min().X(), box.Max().X()), std::min(box.Min().Y(), box.Max().Y()), std::min(box.Min().Z(), box.Max().Z()));
	auto max = ignition::math::Vector3d(std::max(box.Min().X(), box.Max().X()), std::max(box.Min().Y(), box.Max().Y()), std::max(box.Min().Z(), box.Max().Z()));

	auto into_page = ignition::math::Vector3d(0, 1, 0);
	auto across_page = ignition::math::Vector3d(1, 0, 0);
	auto up = ignition::math::Vector3d(0, 0, 1);

	double min_dist = 10e9;

	auto bottom = ignition::math::Planed(into_page.Cross(across_page), -min.Z());

	if (pos.X() >= min.X() && pos.X() <= max.X() && pos.Y() >= min.Y() && pos.Y() <= max.Y()){
		//std::cout << "bot\n";
		min_dist = std::min(min_dist, std::abs(bottom.Distance(pos)));
	}

	auto top = ignition::math::Planed(-1*into_page.Cross(across_page), max.Z());


	if (pos.X() >= min.X() && pos.X() <= max.X()  && pos.Y() >= min.Y() && pos.Y() <= max.Y()){
		//std::cout << "top\n";
		min_dist = std::min(min_dist, std::abs(top.Distance(pos)));
	}

	auto front = ignition::math::Planed(across_page.Cross(up), -min.Y());

	if (pos.X() >= min.X() && pos.X() <= max.X() && pos.Z() >= min.Z() && pos.Z() <= max.Z()){
		//std::cout << "front\n";
		min_dist = std::min(min_dist, std::abs(front.Distance(pos)));
	}

	auto back = ignition::math::Planed(-1*across_page.Cross(up), max.Y());

	if (pos.X() >= min.X() && pos.X() <= max.X()  && pos.Z() >= min.Z() && pos.Z() <= max.Z()){
		//std::cout << "back\n";
		min_dist = std::min(min_dist, std::abs(back.Distance(pos)));
	}

	auto left = ignition::math::Planed(-1*into_page.Cross(up), -min.X());

	if (pos.Y() >= min.Y() && pos.Y() <= max.Y() && pos.Z() >= min.Z() && pos.Z() <= max.Z()){
		//std::cout << "left\n";
		min_dist = std::min(min_dist, std::abs(left.Distance(pos)));
		//std::cout << min_dist << std::endl;
	}

	auto right = ignition::math::Planed(into_page.Cross(up), max.X());

	if (pos.Y() >= min.Y() && pos.Y() <= max.Y() && pos.Z() >= min.Z() && pos.Z() <= max.Z()){
		//std::cout << "right\n";
		min_dist = std::min(min_dist, std::abs(right.Distance(pos)));
		//std::cout << min_dist << std::endl;
	}


	if (min_dist == 10e9){
		//std::cout << "CORNERS\n";
		min_dist = std::min(min_dist, (min-pos).Length());
		min_dist = std::min(min_dist, (max-pos).Length());
		min_dist = std::min(min_dist, (ignition::math::Vector3d(0,0,max.Z() - min.Z())+min-pos).Length());
		min_dist = std::min(min_dist, (ignition::math::Vector3d(max.X() - min.X(),0,max.Z() - min.Z())+min-pos).Length());
		min_dist = std::min(min_dist, (ignition::math::Vector3d(max.X() - min.X(),0,0)+min-pos).Length());
		min_dist = std::min(min_dist, (ignition::math::Vector3d(max.X() - min.X(),max.Y() - min.Y(),0)+min-pos).Length());
		min_dist = std::min(min_dist, (ignition::math::Vector3d(0,max.Y() - min.Y(),0)+min-pos).Length());
		min_dist = std::min(min_dist, (ignition::math::Vector3d(0,max.Y() - min.Y(),max.Z() - min.Z())+min-pos).Length());
	} 
	
	return min_dist;
	
}

utilities::Path::Path(){
	this->radius = 0.5;
}

utilities::Path::Path(double _radius){
	this->radius = _radius;
	
}

void utilities::Path::AddPoint(ignition::math::Vector3d _point){
	this->points.push_back(_point);
}

ignition::math::Pose3d utilities::InterpolatePose(double target_time, double t1, double t2, ignition::math::Pose3d pose1, ignition::math::Pose3d pose2){
	double alpha = 0;
	if (t2 != t1){
		alpha = (target_time-t1)/(t2-t1);
	}

	auto rot1 = pose1.Rot();
	auto rot2 = pose2.Rot();


	ignition::math::Pose3d res;	
	res.Pos() = ((pose2.Pos() - pose1.Pos())/(t2-t1))*(target_time-t1)+pose1.Pos();
	res.Rot() = ignition::math::Quaterniond::Slerp(alpha, rot1, rot2);

	return res;
}

std::vector<std::string> utilities::split(std::string in, char delim){
    std::vector<std::string> res;
    int last = 0;
    for (int i = 0; i < in.size(); ++i){
        if (in[i] ==  delim){
            res.push_back(in.substr(last, i-last));
            last = i + 1;
        }
    }
    return res;
}
