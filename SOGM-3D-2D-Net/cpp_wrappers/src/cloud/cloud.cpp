//
//
//		0==========================0
//		|    Local feature test    |
//		0==========================0
//
//		version 1.0 : 
//			> 
//
//---------------------------------------------------
//
//		Cloud source :
//		Define usefull Functions/Methods
//
//----------------------------------------------------
//
//		Hugues THOMAS - 10/02/2017
//


#include "cloud.h"


// Filter functions
// ****************

void filter_pointcloud(std::vector<PointXYZ>& pts, std::vector<float>& scores, float filter_value)
{
	// Remove every points whose score is < filter_value
	auto pts_address = pts.data();
	pts.erase(std::remove_if(pts.begin(), pts.end(),
		[&scores, pts_address, filter_value](const PointXYZ& p) { return scores[(size_t)(&p - pts_address)] < filter_value; }),
		pts.end());
}

void filter_floatvector(std::vector<float>& vec, std::vector<float>& scores, float filter_value)
{
	// Remove every element whose score is < filter_value
	auto vec_address = vec.data();
	vec.erase(std::remove_if(vec.begin(), vec.end(),
		[&scores, vec_address, filter_value](const float& f) { return scores[(size_t)(&f - vec_address)] < filter_value; }),
		vec.end());
}

void filter_floatvector(std::vector<float>& vec, float filter_value)
{
	vec.erase(std::remove_if(vec.begin(), vec.end(), [filter_value](const float s) { return s < filter_value; }), vec.end());
}


// Debug functions
// ***************

void save_cloud(std::string dataPath, 
	std::vector<PointXYZ>& points, 
	std::vector<PointXYZ>& normals, 
	std::vector<float>& features)
{
	// Variables
	uint64_t num_points = points.size();
	uint64_t num_normals = normals.size();
	uint64_t num_features = features.size() / num_points;

	// Safe check
	if (num_features * num_points != features.size())
	{
		std::cout << "Warning: features dimension do not match point cloud" << std::endl;
		std::cout << "         ply saving canceled" << std::endl;
		return;
	}
	if (num_normals != num_points && num_normals != 0)
	{
		std::cout << "Warning: normal dimension do not match point cloud" << std::endl;
		std::cout << "         ply saving canceled" << std::endl;
		return;
	}

	// Open file
	npm::PLYFileOut file(dataPath);

	// Push fields
	file.pushField(num_points, 3, npm::PLY_FLOAT, { "x", "y", "z" }, points);
	if (num_normals > 0)
		file.pushField(num_points, 3, npm::PLY_FLOAT, { "nx", "ny", "nz" }, normals);

	std::vector<std::vector<float>> fields(num_features);
	for (size_t i = 0; i < num_features; i++)
	{
		char buffer[100];
		sprintf(buffer, "f%d", (int)i);
		fields[i] = std::vector<float>(features.begin() + i * num_points, features.begin() + (i + 1) * num_points);
		file.pushField(num_points, 1, npm::PLY_FLOAT, { std::string(buffer) }, fields[i]);
	}
	file.write();
}

void save_cloud(std::string dataPath,
	std::vector<PointXYZ>& points,
	std::vector<PointXYZ>& normals)
{
	std::vector<float> no_f;
	save_cloud(dataPath, points, normals, no_f);
}

void save_cloud(std::string dataPath,
	std::vector<PointXYZ>& points,
	std::vector<float>& features)
{
	std::vector<PointXYZ> no_norms;
	save_cloud(dataPath, points, no_norms, features);
}

void save_cloud(std::string dataPath,
	std::vector<PointXYZ>& points)
{
	std::vector<float> no_f;
	save_cloud(dataPath, points, no_f);
}













void load_cloud(std::string& dataPath,
	std::vector<PointXYZ>& points)
{
	std::vector<float> float_scalar;
	std::string float_scalar_name = "";
	std::vector<int> int_scalar;
	std::string int_scalar_name = "";

	load_cloud(dataPath, points, float_scalar, float_scalar_name, int_scalar, int_scalar_name);

}


void load_cloud(std::string& dataPath, 
	std::vector<PointXYZ>& points, 
	std::vector<float>& float_scalar, 
	std::string& float_scalar_name,
	std::vector<int>& int_scalar,
	std::string& int_scalar_name)
{
	// Variables 
	uint64_t num_points(0);
	std::vector<npm::PLYType> types;
	std::vector<std::string> properties;
	char buffer[500];

	size_t float_str_n = strlen(float_scalar_name.c_str());
	size_t int_str_n = strlen(int_scalar_name.c_str());


	// Open file
	npm::PLYFileIn file(dataPath);

	// Read Header
	if (!file.read(&num_points, &types, &properties))
	{
		std::cout << "ERROR: wrong ply header" << std::endl;
		return;
	}

	// Prepare containers
	points.reserve(num_points);
	float_scalar.reserve(num_points);
	int_scalar.reserve(num_points);

	// Get the points
	for (size_t i = 0; i < properties.size(); i++)
	{
		if (properties[i].size() == 1 && strncmp(properties[i].c_str(), "x", 1) == 0)
			file.getField(i, 3, points);

		if (properties[i].size() == float_str_n && strncmp(properties[i].c_str(), float_scalar_name.c_str(), float_str_n) == 0)
			file.getField(i, 1, float_scalar);

		if (properties[i].size() == int_str_n && strncmp(properties[i].c_str(), int_scalar_name.c_str(), int_str_n) == 0)
			file.getField(i, 1, int_scalar);

	}
	
	return;
}




void load_annot(std::string& dataPath,
	std::vector<int>& int_scalar,
	std::string& int_scalar_name)
{
	// Variables 
	uint64_t num_points(0);
	std::vector<npm::PLYType> types;
	std::vector<std::string> properties;
	char buffer[500];

	size_t int_str_n = strlen(int_scalar_name.c_str());

	// Open file
	npm::PLYFileIn file(dataPath);

	// Read Header
	if (!file.read(&num_points, &types, &properties))
	{
		std::cout << "ERROR: wrong ply header" << std::endl;
		return;
	}

	// Prepare containers
	int_scalar.reserve(num_points);

	// Get the points
	for (size_t i = 0; i < properties.size(); i++)
	{
		if (properties[i].size() == int_str_n && strncmp(properties[i].c_str(), int_scalar_name.c_str(), int_str_n) == 0)
			file.getField(i, 1, int_scalar);
	}
	
	return;
}

void load_frame(std::string &dataPath,
				std::vector<PointXYZ> &f_pts,
				std::vector<float> &timestamps,
				std::vector<int> &rings,
				std::vector<int> &loc_labels,
				std::string &save_path,
				std::string &time_name,
				std::string &ring_name)
{

	if (loc_labels.size() > 0)
	{
		// Load annotations
		size_t i0 = dataPath.rfind(std::string("/"));
		std::string f_name = dataPath.substr(i0, dataPath.size() - i0);
		std::vector<int> categories;
		std::string int_scalar_name = "cat";
		std::string annot_path = save_path + "/tmp_frames" + f_name;

		load_annot(annot_path, categories, int_scalar_name);

		// Load raw points
		std::vector<PointXYZ> raw_f_pts;
		std::vector<float> raw_timestamps;
		std::vector<int> raw_rings;
		load_cloud(dataPath, raw_f_pts, raw_timestamps, time_name, raw_rings, ring_name);

		// Only get point from valid categories
		f_pts.reserve(raw_f_pts.size());
		timestamps.reserve(raw_timestamps.size());
		rings.reserve(raw_rings.size());
		int i = 0;
		for (auto &p : raw_f_pts)
		{
			// Add points with good labels
			if (find(loc_labels.begin(), loc_labels.end(), categories[i]) != loc_labels.end())
			{
				f_pts.push_back(p);
				timestamps.push_back(raw_timestamps[i]);
				rings.push_back(raw_rings[i]);
			}
			i++;
		}
	}
	else
	{
		load_cloud(dataPath, f_pts, timestamps, time_name, rings, ring_name);
	}
}


