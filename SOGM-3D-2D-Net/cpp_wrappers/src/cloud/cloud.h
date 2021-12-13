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
//		Cloud header
//
//----------------------------------------------------
//
//		Hugues THOMAS - 10/02/2017
//


# pragma once

#include <Eigen/Eigenvalues>

#include "points.h"
#include <chrono>
#include <random>


//------------------------------------------------------------------------------------------------------------
// PointCloud class
// ****************
//
//------------------------------------------------------------------------------------------------------------

struct PointCloud
{

	std::vector<PointXYZ>  pts;

	// Must return the number of data points
	inline size_t kdtree_get_point_count() const { return pts.size(); }

	// Returns the dim'th component of the idx'th point in the class:
	// Since this is inlined and the "dim" argument is typically an immediate value, the
	//  "if/else's" are actually solved at compile time.
	inline float kdtree_get_pt(const size_t idx, const size_t dim) const
	{
		if (dim == 0) return pts[idx].x;
		else if (dim == 1) return pts[idx].y;
		else return pts[idx].z;
	}

	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

};


// Utility function for pointclouds
void filter_pointcloud(std::vector<PointXYZ>& pts, std::vector<float>& scores, float filter_value);
void filter_floatvector(std::vector<float>& vec, float filter_value);

template <class T>
void filter_anyvector(std::vector<T>& vec, std::vector<float>& scores, float filter_value)
{
	// Remove every element whose score is < filter_value
	auto vec_address = vec.data();
	vec.erase(std::remove_if(vec.begin(), vec.end(),
		[&scores, vec_address, filter_value](const T& f) { return scores[(size_t)(&f - vec_address)] < filter_value; }),
		vec.end());
}

// PLY reading/saving functions
void save_cloud(std::string dataPath, std::vector<PointXYZ>& points, std::vector<PointXYZ>& normals, std::vector<float>& features);
void save_cloud(std::string dataPath, std::vector<PointXYZ>& points, std::vector<float>& features);
void save_cloud(std::string dataPath, std::vector<PointXYZ>& points, std::vector<PointXYZ>& normals);
void save_cloud(std::string dataPath, std::vector<PointXYZ>& points);


void load_cloud(std::string& dataPath,
	std::vector<PointXYZ>& points);

void load_cloud(std::string& dataPath,
	std::vector<PointXYZ>& points,
	std::vector<float>& float_scalar,
	std::string& float_scalar_name,
	std::vector<int>& int_scalar,
	std::string& int_scalar_name);

void load_annot(std::string &dataPath,
				std::vector<int> &int_scalar,
				std::string &int_scalar_name);

void load_frame(std::string &dataPath,
				std::vector<PointXYZ> &f_pts,
				std::vector<float> &timestamps,
				std::vector<int> &rings,
				std::vector<int> &loc_labels,
				std::string &save_path,
				std::string &time_name,
				std::string &ring_name);

void random_3_pick(int &A_i, int &B_i, int &C_i,
				   std::uniform_int_distribution<int> &distribution,
				   std::default_random_engine &generator);

bool is_triplet_bad(PointXYZ &A, PointXYZ &B, PointXYZ &C, PointXYZ &u);

Plane3D plane_ransac(std::vector<PointXYZ> &points,
					 float max_dist = 0.1,
					 int max_steps = 100);
					 
Plane3D frame_ground_ransac(std::vector<PointXYZ> &points,
							std::vector<PointXYZ> &normals,
							float vertical_thresh_deg = 10.0,
							float max_dist = 0.1,
							float ground_z = 0.0);

bool rot_u_to_v(PointXYZ u, PointXYZ v, Eigen::Matrix3d &R);

// float tukey(float x);