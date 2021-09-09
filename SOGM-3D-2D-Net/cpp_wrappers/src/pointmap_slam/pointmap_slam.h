#pragma once

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <random>
#include <unordered_set>
#include <numeric>

#define _USE_MATH_DEFINES
#include <math.h>

#include <../Eigen/Eigenvalues>
#include "../cloud/cloud.h"
#include "../nanoflann/nanoflann.hpp"

#include "../grid_subsampling/grid_subsampling.h"
#include "../polar_processing/polar_processing.h"
#include "../pointmap/pointmap.h"
#include "../icp/icp.h"

using namespace std;

// KDTree type definition
typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, 3> PointXYZ_KDTree;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

// ICP params and result classes
// *****************************

class SLAM_params
{
public:
	// Elements
	// ********

	// Number of lines of scan in the lidar
	int lidar_n_lines;

	// Size of the map voxels
	float map_voxel_size;

	// Size of the voxels for frame subsampling
	float frame_voxel_size;

	// Account for motion distortion (fasle in the case of simulated data)
	bool motion_distortion;

	// Are we filtering frames
	bool filtering;

	// Verbose option (time in sec between each verbose negative for no verbose)
	float verbose_time;

	// Transformation matrix from velodyne frame to base frame
	Eigen::Matrix4d H_velo_base;

	// Params of ICP used in this SLAM
	ICP_params icp_params;

	// Params of frame normal computation
	float h_scale;
	float r_scale;
	int outl_rjct_passes;
	float outl_rjct_thresh;

	// Methods
	// *******

	// Constructor
	SLAM_params()
	{
		lidar_n_lines = 32;
		map_voxel_size = 0.08;
		frame_voxel_size = 0.2;
		motion_distortion = false;
		filtering = false;
		verbose_time = -1;
		H_velo_base = Eigen::Matrix4d::Identity(4, 4);

		h_scale = 0.5;
		r_scale = 4.0;
		outl_rjct_passes = 2;
		outl_rjct_thresh = 0.003;
	}
};

class PointMapSLAM
{
public:
	// Elements
	// ********

	// Parameters
	SLAM_params params;

	// Map used by the algorithm
	PointMap map;

	// Pose of the last mapped frame
	Eigen::Matrix4d last_H;

	// Current pose correction from odometry to map
	Eigen::Matrix4d H_OdomToMap;

	// Methods
	// *******

	// Constructor
	PointMapSLAM(SLAM_params slam_params0, vector<PointXYZ> &init_points, vector<PointXYZ> &init_normals, vector<float> &init_scores)
	{
		// Init paramters
		params = slam_params0;

		//// Init map from previous session
		map.dl = params.map_voxel_size;
		if (init_points.size() > 0)
		{
			map.update_idx = -1;
			map.update(init_points, init_normals, init_scores);
		}

		// Dummy first last_H
		last_H = Eigen::Matrix4d::Identity(4, 4);
		H_OdomToMap = Eigen::Matrix4d::Identity(4, 4);
	}

	// Mapping functions
	void init_map() { return; }
	void add_new_frame(vector<PointXYZ> &f_pts, Eigen::Matrix4d &H_OdomToScanner, int verbose = 0);
};

// Function declaration
// ********************

Eigen::MatrixXd call_on_sim_sequence(string &frame_names,
									 vector<double> &frame_times,
									 Eigen::MatrixXd &gt_H,
									 vector<double> &gt_t,
									 Eigen::MatrixXd &odom_H,
									 vector<PointXYZ> &init_pts,
									 vector<PointXYZ> &init_normals,
									 vector<float> &init_scores,
									 SLAM_params &slam_params,
									 string save_path);
