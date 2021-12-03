#pragma once

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <random>
#include <unordered_set>
#include <numeric>
#include <fstream>


#define _USE_MATH_DEFINES
#include <math.h>

#include "ceres/ceres.h"
#include "glog/logging.h"

#include <Eigen/Eigenvalues>
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

	// max distance travelled before frames are removed from local map
	float local_map_dist;

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
	vector<float> polar_r2s;
	float min_theta_radius;

	// Methods
	// *******

	// Constructor
	SLAM_params()
	{
		lidar_n_lines = 32;
		min_theta_radius = 0.015;
		map_voxel_size = 0.08;
		frame_voxel_size = 0.2;
		local_map_dist = 10.0;
		motion_distortion = false;
		filtering = false;
		verbose_time = -1;
		H_velo_base = Eigen::Matrix4d::Identity(4, 4);

		h_scale = 0.3;
		r_scale = 10.0;
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
	PointMap map0;

	// Pose of the last mapped frame
	Eigen::Matrix4d last_H;

	// Current pose correction from odometry to map
	Eigen::Matrix4d H_OdomToMap;

	// Indice of frame
	int frame_i;

	// Count errors to stop if this is not going well
	int warning_count;

	// Container for the motion corrected frame used to update the map
	vector<PointXYZ> corrected_frame;
	vector<double> corrected_scores;
	float t_min, t_max;

	// Methods
	// *******

	// Constructor
	PointMapSLAM(SLAM_params slam_params0, vector<PointXYZ> &init_points, vector<PointXYZ> &init_normals, vector<float> &init_scores)
	{
		// Init paramters
		params = slam_params0;

		//// Init map from previous session
		map.dl = params.map_voxel_size;
		map0.dl = params.map_voxel_size;
		if (init_points.size() > 0)
		{
			map0.update_idx = -1;
			map0.update(init_points, init_normals, init_scores, -1);
			map.update_idx = -1;
			map.update(init_points, init_normals, init_scores, -1);
		}

		// Dummy first last_H
		last_H = Eigen::Matrix4d::Identity(4, 4);
		H_OdomToMap = Eigen::Matrix4d::Identity(4, 4);
		frame_i = 0;
		warning_count = 0;
	}

	// Mapping functions
	void init_map() { return; }
	void add_new_frame(vector<PointXYZ> &f_pts, vector<float>& f_ts, vector<int>& f_rings, Eigen::Matrix4d &H_OdomToScanner, int verbose = 0);
};

// Function declaration
// ********************

void complete_map(string &frame_names,
				  vector<double> &frame_times,
				  Eigen::MatrixXd &slam_H,
				  vector<float> &slam_times,
				  PointMap& map,
				  vector<int> &loc_labels,
				  std::string &save_path,
				  std::string &time_name,
				  std::string &ring_name,
				  size_t start_ind,
				  size_t last_ind,
				  SLAM_params &params);

void preprocess_frame(vector<PointXYZ> &f_pts,
					  vector<float> &f_ts,
					  vector<int> &f_rings,
					  vector<PointXYZ> &sub_pts,
					  vector<PointXYZ> &normals,
					  vector<float> &norm_scores,
					  vector<double> &icp_scores,
					  vector<size_t> &sub_inds,
					  Plane3D &frame_ground,
					  SLAM_params &params);

void ceres_hello();

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

Eigen::MatrixXd call_on_real_sequence(string &frame_names,
									  vector<double> &frame_times,
									  Eigen::MatrixXd &odom_H,
									  vector<PointXYZ> &init_pts,
									  vector<PointXYZ> &init_normals,
									  vector<float> &init_scores,
									  SLAM_params &slam_params,
									  string save_path);

