
#include "pointmap_slam.h"


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
				vector<PointXYZ> &f_pts,
				vector<float> &timestamps,
				vector<int> &rings,
				vector<int> &loc_labels,
				std::string &save_path,
				std::string &time_name,
				std::string &ring_name)
{

	if (loc_labels.size() > 0)
	{
		// Load annotations
		size_t i0 = dataPath.rfind(string("/"));
		string f_name = dataPath.substr(i0, dataPath.size() - i0);
		vector<int> categories;
		string int_scalar_name = "cat";
		string annot_path = save_path + "/tmp_frames" + f_name;

		load_annot(annot_path, categories, int_scalar_name);

		// Load raw points
		vector<PointXYZ> raw_f_pts;
		vector<float> raw_timestamps;
		vector<int> raw_rings;
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

// Eigen::MatrixXd call_on_real_sequence(string &frame_names,
// 									  vector<double> &frame_times,
// 									  Eigen::MatrixXd &odom_H,
// 									  vector<PointXYZ> &init_pts,
// 									  vector<PointXYZ> &init_normals,
// 									  vector<float> &init_scores,
// 									  SLAM_params &slam_params,
// 									  string save_path)

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
				  SLAM_params &params)
{
	double timestamp_0 = frame_times[0];
	istringstream iss(frame_names);
	size_t frame_ind = 0;
	for (string line; getline(iss, line);)
	{
		if (frame_ind < start_ind)
		{
			frame_ind++;
			continue;
		}

		// Load ply file
		vector<PointXYZ> f_pts;
		vector<float> timestamps;
		vector<int> rings;
		load_frame(line, f_pts, timestamps, rings, loc_labels, save_path, time_name, ring_name);

		// Get timestamps
		float frame_time = (float)(frame_times[frame_ind] - timestamp_0);
		for (int j = 0; j < timestamps.size(); j++)
			timestamps[j] += frame_time;
		
		// Get preprocessed frame
		vector<PointXYZ> sub_pts;
		vector<PointXYZ> normals;
		vector<float> norm_scores;
		vector<double> icp_scores;
		vector<size_t> sub_inds;
		preprocess_frame(f_pts, timestamps, rings, sub_pts, normals, norm_scores, icp_scores, sub_inds, params);
		
		// Get current pose (pose of the last timestamp of the current frame in case of motion distortion)
		Eigen::Matrix4d H1 = slam_H.block(frame_ind * 4, 0, 4, 4);

		// Align frame with new pose
		Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> pts_mat((float *)sub_pts.data(), 3, sub_pts.size());
		Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> norms_mat((float *)normals.data(), 3, normals.size());
		if (params.motion_distortion)
		{
			// Init motion distortion values
			Eigen::Matrix4d H0 = slam_H.block((frame_ind - 2) * 4, 0, 4, 4);
			float t0 = slam_times[frame_ind - 2];
			float t_max = slam_times[frame_ind];

			// Apply motion distortion
			float inv_factor = 1 / (t_max - t0);
			size_t i_inds = 0;
			for (int j = 0; j < sub_inds.size(); j++)
			{
				float alpha = (timestamps[sub_inds[j]] - t0) * inv_factor;
				Eigen::Matrix4d H_rect = pose_interp(alpha, H0, H1, 0);
				Eigen::Matrix3f R_rect = (H_rect.block(0, 0, 3, 3)).cast<float>();
				Eigen::Vector3f T_rect = (H_rect.block(0, 3, 3, 1)).cast<float>();
				pts_mat.col(i_inds) = (R_rect * pts_mat.col(i_inds)) + T_rect;
				norms_mat.col(i_inds) = (R_rect * norms_mat.col(i_inds));
				i_inds++;
			}
		}
		else
		{
			// Apply transform without distortion
			Eigen::Matrix3f R_tot = (H1.block(0, 0, 3, 3)).cast<float>();
			Eigen::Vector3f T_tot = (H1.block(0, 3, 3, 1)).cast<float>();
			pts_mat = (R_tot * pts_mat).colwise() + T_tot;
			norms_mat = R_tot * norms_mat;
		}

		// Update map
		map.update(sub_pts, normals, norm_scores, frame_ind);

		frame_ind++;

		if (frame_ind > last_ind)
			break;
	}
}

// SLAM functions
// **************

void preprocess_frame(vector<PointXYZ> &f_pts,
					  vector<float> &f_ts,
					  vector<int> &f_rings,
					  vector<PointXYZ> &sub_pts,
					  vector<PointXYZ> &normals,
					  vector<float> &norm_scores,
					  vector<double> &icp_scores,
					  vector<size_t> &sub_inds,
					  SLAM_params &params)
{
	//////////////////////////////////////////
	// Preprocess frame and compute normals //
	//////////////////////////////////////////

	// Parameters

	// Create a copy of points in polar coordinates
	vector<PointXYZ> polar_pts(f_pts);
	cart2pol_(polar_pts);

	// Get angle for each lidar point, and the corresponding polar_r2
	if (params.polar_r2s.size() < 1)
	{
		vector<float> lidar_angles;
		get_lidar_angles(polar_pts, lidar_angles, params.lidar_n_lines);

		// Fill from last to first to respect ring order
		int j = lidar_angles.size() - 1;
		float tmp = 1.5 * (lidar_angles[j] - lidar_angles[j-1]);
		params.polar_r2s.push_back(pow(max(tmp, params.min_theta_radius), 2));
		j--;
		while (j > 0)
		{
			tmp = 1.5 * min(lidar_angles[j+1] - lidar_angles[j], lidar_angles[j] - lidar_angles[j-1]);
			params.polar_r2s.push_back(pow(max(tmp, params.min_theta_radius), 2));
			j--;
		}
		tmp = 1.5 * (lidar_angles[j + 1] - lidar_angles[j]);
		params.polar_r2s.push_back(pow(max(tmp, params.min_theta_radius), 2));
	}

	// Get lidar angle resolution
	float minTheta, maxTheta;
	float lidar_angle_res = get_lidar_angle_res(polar_pts, minTheta, maxTheta, params.lidar_n_lines);

	// cout << "angle_res = " << lidar_angle_res << endl;
	// cout << "angles :" << endl;
	// for (auto& angle : lidar_angles)
	// {
	// 	cout << angle << endl;
	// }

	// Define the polar neighbors radius in the scaled polar coordinates
	float polar_r = 1.5 * lidar_angle_res;

	// Apply log scale to radius coordinate (in place)
	lidar_log_radius(polar_pts, polar_r, params.r_scale);

	// // Remove outliers (only for real frames)
	// if (params.motion_distortion)
	// {
	// 	// Get an outlier score
	// 	vector<float> scores(polar_pts.size(), 0.0);
	// 	detect_outliers(polar_pts, scores, params.lidar_n_lines, lidar_angle_res, minTheta, params.outl_rjct_passes, params.outl_rjct_thresh);

	// 	// Remove points with negative score
	// 	filter_pointcloud(f_pts, scores, 0);
	// 	filter_pointcloud(polar_pts, scores, 0);
	// }

	// Get subsampling of the frame in carthesian coordinates (New points are barycenters or not?)
	grid_subsampling_centers(f_pts, sub_pts, sub_inds, params.frame_voxel_size);
	//grid_subsampling_spheres(f_pts, sub_pts, params.frame_voxel_size);

	// Convert sub_pts to polar and rescale
	vector<PointXYZ> polar_queries0(sub_pts);
	cart2pol_(polar_queries0);
	vector<PointXYZ> polar_queries(polar_queries0);
	lidar_log_radius(polar_queries, polar_r, params.r_scale);
	lidar_horizontal_scale(polar_queries, params.h_scale);

	/////////////////////
	// Compute normals //
	/////////////////////

	// Apply horizontal scaling (to have smaller neighborhoods in horizontal direction)
	lidar_horizontal_scale(polar_pts, params.h_scale);

	// Get sub_rings
	vector<int> sub_rings;
	sub_rings.reserve(sub_inds.size());
	for (int j = 0; j < sub_inds.size(); j++)
		sub_rings.push_back(f_rings[sub_inds[j]]);
	
	// Call polar processing function
	extract_lidar_frame_normals(f_pts, polar_pts, sub_pts, polar_queries, sub_rings, normals, norm_scores, params.polar_r2s);

	// // Debug (save original normal scores from PCA)
	// if (frame_i > 0)
	// {
	// 	string path000 = "/home/hth/Deep-Collison-Checker/SOGM-3D-2D-Net/results/";
	// 	char buffer00[100];
	// 	sprintf(buffer00, "f_%05d_init0.ply", int(frame_i));
		
	// 	string filepath00 = path000 + string(buffer00);
	// 	save_cloud(filepath00, sub_pts, normals, norm_scores);
	// }

	// Better normal score vased on distance and incidence angle
	smart_normal_score(sub_pts, polar_queries0, normals, norm_scores);

	// ICP score between 1.0 and 6.0 (chance of being sampled during ICP)
	icp_scores = vector<double>(norm_scores.begin(), norm_scores.end());
	smart_icp_score(polar_queries0, normals, icp_scores);

	// Remove points with a negative score
	float min_score = 0.0001;
	filter_pointcloud(sub_pts, norm_scores, min_score);
	filter_pointcloud(normals, norm_scores, min_score);
	filter_anyvector(sub_inds, norm_scores, min_score);
	filter_anyvector(icp_scores, norm_scores, min_score);
	filter_floatvector(norm_scores, min_score);
}

void PointMapSLAM::add_new_frame(vector<PointXYZ> &f_pts,
								 vector<float> &f_ts,
								 vector<int> &f_rings,
								 Eigen::Matrix4d &H_OdomToScanner,
								 int verbose)
{

	//////////////////////
	// Timing variables //
	//////////////////////

	vector<string> clock_str;
	vector<clock_t> t;
	clock_str.reserve(20);
	t.reserve(20);
	if (verbose)
	{
		clock_str.push_back("Frame to polar .... ");
		clock_str.push_back("Lidar angle res ... ");
		clock_str.push_back("Outlier reject .... ");
		clock_str.push_back("Grid subsampling .. ");
		clock_str.push_back("Sub to polar ...... ");
		clock_str.push_back("Frame normals ..... ");
		clock_str.push_back("ICP localization .. ");
		clock_str.push_back("Map update ........ ");
	}
	t.push_back(std::clock());


	//////////////////////////////////////////
	// Preprocess frame and compute normals //
	//////////////////////////////////////////

	vector<PointXYZ> sub_pts;
	vector<PointXYZ> normals;
	vector<float> norm_scores;
	vector<double> icp_scores;
	vector<size_t> sub_inds;

	preprocess_frame(f_pts, f_ts, f_rings, sub_pts, normals, norm_scores, icp_scores, sub_inds, params);

	// // Debug (Points with scores)
	// if (frame_i > 0)
	// {
	// 	string path000 = "/home/hth/Deep-Collison-Checker/SOGM-3D-2D-Net/results/";
	// 	char buffer00[100];
	// 	char buffer01[100];
	// 	char buffer02[100];
	// 	sprintf(buffer00, "f_%05d_map.ply", int(frame_i));
	// 	sprintf(buffer01, "f_%05d_init1.ply", int(frame_i));
	// 	sprintf(buffer02, "f_%05d_init2.ply", int(frame_i));
	// 	string filepath00 = path000 + string(buffer00);
	// 	string filepath01 = path000 + string(buffer01);
	// 	string filepath02 = path000 + string(buffer02);
	// 	save_cloud(filepath00, map.cloud.pts, map.normals, map.scores);
	// 	save_cloud(filepath01, sub_pts, normals, norm_scores);
		
	// 	vector<float> f1(icp_scores.begin(), icp_scores.end());
	// 	save_cloud(filepath02, sub_pts, f1);
	// }

	// Min and max times (dont loop on the whole frame as it is useless)
	float loop_ratio = 0.01;
	t_min = f_ts[0];
	t_max = f_ts[0];
	for (int j = 0; (float)j < loop_ratio * (float)f_ts.size();  j++)
	{
		if (f_ts[j] < t_min)
			t_min = f_ts[j];
	}
	for (int j = (int)floor((1 - loop_ratio) * f_ts.size()); j < f_ts.size();  j++)
	{
		if (f_ts[j] > t_max)
			t_max = f_ts[j];
	}
	
	// Init last_time
	if (frame_i < 1)
		params.icp_params.last_time = t_min - (t_max - t_min);

	// Get the motion_distorTion values from timestamps
	// 0 for the last_time (old t_min) and 1 for t_max
	vector<float> sub_alphas;
	if (params.motion_distortion)
	{
		float inv_factor = 1 / (t_max - params.icp_params.last_time);
		sub_alphas.reserve(sub_inds.size());
		for (int j = 0; j < sub_inds.size(); j++)
			sub_alphas.push_back((f_ts[sub_inds[j]] - params.icp_params.last_time) * inv_factor);
	}

	t.push_back(std::clock());

	/////////////////////////////////
	// Align frame on map with ICP //
	/////////////////////////////////

	// Create result containers
	ICP_results icp_results;
	bool warning = false;

	// If no odometry is given, use identity
	if (H_OdomToScanner.lpNorm<1>() < 0.001)
		H_OdomToScanner = Eigen::Matrix4d::Identity(4, 4);

	// Use odometry as init
	Eigen::Matrix4d H_scannerToMap_init = H_OdomToMap * H_OdomToScanner.inverse();

	if (frame_i < 1)
	{
		// In all cases if it is the first frame init the last_transform0
		params.icp_params.last_transform0 = H_scannerToMap_init;
	}

	if (params.icp_params.max_iter < 1)
	{
		// Case where we do not redo ICP but just align frames according to the given odom
		icp_results.transform = H_scannerToMap_init;
	}
	else
	{
		if (map.size() < 1)
		{
			// Case where we do not have a map yet. Override the first cloud position so that ground is at z=0
			icp_results.transform = Eigen::Matrix4d::Identity();
			icp_results.transform(2, 3) = 0.7;
			params.icp_params.last_transform0 = icp_results.transform;
		}
		else
		{
			if (frame_i < 1)
			{
				// Case where we have a map, and the first frame needs to be aligned
				// We assume robot is still in the beginning so no motion distortion

				// 1. Initial RANSAC alignment



				// 2. ICP refine
				params.icp_params.init_transform = H_scannerToMap_init;
				params.icp_params.motion_distortion = false;
				PointToMapICP(sub_pts, sub_alphas, icp_scores, map, params.icp_params, icp_results);
				params.icp_params.motion_distortion = params.motion_distortion;

				// We override last_transform0 too to neglate motion distortion for this first frame
				params.icp_params.last_transform0 = icp_results.transform;
			}
			else
			{
				params.icp_params.init_transform = H_scannerToMap_init;
				PointToMapICP(sub_pts, sub_alphas, icp_scores, map, params.icp_params, icp_results);
			}

			// Safe Check
			if (icp_results.all_plane_rms.size() > 3 * params.icp_params.avg_steps)
			{
				warning = true;
				if (icp_results.all_plane_rms.size() > 0.9 * params.icp_params.max_iter)
					cout << "ERROR: at frame " << frame_i << ", ICP not converging, num_iter = " << icp_results.all_plane_rms.size() << endl;
				else
					cout << "WARNING: at frame " << frame_i << ", ICP num_iter = " << icp_results.all_plane_rms.size() << endl;

				// Debug (Points with scores)
				string path000 = "/home/hth/Deep-Collison-Checker/SOGM-3D-2D-Net/results/";
				char buffer00[100];
				char buffer01[100];
				sprintf(buffer00, "f_%05d_%03d-iter_map.ply", int(frame_i), icp_results.all_plane_rms.size());
				sprintf(buffer01, "f_%05d_%03d-iter_init.ply", int(frame_i), icp_results.all_plane_rms.size());
				string filepath00 = path000 + string(buffer00);
				string filepath01 = path000 + string(buffer01);
				save_cloud(filepath00, map.cloud.pts, map.normals, map.scores);

				vector<float> f12(icp_scores.begin(), icp_scores.end());
				f12.insert(f12.end(), norm_scores.begin(),  norm_scores.end());
				save_cloud(filepath01, sub_pts, normals, f12);

				
			}
		}
	}
	

	// Save RMS for debug
	bool saving_rms = true;
	if (saving_rms)
	{
		string rms_path = "/home/hth/Deep-Collison-Checker/SOGM-3D-2D-Net/results/all_rms.txt";
		std::fstream outfile;
		if (frame_i < 1)
			outfile.open(rms_path, std::fstream::out);
		else
			outfile.open(rms_path, std::fstream::app);
		outfile << frame_i;
		for (auto& rms : icp_results.all_rms)
			outfile << " " << rms;
		for (auto& prms : icp_results.all_plane_rms)
			outfile << " " << prms;
		outfile << "\n";
		outfile.close();
	}

	t.push_back(std::clock());


	////////////////////
	// Update the map //
	////////////////////
	
	if (params.motion_distortion)
	{
		// Update map taking motion distortion into account
		size_t i_inds = 0;
		Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> pts_mat((float *)sub_pts.data(), 3, sub_pts.size());
		Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> norms_mat((float *)normals.data(), 3, normals.size());
		for (auto& alpha : sub_alphas)
		{
			Eigen::Matrix4d H_rect = pose_interp(alpha, params.icp_params.last_transform0, icp_results.transform, 0);
			Eigen::Matrix3f R_rect = (H_rect.block(0, 0, 3, 3)).cast<float>();
			Eigen::Vector3f T_rect = (H_rect.block(0, 3, 3, 1)).cast<float>();
			pts_mat.col(i_inds) = (R_rect * pts_mat.col(i_inds)) + T_rect;
			norms_mat.col(i_inds) = (R_rect * norms_mat.col(i_inds));
			i_inds++;
		}
	}
	else
	{
		Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> pts_mat((float *)sub_pts.data(), 3, sub_pts.size());
		Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> norms_mat((float *)normals.data(), 3, normals.size());
		Eigen::Matrix3f R_tot = (icp_results.transform.block(0, 0, 3, 3)).cast<float>();
		Eigen::Vector3f T_tot = (icp_results.transform.block(0, 3, 3, 1)).cast<float>();
		pts_mat = (R_tot * pts_mat).colwise() + T_tot;
		norms_mat = R_tot * norms_mat;
	}

	// Save the corrected sub_pts
	corrected_frame = sub_pts;
	corrected_scores = icp_scores;
	
	if (warning)
	{
		string path000 = "/home/hth/Deep-Collison-Checker/SOGM-3D-2D-Net/results/";
		char buffer02[100];
		sprintf(buffer02, "f_%05d_%03d-iter_last.ply", int(frame_i), icp_results.all_plane_rms.size());
		string filepath02 = path000 + string(buffer02);
		save_cloud(filepath02, sub_pts, normals);
	}

	// The update function is called only on subsampled points as the others have no normal
	map.update(sub_pts, normals, norm_scores, frame_i);

	// Update the last pose for future frames
	Eigen::Matrix4d H0 = params.icp_params.last_transform0;
	params.icp_params.last_transform1 = icp_results.transform;
	float alpha0 = (t_min - params.icp_params.last_time) / (t_max - params.icp_params.last_time);
	params.icp_params.last_transform0 = pose_interp(alpha0, H0, icp_results.transform, 0);
	params.icp_params.last_time = t_min;

	// Update the pose correction from map to odom
	H_OdomToMap = params.icp_params.last_transform1 * H_OdomToScanner;

	// Get rotation and translation difference
	// Eigen::Matrix3d R2 = params.icp_params.last_transform0.block(0, 0, 3, 3);
	// Eigen::Matrix3d R1 = icp_results.transform.block(0, 0, 3, 3);
	// Eigen::Vector3d T2 = params.icp_params.last_transform0.block(0, 3, 3, 1);
	// Eigen::Vector3d T1 = icp_results.transform.block(0, 3, 3, 1);
	// R1 = R2 * R1.transpose();
	// T1 = T2 - T1;
	// float T_error = T1.norm();
	// float R_error = acos((R1.trace() - 1) / 2);
	// cout << "dT = " << T_error << " / dR = " << R_error << endl;

	frame_i++;

	t.push_back(std::clock());

	////////////////////////
	// Debugging messages //
	////////////////////////

	if (verbose)
	{
		for (size_t i = 0; i < min(t.size() - 1, clock_str.size()); i++)
		{
			double duration = 1000 * (t[i + 1] - t[i]) / (double)CLOCKS_PER_SEC;
			cout << clock_str[i] << duration << " ms" << endl;
		}
		cout << endl << "***********************" << endl << endl;
	}

	return;
}





// main functions
// **************


Eigen::MatrixXd call_on_sim_sequence(string& frame_names,
	vector<double>& frame_times,
	Eigen::MatrixXd& gt_H,
	vector<double>& gt_t,
	Eigen::MatrixXd& odom_H,
	vector<PointXYZ>& init_pts,
	vector<PointXYZ>& init_normals,
	vector<float>& init_scores,
	SLAM_params& slam_params,
	string save_path)
{
	// --------------------------------------------------------------------------------
	//
	//	This function start a SLAM on a sequence of frames that come from a simulator.
	//	It uses the groundtruth to verify that the SLAM is not making any mistake.
	//
	// --------------------------------------------------------------------------------

	////////////////////////
	// Initiate variables //
	////////////////////////

	// Create a the SLAM class
	PointMapSLAM mapper(slam_params, init_pts, init_normals, init_scores);

	// Results container
	Eigen::MatrixXd all_H = Eigen::MatrixXd::Zero(4 * frame_times.size(), 4);

	// Timing
	float fps = 0.0;
	float fps_regu = 0.9;


	////////////////
	// Start SLAM //
	////////////////

	// Frame index
	size_t frame_i = 0;
	size_t gt_i0 = 0;
	size_t gt_i1 = 0;
	clock_t last_disp_t1 = clock();

	// Loop on the lines of "frame_names" string
	istringstream iss(frame_names);
	for (string line; getline(iss, line);)
	{

		// Load frame
		// **********

		// Load ply file
		vector<int> loc_labels = {0, 1, 2, 3};
		vector<PointXYZ> f_pts;

		if (slam_params.filtering)
		{
			vector<PointXYZ> raw_f_pts;
			vector<float> float_scalar;
			vector<int> categories;
			string float_scalar_name = "";
			string int_scalar_name = "cat";
			load_cloud(line, raw_f_pts, float_scalar, float_scalar_name, categories, int_scalar_name);

			// Only get point from valid categories
			f_pts.reserve(raw_f_pts.size());
			int i = 0;
			for (auto& p: raw_f_pts)
			{
				// Add points with good labels
				if (find(loc_labels.begin(), loc_labels.end(), categories[i]) != loc_labels.end())
					f_pts.push_back(p);
				i++;
			}
		}
		else
		{
			load_cloud(line, f_pts);
		}


		// Get GT pose for debug
		// *********************
		//	(specific to simulation data)

		// Find closest groundtruth
		while (gt_t[gt_i1] < frame_times[frame_i])
			gt_i1++;
		if (gt_i1 > 0)
			gt_i0 = gt_i1 - 1;

		// Interpolate frame pose between gt poses
		float interp_t = (frame_times[frame_i] - gt_t[gt_i0]) / (gt_t[gt_i1] - gt_t[gt_i0]);
		Eigen::Matrix4d gt_init_H = pose_interp(interp_t, gt_H.block(gt_i0 * 4, 0, 4, 4), gt_H.block(gt_i1 * 4, 0, 4, 4), 0);

		// Offset the velodyne pose
		gt_init_H = gt_init_H * slam_params.H_velo_base;


		// Map this frame
		// **************

		clock_t t0 = clock();

		// Get odometry matrix (pose of the scanner in odometry world frame)
		Eigen::Matrix4d H_OdomToScanner = odom_H.block(frame_i * 4, 0, 4, 4);

		// Get frame pose and update map
		vector<float> timestamps;
		vector<int> f_rings;
		mapper.add_new_frame(f_pts, timestamps, f_rings, H_OdomToScanner, 0);

		// Save transform
		all_H.block(frame_i * 4, 0, 4, 4) = mapper.last_H;

		clock_t t1 = clock();

		// Debug: compare pose to gt
		// *************************

		// Get rotation and translation error
		Eigen::Matrix3d R2 = gt_init_H.block(0, 0, 3, 3);
		Eigen::Matrix3d R1 = mapper.last_H.block(0, 0, 3, 3);
		Eigen::Vector3d T2 = gt_init_H.block(0, 3, 3, 1);
		Eigen::Vector3d T1 = mapper.last_H.block(0, 3, 3, 1);
		R1 = R2 * R1.transpose();
		T1 = T2 - T1;
		float T_error = T1.norm();
		float R_error = acos((R1.trace() - 1) / 2);

		if (T_error > 1.0 || R_error > 10.0 * M_PI / 180)
		{
			cout << endl << "*************************************************" << endl;
			cout << "Error in the mapping:" << endl;
			cout << "T_error = " << T_error << endl;
			cout << "R_error = " << R_error << endl;
			cout << "Stopping mapping and saving debug frames" << endl;
			char buffer[100];
			sprintf(buffer, "Frame %d named %s", (int)frame_i, line);
			cout << string(buffer) << endl;
			cout <<"*************************************************" << endl << endl;
			save_cloud("debug_map.ply", mapper.map.cloud.pts, mapper.map.normals, mapper.map.scores);
			break;
		}

		// if (frame_i % 100 == 0)
		// {
		// 	char buffer[100];
		// 	sprintf(buffer, "cc_map_%05d.ply", (int)frame_i);
		// 	vector<float> counts(mapper.map.counts.begin(), mapper.map.counts.end());
		// 	counts.insert(counts.end(), mapper.map.scores.begin(),  mapper.map.scores.end());
		// 	save_cloud(string(buffer), mapper.map.cloud.pts, mapper.map.normals, counts);
		// }

		// Timing
		// ******

		double duration = (t1 - t0) / (double)CLOCKS_PER_SEC;
		fps = fps_regu * fps + (1.0 - fps_regu) / duration;

		if (slam_params.verbose_time > 0 && (t1 - last_disp_t1) / (double)CLOCKS_PER_SEC > slam_params.verbose_time)
		{
			double remaining_sec = (frame_times.size() - frame_i) / fps;
			int remaining_min = (int)floor(remaining_sec / 60.0);
			remaining_sec = remaining_sec - remaining_min * 60.0;
			char buffer[100];
			sprintf(buffer, "Mapping %5d/%d at %5.1f fps - %d min %.0f sec remaining", (int)frame_i, frame_times.size(), fps, remaining_min, remaining_sec);
			cout << string(buffer) << endl;
			last_disp_t1 = t1;
		}

		frame_i++;

	}


	// Save map in a ply file init containers with results
	size_t ns1 = 19;
	size_t ns0 = save_path.size() - ns1;
	string day_str = save_path.substr(ns0, ns1);
	vector<float> counts(mapper.map.counts.begin(), mapper.map.counts.end());
	counts.insert(counts.end(), mapper.map.scores.begin(),  mapper.map.scores.end());
	save_cloud(save_path + "/map_" + day_str + ".ply", mapper.map.cloud.pts, mapper.map.normals, counts);

	return all_H;
}



Eigen::MatrixXd call_on_real_sequence(string& frame_names,
	vector<double>& frame_times,
	Eigen::MatrixXd& odom_H,
	vector<PointXYZ>& init_pts,
	vector<PointXYZ>& init_normals,
	vector<float>& init_scores,
	SLAM_params& slam_params,
	string save_path)
{
	// --------------------------------------------------------------------------------
	//
	//	This function start a SLAM on a sequence of frames that come from a simulator.
	//	It uses the groundtruth to verify that the SLAM is not making any mistake.
	//
	// --------------------------------------------------------------------------------

	////////////////////////
	// Initiate variables //
	////////////////////////

	// Create a the SLAM class
	PointMapSLAM mapper(slam_params, init_pts, init_normals, init_scores);

	// Results container
	Eigen::MatrixXd all_H = Eigen::MatrixXd::Zero(4 * frame_times.size(), 4);
	vector<float> all_times;
	vector<PointXYZ> sparse_positions;
	vector<size_t> sparse_f_inds;

	// Number of closed loops
	int closed_loops = 0;

	// Timing
	float fps = 0.0;
	float fps_regu = 0.9;

	// Initial timestamp for motion distorsiob
	double timestamp_0 = frame_times[0];

	////////////////
	// Start SLAM //
	////////////////
	
	// Parameters
	vector<int> loc_labels;
	if (slam_params.filtering)
		loc_labels = {0, 1, 2, 3};
	std::string time_name = "time";
	std::string ring_name = "ring";

	// Frame index
	size_t frame_ind = 0;
	clock_t last_disp_t1 = clock();

	// Loop on the lines of "frame_names" string
	istringstream iss(frame_names);
	for (string line; getline(iss, line);)
	{
		// Load frame
		// **********

		// Load ply file
		vector<PointXYZ> f_pts;
		vector<float> timestamps;
		vector<int> rings;
		load_frame(line, f_pts, timestamps, rings, loc_labels, save_path, time_name, ring_name);

		// Map this frame
		// **************

		clock_t t0 = clock();

		// Get odometry matrix (pose of the scanner in odometry world frame)
		Eigen::Matrix4d H_OdomToScanner = odom_H.block(frame_ind * 4, 0, 4, 4);

		// Get timestamps
		float frame_time = (float)(frame_times[frame_ind] - timestamp_0);
		for (int j = 0; j < timestamps.size(); j++)
			timestamps[j] += frame_time;

		// Get frame pose and update map
		if (slam_params.verbose_time > 0 && slam_params.verbose_time < 0.001)
			mapper.add_new_frame(f_pts, timestamps, rings, H_OdomToScanner, 1);
		else
			mapper.add_new_frame(f_pts, timestamps, rings, H_OdomToScanner, 0);

		// Save transform
		all_H.block(frame_ind * 4, 0, 4, 4) = mapper.params.icp_params.last_transform1;
		all_times.push_back(mapper.t_max);

		// Save position for loop closure
		float closure_d = 1.0;
		float closure_d2 =  closure_d * closure_d;
		float save_d = closure_d / 2;
		float save_d2 = save_d * save_d;
		float closure_t = 20.0;
		PointXYZ current_position(mapper.params.icp_params.last_transform1(0, 3),
								  mapper.params.icp_params.last_transform1(1, 3),
								  mapper.params.icp_params.last_transform1(2, 3));
		if (sparse_positions.size() == 0 || (current_position - sparse_positions.back()).sq_norm() > save_d2)
		{
			sparse_positions.push_back(current_position);
			sparse_f_inds.push_back(frame_ind);
		}

		clock_t t1 = clock();

		// Loop closure
		// ************

		// Very simple detection by checking if we come back to the same place
		int closure_ind = -1;
		if (closed_loops < 1)
		{
			for (size_t i = 0; i < sparse_positions.size(); i++)
			{
				if ((frame_time - all_times[sparse_f_inds[i]]) > closure_t)
				{
					PointXYZ diff = sparse_positions[i] - current_position;
					diff.z = 0;
					float d2 = diff.sq_norm();

					if (d2 < closure_d2)
					{
						closure_ind = (int)sparse_f_inds[i];
						break;
					}
				}
			}
		}

		// Close loop if necessary
		if (closure_ind >= 0)
		{
			cout << "\n  >>> Loop detected. Performing closure" << endl;

			string path000 = "/home/hth/Deep-Collison-Checker/SOGM-3D-2D-Net/results/";
			char buffer00[100];
			sprintf(buffer00, "f_%05d_map_before.ply", int(frame_ind));
			vector<float> all_features(mapper.map.counts.begin(), mapper.map.counts.end());
			all_features.insert(all_features.end(), mapper.map.scores.begin(),  mapper.map.scores.end());
			save_cloud(path000 + string(buffer00), mapper.map.cloud.pts, mapper.map.normals, all_features);

			// 1. Recreate a new cleam map from current one until closure_ind
			PointMap clean_map(mapper.map, closure_ind);
			cout << "\n  >>> Map cleaned" << endl;
			
			char buffer01[100];
			sprintf(buffer01, "f_%05d_map_clean.ply", int(frame_ind));
			vector<float> all_features1(clean_map.counts.begin(), clean_map.counts.end());
			all_features1.insert(all_features1.end(), clean_map.scores.begin(),  clean_map.scores.end());
			save_cloud(path000 + string(buffer01), clean_map.cloud.pts, clean_map.normals, all_features1);

			// 2. Perform ICP with this frame on this clean map
			ICP_results icp_results;
			mapper.params.icp_params.init_transform = Eigen::Matrix4d::Identity(4, 4);
			vector<float> alphas;

			//
			// TODO: HERE initial transform for ICP 
			//

			mapper.params.icp_params.motion_distortion = false;
			PointToMapICP(mapper.corrected_frame, alphas, mapper.corrected_scores, clean_map, mapper.params.icp_params, icp_results);
			mapper.params.icp_params.motion_distortion = mapper.params.motion_distortion;


			cout << "\n  >>> Loop closed" << endl;

			// 3. Correct all transforms (assumnes a constant drift)
			float inv_factor = 1.0 / (float)(frame_ind - closure_ind);
			for (size_t i = closure_ind + 1; i <= frame_ind; i++)
			{
				float t = (float)(i - closure_ind) * inv_factor;
				Eigen::Matrix4d dH = pose_interp(t, Eigen::Matrix4d::Identity(4, 4), icp_results.transform, 0);
				all_H.block(i * 4, 0, 4, 4) = dH * all_H.block(i * 4, 0, 4, 4);
			}
			mapper.params.icp_params.last_transform1 = icp_results.transform * mapper.params.icp_params.last_transform1;
			mapper.params.icp_params.last_transform0 = icp_results.transform * mapper.params.icp_params.last_transform0;
			mapper.H_OdomToMap = icp_results.transform * mapper.H_OdomToMap;

			cout << "\n  >>> Transforms corrected" << endl;

			// 4. Reupdate map with new transforms until current frame
			complete_map(frame_names,
						 frame_times,
						 all_H,
						 all_times,
						 clean_map,
						 loc_labels,
						 save_path,
						 time_name,
						 ring_name,
						 closure_ind + 1,
						 frame_ind,
						 mapper.params);

			// Update mapper.map
			mapper.map = clean_map;
			closed_loops++;
			cout << "\n  >>> Map updated" << endl;

			char buffer02[100];
			sprintf(buffer02, "f_%05d_map_after.ply", int(frame_ind));
			vector<float> all_features2(mapper.map.counts.begin(), mapper.map.counts.end());
			all_features2.insert(all_features2.end(), mapper.map.scores.begin(),  mapper.map.scores.end());
			save_cloud(path000 + string(buffer02), mapper.map.cloud.pts, mapper.map.normals, all_features2);
		}


		// // Debug: compare pose to gt
		// // *************************

		// if (frame_ind % 100 == 0)
		// {
		// 	char buffer[100];
		// 	sprintf(buffer, "cc_map_%05d.ply", (int)frame_ind);
		// 	vector<float> counts(mapper.map.counts.begin(), mapper.map.counts.end());
		// 	counts.insert(counts.end(), mapper.map.scores.begin(),  mapper.map.scores.end());
		// 	save_cloud(string(buffer), mapper.map.cloud.pts, mapper.map.normals, counts);
		// }

		// Timing
		// ******

		double duration = (t1 - t0) / (double)CLOCKS_PER_SEC;
		fps = fps_regu * fps + (1.0 - fps_regu) / duration;

		if (slam_params.verbose_time > 0 && (t1 - last_disp_t1) / (double)CLOCKS_PER_SEC > slam_params.verbose_time)
		{
			double remaining_sec = (frame_times.size() - frame_ind) / fps;
			int remaining_min = (int)floor(remaining_sec / 60.0);
			remaining_sec = remaining_sec - remaining_min * 60.0;
			char buffer[100];
			sprintf(buffer, "Mapping %5d/%d at %5.1f fps - %d min %.0f sec remaining", (int)frame_ind, frame_times.size(), fps, remaining_min, remaining_sec);
			cout << string(buffer) << endl;
			last_disp_t1 = t1;
		}

		frame_ind++;

		// if (frame_ind > 2)
		// 	break;

	}


	// Save map in a ply file init containers with results
	size_t ns1 = 19;
	size_t ns0 = save_path.size() - ns1;
	string day_str = save_path.substr(ns0, ns1);
	vector<float> counts(mapper.map.counts.begin(), mapper.map.counts.end());
	counts.insert(counts.end(), mapper.map.scores.begin(),  mapper.map.scores.end());
	save_cloud(save_path + "/map_" + day_str + ".ply", mapper.map.cloud.pts, mapper.map.normals, counts);

	return all_H;
}



// Eigen::MatrixXd call_on_real_sequence(string& frame_names,
// 	vector<double>& frame_times,
// 	Eigen::MatrixXd& odom_H,
// 	vector<PointXYZ>& init_pts,
// 	vector<PointXYZ>& init_normals,
// 	vector<float>& init_scores,
// 	SLAM_params& slam_params,
// 	string save_path)
// {
// 	// --------------------------------------------------------------------------------
// 	//
// 	//	This function start a SLAM on a sequence of frames that come from a simulator.
// 	//	It uses the groundtruth to verify that the SLAM is not making any mistake.
// 	//
// 	// --------------------------------------------------------------------------------

// 	////////////////////////
// 	// Initiate variables //
// 	////////////////////////

// 	// Create a the SLAM class
// 	PointMapSLAM mapper(slam_params, init_pts, init_normals, init_scores);

// 	// Results container
// 	Eigen::MatrixXd all_H = Eigen::MatrixXd::Zero(4 * frame_times.size(), 4);
// 	vector<float> all_times;
// 	vector<PointXYZ> sparse_positions;
// 	vector<size_t> sparse_f_inds;

// 	// Number of closed loops
// 	int closed_loops = 0;

// 	// Timing
// 	float fps = 0.0;
// 	float fps_regu = 0.9;

// 	// Initial timestamp for motion distorsiob
// 	double timestamp_0 = frame_times[0];

// 	////////////////
// 	// Start SLAM //
// 	////////////////
	
// 	// Parameters
// 	vector<int> loc_labels;
// 	if (slam_params.filtering)
// 		loc_labels = {0, 1, 2, 3};
// 	std::string time_name = "time";
// 	std::string ring_name = "ring";

// 	// Frame index
// 	size_t frame_ind = 0;
// 	clock_t last_disp_t1 = clock();

// 	// Loop on the lines of "frame_names" string
// 	istringstream iss(frame_names);
// 	for (string line; getline(iss, line);)
// 	{
// 		// Load frame
// 		// **********

// 		// Load ply file
// 		vector<PointXYZ> f_pts;
// 		vector<float> timestamps;
// 		vector<int> rings;
// 		load_frame(line, f_pts, timestamps, rings, loc_labels, save_path, time_name, ring_name);

// 		// Map this frame
// 		// **************

// 		clock_t t0 = clock();

// 		// Get odometry matrix (pose of the scanner in odometry world frame)
// 		Eigen::Matrix4d H_OdomToScanner = odom_H.block(frame_ind * 4, 0, 4, 4);

// 		// Get timestamps
// 		float frame_time = (float)(frame_times[frame_ind] - timestamp_0);
// 		for (int j = 0; j < timestamps.size(); j++)
// 			timestamps[j] += frame_time;

// 		// Get frame pose and update map
// 		if (slam_params.verbose_time > 0 && slam_params.verbose_time < 0.001)
// 			mapper.add_new_frame(f_pts, timestamps, rings, H_OdomToScanner, 1);
// 		else
// 			mapper.add_new_frame(f_pts, timestamps, rings, H_OdomToScanner, 0);

// 		// Save transform
// 		all_H.block(frame_ind * 4, 0, 4, 4) = mapper.params.icp_params.last_transform1;
// 		all_times.push_back(mapper.t_max);

// 		// Save position for loop closure
// 		float closure_d = 1.0;
// 		float closure_d2 =  closure_d * closure_d;
// 		float save_d = closure_d / 2;
// 		float save_d2 = save_d * save_d;
// 		float closure_t = 20.0;
// 		PointXYZ current_position(mapper.params.icp_params.last_transform1(0, 3),
// 								  mapper.params.icp_params.last_transform1(1, 3),
// 								  mapper.params.icp_params.last_transform1(2, 3));
// 		if (sparse_positions.size() == 0 || (current_position - sparse_positions.back()).sq_norm() > save_d2)
// 		{
// 			sparse_positions.push_back(current_position);
// 			sparse_f_inds.push_back(frame_ind);
// 		}

// 		clock_t t1 = clock();

// 		// Loop closure
// 		// ************

// 		// Very simple detection by checking if we come back to the same place
// 		int closure_ind = -1;
// 		if (closed_loops < 1)
// 		{
// 			for (size_t i = 0; i < sparse_positions.size(); i++)
// 			{
// 				if ((frame_time - all_times[sparse_f_inds[i]]) > closure_t)
// 				{
// 					PointXYZ diff = sparse_positions[i] - current_position;
// 					diff.z = 0;
// 					float d2 = diff.sq_norm();

// 					if (d2 < closure_d2)
// 					{
// 						closure_ind = (int)sparse_f_inds[i];
// 						break;
// 					}
// 				}
// 			}
// 		}

// 		// Close loop if necessary
// 		if (closure_ind >= 0)
// 		{
// 			cout << "\n  >>> Loop detected. Performing closure" << endl;

// 			string path000 = "/home/hth/Deep-Collison-Checker/SOGM-3D-2D-Net/results/";
// 			char buffer00[100];
// 			sprintf(buffer00, "f_%05d_map_before.ply", int(frame_ind));
// 			vector<float> all_features(mapper.map.counts.begin(), mapper.map.counts.end());
// 			all_features.insert(all_features.end(), mapper.map.scores.begin(),  mapper.map.scores.end());
// 			save_cloud(path000 + string(buffer00), mapper.map.cloud.pts, mapper.map.normals, all_features);

// 			// 1. Recreate a new cleam map from current one until closure_ind
// 			PointMap clean_map(mapper.map, closure_ind);
// 			cout << "\n  >>> Map cleaned" << endl;
			
// 			char buffer01[100];
// 			sprintf(buffer01, "f_%05d_map_clean.ply", int(frame_ind));
// 			vector<float> all_features1(clean_map.counts.begin(), clean_map.counts.end());
// 			all_features1.insert(all_features1.end(), clean_map.scores.begin(),  clean_map.scores.end());
// 			save_cloud(path000 + string(buffer01), clean_map.cloud.pts, clean_map.normals, all_features1);

// 			// 2. Perform ICP with this frame on this clean map
// 			ICP_results icp_results;
// 			mapper.params.icp_params.init_transform = Eigen::Matrix4d::Identity(4, 4);
// 			vector<float> alphas;

// 			//
// 			// TODO: HERE initial transform for ICP 
// 			//

// 			mapper.params.icp_params.motion_distortion = false;
// 			PointToMapICP(mapper.corrected_frame, alphas, mapper.corrected_scores, clean_map, mapper.params.icp_params, icp_results);
// 			mapper.params.icp_params.motion_distortion = mapper.params.motion_distortion;


// 			cout << "\n  >>> Loop closed" << endl;

// 			// 3. Correct all transforms (assumnes a constant drift)
// 			float inv_factor = 1.0 / (float)(frame_ind - closure_ind);
// 			for (size_t i = closure_ind + 1; i <= frame_ind; i++)
// 			{
// 				float t = (float)(i - closure_ind) * inv_factor;
// 				Eigen::Matrix4d dH = pose_interp(t, Eigen::Matrix4d::Identity(4, 4), icp_results.transform, 0);
// 				all_H.block(i * 4, 0, 4, 4) = dH * all_H.block(i * 4, 0, 4, 4);
// 			}
// 			mapper.params.icp_params.last_transform1 = icp_results.transform * mapper.params.icp_params.last_transform1;
// 			mapper.params.icp_params.last_transform0 = icp_results.transform * mapper.params.icp_params.last_transform0;
// 			mapper.H_OdomToMap = icp_results.transform * mapper.H_OdomToMap;

// 			cout << "\n  >>> Transforms corrected" << endl;

// 			// 4. Reupdate map with new transforms until current frame
// 			complete_map(frame_names,
// 						 frame_times,
// 						 all_H,
// 						 all_times,
// 						 clean_map,
// 						 loc_labels,
// 						 save_path,
// 						 time_name,
// 						 ring_name,
// 						 closure_ind + 1,
// 						 frame_ind,
// 						 mapper.params);

// 			// Update mapper.map
// 			mapper.map = clean_map;
// 			closed_loops++;
// 			cout << "\n  >>> Map updated" << endl;

// 			char buffer02[100];
// 			sprintf(buffer02, "f_%05d_map_after.ply", int(frame_ind));
// 			vector<float> all_features2(mapper.map.counts.begin(), mapper.map.counts.end());
// 			all_features2.insert(all_features2.end(), mapper.map.scores.begin(),  mapper.map.scores.end());
// 			save_cloud(path000 + string(buffer02), mapper.map.cloud.pts, mapper.map.normals, all_features2);
// 		}


// 		// // Debug: compare pose to gt
// 		// // *************************

// 		// if (frame_ind % 100 == 0)
// 		// {
// 		// 	char buffer[100];
// 		// 	sprintf(buffer, "cc_map_%05d.ply", (int)frame_ind);
// 		// 	vector<float> counts(mapper.map.counts.begin(), mapper.map.counts.end());
// 		// 	counts.insert(counts.end(), mapper.map.scores.begin(),  mapper.map.scores.end());
// 		// 	save_cloud(string(buffer), mapper.map.cloud.pts, mapper.map.normals, counts);
// 		// }

// 		// Timing
// 		// ******

// 		double duration = (t1 - t0) / (double)CLOCKS_PER_SEC;
// 		fps = fps_regu * fps + (1.0 - fps_regu) / duration;

// 		if (slam_params.verbose_time > 0 && (t1 - last_disp_t1) / (double)CLOCKS_PER_SEC > slam_params.verbose_time)
// 		{
// 			double remaining_sec = (frame_times.size() - frame_ind) / fps;
// 			int remaining_min = (int)floor(remaining_sec / 60.0);
// 			remaining_sec = remaining_sec - remaining_min * 60.0;
// 			char buffer[100];
// 			sprintf(buffer, "Mapping %5d/%d at %5.1f fps - %d min %.0f sec remaining", (int)frame_ind, frame_times.size(), fps, remaining_min, remaining_sec);
// 			cout << string(buffer) << endl;
// 			last_disp_t1 = t1;
// 		}

// 		frame_ind++;

// 		// if (frame_ind > 2)
// 		// 	break;

// 	}


// 	// Save map in a ply file init containers with results
// 	size_t ns1 = 19;
// 	size_t ns0 = save_path.size() - ns1;
// 	string day_str = save_path.substr(ns0, ns1);
// 	vector<float> counts(mapper.map.counts.begin(), mapper.map.counts.end());
// 	counts.insert(counts.end(), mapper.map.scores.begin(),  mapper.map.scores.end());
// 	save_cloud(save_path + "/map_" + day_str + ".ply", mapper.map.cloud.pts, mapper.map.normals, counts);

// 	return all_H;
// }



