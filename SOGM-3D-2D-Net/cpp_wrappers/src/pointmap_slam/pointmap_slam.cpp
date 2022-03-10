
#include "pointmap_slam.h"



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
		for (int j = 0; j < (int)timestamps.size(); j++)
			timestamps[j] += frame_time;
		
		// Get preprocessed frame
		vector<PointXYZ> sub_pts;
		vector<PointXYZ> normals;
		vector<float> norm_scores;
		vector<double> icp_scores;
		vector<size_t> sub_inds;
		Plane3D frame_ground;
		vector<float> heights;
		vector<clock_t> t;
		preprocess_frame(f_pts, timestamps, rings, sub_pts, normals, norm_scores, icp_scores, sub_inds, frame_ground, heights, params, t);
		
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
			for (int j = 0; j < (int)sub_inds.size(); j++)
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
					  Plane3D &frame_ground,
					  vector<float> &heights,
					  SLAM_params &params,
					  vector<clock_t> &t)
{
	//////////////////////////////////////////
	// Preprocess frame and compute normals //
	//////////////////////////////////////////

	// Parameters

	// Create a copy of points in polar coordinates
	vector<PointXYZ> polar_pts(f_pts);
	cart2pol_(polar_pts);
	
	t.push_back(std::clock());

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

	// Apply horizontal scaling (to have smaller neighborhoods in horizontal direction)
	lidar_horizontal_scale(polar_pts, params.h_scale);

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
	
	t.push_back(std::clock());

	// Get subsampling of the frame in carthesian coordinates (New points are barycenters or not?)
	grid_subsampling_centers(f_pts, sub_pts, sub_inds, params.frame_voxel_size);
	//grid_subsampling_spheres(f_pts, sub_pts, params.frame_voxel_size);

	// Convert sub_pts to polar and rescale
	vector<PointXYZ> polar_queries0(sub_pts);
	cart2pol_(polar_queries0);
	vector<PointXYZ> polar_queries(polar_queries0);
	lidar_log_radius(polar_queries, polar_r, params.r_scale);
	lidar_horizontal_scale(polar_queries, params.h_scale);

	// Get sub_rings
	vector<int> sub_rings;
	sub_rings.reserve(sub_inds.size());
	for (int j = 0; j < (int)sub_inds.size(); j++)
		sub_rings.push_back(f_rings[sub_inds[j]]);
	
	
	t.push_back(std::clock());


	/////////////////////
	// Compute normals //
	/////////////////////
	
	// Call polar processing function
	extract_lidar_frame_normals(f_pts, polar_pts, sub_pts, polar_queries, sub_rings, normals, norm_scores, params.polar_r2s);
	
	
	
	t.push_back(std::clock());

	/////////////////////////
	// Get ground in frame //
	/////////////////////////
	
	// Get a first estimate of the heights given previous orientation
	vector<PointXYZ> prealigned(sub_pts);
	Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> pts_mat((float *)prealigned.data(), 3, prealigned.size());
	Eigen::Matrix3f R_tot = (params.icp_params.last_transform1.block(0, 0, 3, 3)).cast<float>();
	Eigen::Vector3f T_tot = (params.icp_params.last_transform1.block(0, 3, 3, 1)).cast<float>();
	pts_mat = (R_tot * pts_mat).colwise() + T_tot;

	// Ransac ground extraction
	float vertical_thresh_deg = 20.0;
	float max_dist = 0.1;
	float ground_z = 0.0;
	frame_ground = frame_ground_ransac(sub_pts, normals, vertical_thresh_deg, max_dist, ground_z);

	// Ensure ground normal is pointing upwards
	if (frame_ground.u.z < 0)
		frame_ground.reverse();

	// Get height above ground
	frame_ground.point_distances_signed(sub_pts, heights);

	t.push_back(std::clock());

	////////////////
	// Get scores //
	////////////////

	// Better normal score vased on distance and incidence angle
	smart_normal_score(sub_pts, polar_queries0, normals, norm_scores);

	// ICP score between 1.0 and 6.0 (chance of being sampled during ICP)
	icp_scores = vector<double>(norm_scores.begin(), norm_scores.end());
	smart_icp_score(polar_queries0, normals, heights, icp_scores);

	// Remove points with a negative score
	float min_score = 0.0001;
	filter_pointcloud(sub_pts, norm_scores, min_score);
	filter_pointcloud(normals, norm_scores, min_score);
	filter_anyvector(sub_inds, norm_scores, min_score);
	filter_anyvector(icp_scores, norm_scores, min_score);
	filter_anyvector(heights, norm_scores, min_score);
	filter_floatvector(norm_scores, min_score);
	
}


void PointMapSLAM::add_new_frame(vector<PointXYZ> &f_pts,
								 vector<float> &f_ts,
								 vector<int> &f_rings,
								 Eigen::Matrix4d &H_OdomToScanner,
								 string save_path,
								 int verbose)
{

	//////////////////////
	// Timing variables //
	//////////////////////

	vector<string> clock_str;
	vector<clock_t> t;
	clock_str.reserve(20);
	clock_str.push_back("Frame to polar .... ");
	clock_str.push_back("Scale polar ....... ");
	clock_str.push_back("Subsampleing  ..... ");
	clock_str.push_back("Normals ........... ");
	clock_str.push_back("Ground ............ ");
	clock_str.push_back("Filter ............ ");
	clock_str.push_back("ICP localization .. ");
	clock_str.push_back("Alignment ......... ");
	clock_str.push_back("Map update ........ ");
	t.reserve(20);
	t.push_back(std::clock());


	//////////////////////
	// Initialize poses //
	//////////////////////
	
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


	//////////////////////////////////////////
	// Preprocess frame and compute normals //
	//////////////////////////////////////////

	vector<PointXYZ> sub_pts;
	vector<PointXYZ> normals;
	vector<float> norm_scores;
	vector<double> icp_scores;
	vector<size_t> sub_inds;
	Plane3D frame_ground;
	vector<float> heights;
	preprocess_frame(f_pts, f_ts, f_rings, sub_pts, normals, norm_scores, icp_scores, sub_inds, frame_ground, heights, params, t);

	// Check icp scores in case too many outliers
	int count_inliers = 0;
	for(auto s: icp_scores)
	{
		if (s > 0.05)
			count_inliers++;
	}

	if (count_inliers < params.icp_params.n_samples)
	{
		cout << "ERROR: at frame " << frame_i << ", Not enough inliers for ICP" << endl;
		string path000 = "/home/hth/Deep-Collison-Checker/SOGM-3D-2D-Net/results/";
		char buffer00[100];
		char buffer02[100];
		sprintf(buffer00, "no_inliers_%05d_map.ply", int(frame_i));
		sprintf(buffer02, "no_inliers_%05d_init.ply", int(frame_i));
		string filepath00 = path000 + string(buffer00);
		string filepath02 = path000 + string(buffer02);
		save_cloud(filepath00, map.cloud.pts, map.normals, map.scores);
		vector<float> f12(icp_scores.begin(), icp_scores.end());
		f12.insert(f12.end(), norm_scores.begin(),  norm_scores.end());
		save_cloud(filepath02, sub_pts, normals, f12);
	}

	// Min and max times (dont loop on the whole frame as it is useless)
	float loop_ratio = 0.01;
	get_min_max_times(f_ts, t_min, t_max, loop_ratio);
	
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
		for (int j = 0; j < (int)sub_inds.size(); j++)
			sub_alphas.push_back((f_ts[sub_inds[j]] - params.icp_params.last_time) * inv_factor);
	}

	t.push_back(std::clock());

	/////////////////////////////////
	// Align frame on map with ICP //
	/////////////////////////////////

	// Create result containers
	ICP_results icp_results;
	bool warning = false;

	if (params.icp_params.max_iter < 1)
	{
		// Case where we do not redo ICP but just align frames according to the given odom
		icp_results.transform = H_scannerToMap_init;
	}
	else
	{
		if (map.size() < 1 && map0.size() < 1)
		{ // Case where we do not have a map yet. Override the first cloud position so that ground is at z=0
			
			// Get transformation that make the ground plane horizontal
			Eigen::Matrix3d ground_R;
			if (!rot_u_to_v(frame_ground.u, PointXYZ(0, 0, 1), ground_R))
				ground_R = Eigen::Matrix3d::Identity();
			
			// Get a point that belongs to the ground
			float min_dist = 1e9;
			PointXYZ ground_P;
			for (int j = 0; j < (int)heights.size(); j++)
			{
				float dist = abs(heights[j]);
				if (dist < min_dist)
				{
					min_dist = dist;
					ground_P = sub_pts[j];
				}
			}

			// Rotate point and get new ground height
			Eigen::Map<Eigen::Matrix<float, 3, 1>> ground_P_mat((float*)&ground_P, 3, 1);
			ground_P_mat = ground_R.cast<float>() * ground_P_mat;

			// Update icp parameters to trigger flat ground
			if (params.force_flat_ground)
			{
				params.icp_params.ground_w = 9.0;
				params.icp_params.ground_z = 0.0;
			}

			// Update result transform
			icp_results.transform = Eigen::Matrix4d::Identity();
			icp_results.transform.block(0, 0, 3, 3) = ground_R;
			icp_results.transform(2, 3) = - ground_P_mat(2);
			params.icp_params.last_transform0 = icp_results.transform;
		}
		else
		{
			if (frame_i < 1)
			{
				// Case where we have a map, and the first frame needs to be aligned
				// We assume robot is still in the beginning so no motion distortion

				// 1. Initial RANSAC alignment

				// Forcing ground at z = 0
				if (params.force_flat_ground)
				{
					params.icp_params.ground_w = 9.0;
					params.icp_params.ground_z = 0.0;
				}

				// 2. ICP refine. For robust init, we try different orientations
				params.icp_params.motion_distortion = false;
				float best_rms = 1e9;
				Eigen::Matrix4d best_init = Eigen::Matrix4d::Identity();
				for (float init_angle = -20.0; init_angle<21.0; init_angle+=5.0)
				{
					// Rotate initial transform in place
					float init_radians = init_angle * M_PI / 180.0;
					Eigen::Matrix3d R_init;
					if (!rot_u_to_v(PointXYZ(1, 0, 0), PointXYZ(cos(init_radians), sin(init_radians), 0), R_init))
						R_init = Eigen::Matrix3d::Identity();

							
					params.icp_params.init_transform = H_scannerToMap_init;
					params.icp_params.init_transform.block(0, 0, 3, 3) = R_init * params.icp_params.init_transform.block(0, 0, 3, 3);

					ICP_results tmp_results;
					if (map0.size() > 0)
						PointToMapICP(sub_pts, sub_alphas, icp_scores, map0, params.icp_params, tmp_results);
					else
						PointToMapICP(sub_pts, sub_alphas, icp_scores, map, params.icp_params, tmp_results);

					// Measure quality of the icp alignement
					if (tmp_results.all_rms[tmp_results.all_rms.size() - 1] < best_rms)
					{
						best_rms = tmp_results.all_rms[tmp_results.all_rms.size() - 1];
						icp_results = tmp_results;
						best_init = params.icp_params.init_transform;
					}

				}
				
				params.icp_params.init_transform = best_init;
				params.icp_params.motion_distortion = params.motion_distortion;

				// We override last_transform0 too to neglate motion distortion for this first frame
				params.icp_params.last_transform0 = icp_results.transform;
			}
			else
			{
				params.icp_params.init_transform = H_scannerToMap_init;
				if (map0.size() > 0)
					PointToMapICP(sub_pts, sub_alphas, icp_scores, map0, params.icp_params, icp_results);
				else
					PointToMapICP(sub_pts, sub_alphas, icp_scores, map, params.icp_params, icp_results);
			}

			// Safe Check
			if (icp_results.all_plane_rms.size() > 0.4 * params.icp_params.max_iter)
			{
				if (icp_results.all_plane_rms.size() > 0.9 * params.icp_params.max_iter)
				{
					warning = true;
					warning_count += 1;
					cout << "ERROR: at frame " << frame_i << ", ICP not converging, num_iter = " << icp_results.all_plane_rms.size() << endl;

					// Debug (Points with scores)
					string path000 = "/home/hth/Deep-Collison-Checker/SOGM-3D-2D-Net/results/";
					char buffer00[100];
					char buffer01[100];
					char buffer02[100];
					sprintf(buffer00, "f_%05d_%03d-iter_map.ply", int(frame_i), (int)icp_results.all_plane_rms.size());
					sprintf(buffer01, "f_%05d_%03d-iter_init.ply", int(frame_i), (int)icp_results.all_plane_rms.size());
					sprintf(buffer02, "f_%05d_%03d-iter_map0.ply", int(frame_i), icp_results.all_plane_rms.size());
					string filepath00 = path000 + string(buffer00);
					string filepath01 = path000 + string(buffer01);
					string filepath02 = path000 + string(buffer02);

					vector<float> oldest(map.oldest.begin(), map.oldest.end());
					oldest.insert(oldest.end(), map.scores.begin(),  map.scores.end());
					oldest.insert(oldest.end(), map.latest.begin(),  map.latest.end());
					save_cloud(filepath00, map.cloud.pts, map.normals, oldest);

					if (map0.size() > 0)
					{
						vector<float> oldest0(map0.oldest.begin(), map0.oldest.end());
						oldest0.insert(oldest0.end(), map0.scores.begin(),  map0.scores.end());
						oldest0.insert(oldest0.end(), map0.latest.begin(),  map0.latest.end());
						save_cloud(filepath02, map0.cloud.pts, map0.normals, oldest0);
					}

					vector<PointXYZ> copy_pts(sub_pts);
					if (params.motion_distortion)
					{
						// Update map taking motion distortion into account
						size_t i_inds = 0;
						Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> pts_mat((float *)copy_pts.data(), 3, copy_pts.size());
						for (auto& alpha : sub_alphas)
						{
							Eigen::Matrix4d H_rect = pose_interp(alpha, params.icp_params.last_transform0, params.icp_params.init_transform, 0);
							Eigen::Matrix3f R_rect = (H_rect.block(0, 0, 3, 3)).cast<float>();
							Eigen::Vector3f T_rect = (H_rect.block(0, 3, 3, 1)).cast<float>();
							pts_mat.col(i_inds) = (R_rect * pts_mat.col(i_inds)) + T_rect;
							i_inds++;
						}
					}
					else
					{
						Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> pts_mat((float *)copy_pts.data(), 3, copy_pts.size());
						Eigen::Matrix3f R_tot = (params.icp_params.init_transform.block(0, 0, 3, 3)).cast<float>();
						Eigen::Vector3f T_tot = (params.icp_params.init_transform.block(0, 3, 3, 1)).cast<float>();
						pts_mat = (R_tot * pts_mat).colwise() + T_tot;
					}

					vector<float> f12(icp_scores.begin(), icp_scores.end());
					f12.insert(f12.end(), norm_scores.begin(),  norm_scores.end());
					save_cloud(filepath01, copy_pts, f12);

				}
				else
					cout << "WARNING: at frame " << frame_i << ", ICP num_iter = " << icp_results.all_plane_rms.size() << endl;

				
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
		sprintf(buffer02, "f_%05d_%03d-iter_last.ply", int(frame_i), (int)icp_results.all_plane_rms.size());
		string filepath02 = path000 + string(buffer02);
		vector<float> f12(icp_scores.begin(), icp_scores.end());
		f12.insert(f12.end(), norm_scores.begin(),  norm_scores.end());
		save_cloud(filepath02, sub_pts, normals, f12);
	}
	else
	{
		warning_count = 0;
	}

	// ----------------------------------------------------------------
	// Saving all frames for loop closure
	if (params.saving_for_loop_closure)
	{
		if (!std::filesystem::exists(save_path + string("/icp_frames")))
			std::filesystem::create_directories(save_path + string("/icp_frames"));
		char buffer02[100];
		sprintf(buffer02, "f_%05d.ply", int(frame_i));
		string path_f = save_path + string("/icp_frames/") +  string(buffer02);
		if (!std::filesystem::exists(path_f))
		{
			vector<float> f12(icp_scores.begin(), icp_scores.end());
			f12.insert(f12.end(), norm_scores.begin(),  norm_scores.end());
			save_cloud(path_f, sub_pts, normals, f12);
		}
	}
	// ----------------------------------------------------------------

	t.push_back(std::clock());
	
	// The update function is called only on subsampled points as the others have no normal
	if (map0.size() > 0 && params.update_init_map)
		map.update_double(sub_pts, normals, norm_scores, frame_i, map0);
	else
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

	
	// Save timings for a report
	bool saving_timings = true;
	if (saving_timings)
	{
		string t_path = "/home/hth/Deep-Collison-Checker/SOGM-3D-2D-Net/results/all_timings.txt";
		std::fstream outfile;
		if (frame_i < 2)
		{
			outfile.open(t_path, std::fstream::out);
			for (size_t i = 0; i < min(t.size() - 1, clock_str.size()); i++)
				outfile << clock_str[i] << "\n";
		}
		else
			outfile.open(t_path, std::fstream::app);
		outfile << frame_i;
		for (size_t i = 0; i < min(t.size() - 1, clock_str.size()); i++)
			outfile << " " << 1000 * (t[i + 1] - t[i]) / (double)CLOCKS_PER_SEC;
		outfile << "\n";
		outfile.close();
	}

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
		mapper.add_new_frame(f_pts, timestamps, f_rings, H_OdomToScanner, string(""));

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
			sprintf(buffer, "Frame %d named %s", (int)frame_i, line.c_str());
			cout << string(buffer) << endl;
			cout <<"*************************************************" << endl << endl;
			save_cloud("debug_map.ply", mapper.map.cloud.pts, mapper.map.normals, mapper.map.scores);
			break;
		}

		// if (frame_i % 100 == 0)
		// {
		// 	char buffer[100];
		// 	sprintf(buffer, "cc_map_%05d.ply", (int)frame_i);
		// 	vector<float> oldest(mapper.map.oldest.begin(), mapper.map.oldest.end());
		// 	oldest.insert(oldest.end(), mapper.map.scores.begin(),  mapper.map.scores.end());
		// 	save_cloud(string(buffer), mapper.map.cloud.pts, mapper.map.normals, oldest);
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
			sprintf(buffer, "Mapping %5d/%d at %5.1f fps - %d min %.0f sec remaining", (int)frame_i, (int)frame_times.size(), fps, remaining_min, remaining_sec);
			cout << string(buffer) << endl;
			last_disp_t1 = t1;
		}

		frame_i++;

	}


	// Save map in a ply file init containers with results
	size_t ns1 = 19;
	size_t ns0 = save_path.size() - ns1;
	string day_str = save_path.substr(ns0, ns1);
	vector<float> oldest(mapper.map.oldest.begin(), mapper.map.oldest.end());
	oldest.insert(oldest.end(), mapper.map.scores.begin(),  mapper.map.scores.end());
	save_cloud(save_path + "/map_" + day_str + ".ply", mapper.map.cloud.pts, mapper.map.normals, oldest);

	return all_H;
}



// --------
// ---------------------------
// ----------------------------------------------------------------------------------------------------------

// clang-format on
struct ExponentialResidual {
  ExponentialResidual(double x, double y) : x_(x), y_(y) {}
  template <typename T>
  bool operator()(const T* const m, const T* const c, T* residual) const {
    residual[0] = y_ - exp(m[0] * x_ + c[0]);
    return true;
  }
 private:
  const double x_;
  const double y_;
};

void ceres_hello()
{
	std::string str = "TESTHOHOHO";
	google::InitGoogleLogging(str.c_str());
	int kNumObservations = 67;

	double data[] = {0.000000e+00, 1.133898e+00,
					 7.500000e-02, 1.334902e+00,
					 1.500000e-01, 1.213546e+00,
					 2.250000e-01, 1.252016e+00,
					 3.000000e-01, 1.392265e+00,
					 3.750000e-01, 1.314458e+00,
					 4.500000e-01, 1.472541e+00,
					 5.250000e-01, 1.536218e+00,
					 6.000000e-01, 1.355679e+00,
					 6.750000e-01, 1.463566e+00,
					 7.500000e-01, 1.490201e+00,
					 8.250000e-01, 1.658699e+00,
					 9.000000e-01, 1.067574e+00,
					 9.750000e-01, 1.464629e+00,
					 1.050000e+00, 1.402653e+00,
					 1.125000e+00, 1.713141e+00,
					 1.200000e+00, 1.527021e+00,
					 1.275000e+00, 1.702632e+00,
					 1.350000e+00, 1.423899e+00,
					 1.425000e+00, 1.543078e+00,
					 1.500000e+00, 1.664015e+00,
					 1.575000e+00, 1.732484e+00,
					 1.650000e+00, 1.543296e+00,
					 1.725000e+00, 1.959523e+00,
					 1.800000e+00, 1.685132e+00,
					 1.875000e+00, 1.951791e+00,
					 1.950000e+00, 2.095346e+00,
					 2.025000e+00, 2.361460e+00,
					 2.100000e+00, 2.169119e+00,
					 2.175000e+00, 2.061745e+00,
					 2.250000e+00, 2.178641e+00,
					 2.325000e+00, 2.104346e+00,
					 2.400000e+00, 2.584470e+00,
					 2.475000e+00, 1.914158e+00,
					 2.550000e+00, 2.368375e+00,
					 2.625000e+00, 2.686125e+00,
					 2.700000e+00, 2.712395e+00,
					 2.775000e+00, 2.499511e+00,
					 2.850000e+00, 2.558897e+00,
					 2.925000e+00, 2.309154e+00,
					 3.000000e+00, 2.869503e+00,
					 3.075000e+00, 3.116645e+00,
					 3.150000e+00, 3.094907e+00,
					 3.225000e+00, 2.471759e+00,
					 3.300000e+00, 3.017131e+00,
					 3.375000e+00, 3.232381e+00,
					 3.450000e+00, 2.944596e+00,
					 3.525000e+00, 3.385343e+00,
					 3.600000e+00, 3.199826e+00,
					 3.675000e+00, 3.423039e+00,
					 3.750000e+00, 3.621552e+00,
					 3.825000e+00, 3.559255e+00,
					 3.900000e+00, 3.530713e+00,
					 3.975000e+00, 3.561766e+00,
					 4.050000e+00, 3.544574e+00,
					 4.125000e+00, 3.867945e+00,
					 4.200000e+00, 4.049776e+00,
					 4.275000e+00, 3.885601e+00,
					 4.350000e+00, 4.110505e+00,
					 4.425000e+00, 4.345320e+00,
					 4.500000e+00, 4.161241e+00,
					 4.575000e+00, 4.363407e+00,
					 4.650000e+00, 4.161576e+00,
					 4.725000e+00, 4.619728e+00,
					 4.800000e+00, 4.737410e+00,
					 4.875000e+00, 4.727863e+00,
					 4.950000e+00, 4.669206e+00};

	double m = 0.0;
	double c = 0.0;
	ceres::Problem problem;
	for (int i = 0; i < kNumObservations; ++i)
	{
		problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ExponentialResidual, 1, 1, 1>(new ExponentialResidual(data[2 * i], data[2 * i + 1])),
								 NULL,
								 &m,
								 &c);
	}
	ceres::Solver::Options options;
	options.max_num_iterations = 25;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << "\n";
	std::cout << "Initial m: " << 0.0 << " c: " << 0.0 << "\n";
	std::cout << "Final   m: " << m << " c: " << c << "\n";
	return;
}

// ----------------------------------------------------------------------------------------------------------
// ---------------------------
// --------

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

	bool try_ceres = false;
	if (try_ceres)
	{

		// TODO: HERE use ceres for bundle_pt2pl_icp
		// 			inspired from http://ceres-solver.org/nnls_tutorial.html#bundle-adjustment
		// 			also https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/slam/pose_graph_3d/pose_graph_3d_error_term.h

		// CostFunctor :
		//	1 - align given point/normal with given pose, 
		//	2 - align measurement point/normal with measurement pose
		//	3 - Compute point to plane error as residual
	

		ceres_hello();
		Eigen::MatrixXd all_H = Eigen::MatrixXd::Zero(4 * frame_times.size(), 4);
		return all_H;

	}

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
	int last_min_frame_i = -2;

	// Timing
	float fps = 0.0;
	float fps_regu = 0.9;

	// Initial timestamp for motion distorsiob
	double timestamp_0 = frame_times[0];

	// Some safe checks
	if (slam_params.barycenter_map && mapper.map0.size() < 0)
		throw std::invalid_argument(string("\nERROR: cannot create a barycenter map if no initial map is given\n"));

	// if (slam_params.barycenter_map && slam_params.update_init_map)
	// 	throw std::invalid_argument(string("\nERROR: cannot create a barycenter map AND update the initial map\n"));

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

		clock_t t0 = clock();

		// Get odometry matrix (pose of the scanner in odometry world frame)
		Eigen::Matrix4d H_OdomToScanner = odom_H.block(frame_ind * 4, 0, 4, 4);


		// Map this frame
		// **************


		// Get timestamps
		float frame_time = (float)(frame_times[frame_ind] - timestamp_0);
		for (int j = 0; j < (int)timestamps.size(); j++)
			timestamps[j] += frame_time;

		// Get frame pose and update map
		if (slam_params.verbose_time > 0 && slam_params.verbose_time < 0.001)
			mapper.add_new_frame(f_pts, timestamps, rings, H_OdomToScanner, save_path, 1);
		else
			mapper.add_new_frame(f_pts, timestamps, rings, H_OdomToScanner, save_path, 0);

		// Save transform
		all_H.block(frame_ind * 4, 0, 4, 4) = mapper.params.icp_params.last_transform1;
		all_times.push_back(mapper.t_max);

		// Save position for loop closure
		float closure_d = 2.0;
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

		// We remove old points from the map neighbor tree to make sure there is no jump
		// in the lidar odom when comming in a loop closure
		if (mapper.params.icp_params.max_iter > 0 &&
			mapper.map0.size() < 1 &&
			mapper.params.local_map_dist > save_d)
		{
			// vector<clock_t> t;
			// t.reserve(2);
			// t.push_back(std::clock());

			// Get the new frame index limit based on travelled distance
			int min_frame_i = -3;

			// Convert dist threshold to number of sparse poses
			int n_sparse = (int)floor(mapper.params.local_map_dist / save_d) + 1;
			if ((int)sparse_f_inds.size() > n_sparse)
				min_frame_i = sparse_f_inds[sparse_f_inds.size() - n_sparse];

			// Remove indics from tree
			int removed_count = 0;
			if (min_frame_i > last_min_frame_i)
			{
				removed_count = mapper.map.remove_old(min_frame_i, last_min_frame_i);
				last_min_frame_i = min_frame_i;
			}
			
			// if (removed_count > 0)
			// {
			// 	t.push_back(std::clock());
			// 	double duration = 1000 * (t[1] - t[0]) / (double)CLOCKS_PER_SEC;
			// 	cout << "  >>> Removed " << removed_count << " inds in " << duration << " ms" << endl;
			// }
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
		closure_ind = -1;
		if (closure_ind >= 0)
		{
			cout << "\n  >>> Loop detected. Performing closure" << endl;

			string path000 = "/home/hth/Deep-Collison-Checker/SOGM-3D-2D-Net/results/";
			char buffer00[100];
			sprintf(buffer00, "f_%05d_map_before.ply", int(frame_ind));
			vector<float> all_features(mapper.map.oldest.begin(), mapper.map.oldest.end());
			all_features.insert(all_features.end(), mapper.map.scores.begin(),  mapper.map.scores.end());
			save_cloud(path000 + string(buffer00), mapper.map.cloud.pts, mapper.map.normals, all_features);

			// 1. Recreate a new cleam map from current one until closure_ind
			PointMap clean_map(mapper.map, closure_ind);
			cout << "\n  >>> Map cleaned" << endl;
			
			char buffer01[100];
			sprintf(buffer01, "f_%05d_map_clean.ply", int(frame_ind));
			vector<float> all_features1(clean_map.oldest.begin(), clean_map.oldest.end());
			all_features1.insert(all_features1.end(), clean_map.scores.begin(),  clean_map.scores.end());
			save_cloud(path000 + string(buffer01), clean_map.cloud.pts, clean_map.normals, all_features1);

			// 2. Perform ICP with this frame on this clean map
			ICP_results icp_results;
			mapper.params.icp_params.init_transform = Eigen::Matrix4d::Identity(4, 4);
			vector<float> alphas;

			//
			// TODO: initial transform for ICP 
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
			vector<float> all_features2(mapper.map.oldest.begin(), mapper.map.oldest.end());
			all_features2.insert(all_features2.end(), mapper.map.scores.begin(),  mapper.map.scores.end());
			save_cloud(path000 + string(buffer02), mapper.map.cloud.pts, mapper.map.normals, all_features2);
		}

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
			sprintf(buffer, "Mapping %5d/%d at %5.1f fps - %d min %.0f sec remaining", (int)frame_ind, (int)frame_times.size(), fps, remaining_min, remaining_sec);
			cout << string(buffer) << endl;
			last_disp_t1 = t1;
		}

		frame_ind++;

		if (mapper.warning_count > 3)
			break;

		// if (frame_ind > 500)
		// 	break;
	}

	// Saving map depending on the use of barycenters
	
	// Eliminate barycenter not in their cell
	// **************************************

	size_t ns1 = 19;
	size_t ns0 = save_path.size() - ns1;
	string day_str = save_path.substr(ns0, ns1);
	VoxKey k0;
	vector<PointXYZ> clean_points;
	vector<PointXYZ> clean_normals;
	vector<float> clean_oldest;
	vector<float> clean_scores;
	vector<float> clean_counts;
	clean_points.reserve(mapper.map.size());
	clean_normals.reserve(mapper.map.size());
	clean_oldest.reserve(mapper.map.size());
	clean_scores.reserve(mapper.map.size());
	clean_counts.reserve(mapper.map.size());


	if (slam_params.barycenter_map)
	{
		for (auto& v : mapper.map.samples)
		{
			size_t i = v.second;

			// Position of barycenter in sample map
			PointXYZ centroid = mapper.map.cloud.pts[i];
			PointXYZ p_pos = centroid * mapper.map.inv_dl;

			// Corresponding key
			k0.x = (int)floor(p_pos.x);
			k0.y = (int)floor(p_pos.y);
			k0.z = (int)floor(p_pos.z);
			
			// if (k0 == v.first)
			if (mapper.map.valid[i])
			{
				clean_points.push_back(centroid);
				clean_normals.push_back(mapper.map.normals[i]);
				clean_oldest.push_back((float)mapper.map.oldest[i]);
				clean_scores.push_back(mapper.map.scores[i]);
				clean_counts.push_back((float)mapper.map.latest[i]);
			}
			
		}

		// Save map in a ply file init containers with results
		clean_oldest.insert(clean_oldest.end(), clean_scores.begin(),  clean_scores.end());
		clean_oldest.insert(clean_oldest.end(), clean_counts.begin(),  clean_counts.end());
		save_cloud(save_path + "/barymap_" + day_str + ".ply", clean_points, clean_normals, clean_oldest);

	}
	else
	{
		// Save map in a ply file init containers with results
		vector<float> oldest(mapper.map.oldest.begin(), mapper.map.oldest.end());
		oldest.insert(oldest.end(), mapper.map.scores.begin(),  mapper.map.scores.end());
		oldest.insert(oldest.end(), mapper.map.latest.begin(),  mapper.map.latest.end());
		save_cloud(save_path + "/map_" + day_str + ".ply", mapper.map.cloud.pts, mapper.map.normals, oldest);
	}
	




	return all_H;
}





