
#include "pointmap_slam.h"


// SLAM functions
// **************


void PointMapSLAM::add_new_frame(vector<PointXYZ>& f_pts, Eigen::Matrix4d& H_OdomToScanner, int verbose)
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

	// Parameters

	// Create a copy of points in polar coordinates
	vector<PointXYZ> polar_pts(f_pts);
	cart2pol_(polar_pts);

	t.push_back(std::clock());

	// Get lidar angle resolution
	float minTheta, maxTheta;
	float lidar_angle_res = get_lidar_angle_res(polar_pts, minTheta, maxTheta, params.lidar_n_lines);

	t.push_back(std::clock());

	// Define the polar neighbors radius in the scaled polar coordinates
	float polar_r = 1.5 * lidar_angle_res;

	// Apply log scale to radius coordinate (in place)
	lidar_log_radius(polar_pts, polar_r, params.r_scale);

	// Remove outliers (only for real frames)
	if (params.motion_distortion)
	{
		// Get an outlier score
		vector<float> scores(polar_pts.size(), 0.0);
		detect_outliers(polar_pts, scores, params.lidar_n_lines, lidar_angle_res, minTheta, params.outl_rjct_passes, params.outl_rjct_thresh);

		// Remove points with negative score
		filter_pointcloud(f_pts, scores, 0);
		filter_pointcloud(polar_pts, scores, 0);
	}

	// Apply horizontal scaling (to have smaller neighborhoods in horizontal direction)
	lidar_horizontal_scale(polar_pts, params.h_scale);

	t.push_back(std::clock());

	// Get subsampling of the frame in carthesian coordinates (New points are barycenters or not?)
	vector<PointXYZ> sub_pts;
	//vector<size_t> sub_inds;
	//grid_subsampling_centers(f_pts, sub_pts, sub_inds, params.frame_voxel_size);
	grid_subsampling_spheres(f_pts, sub_pts, params.frame_voxel_size);

	t.push_back(std::clock());

	// Convert sub_pts to polar and rescale
	vector<PointXYZ> polar_queries0(sub_pts);
	cart2pol_(polar_queries0);
	vector<PointXYZ> polar_queries(polar_queries0);
	lidar_log_radius(polar_queries, polar_r, params.r_scale);
	lidar_horizontal_scale(polar_queries, params.h_scale);

	t.push_back(std::clock());

	/////////////////////
	// Compute normals //
	/////////////////////

	// Init result containers
	vector<PointXYZ> normals;
	vector<float> norm_scores;

	// Call polar processing function
	extract_lidar_frame_normals(f_pts, polar_pts, sub_pts, polar_queries, normals, norm_scores, polar_r);

	// Better normal score vased on distance and incidence angle
	vector<float> icp_scores(norm_scores);
	smart_icp_score(polar_queries0, icp_scores);
	smart_normal_score(sub_pts, polar_queries0, normals, norm_scores);

	// Remove points with a low score
	float min_score = 0.01;
	filter_pointcloud(sub_pts, norm_scores, min_score);
	filter_pointcloud(normals, norm_scores, min_score);
	filter_floatvector(icp_scores, norm_scores, min_score);
	filter_floatvector(norm_scores, min_score);

	t.push_back(std::clock());

	/////////////////////////////////
	// Align frame on map with ICP //
	/////////////////////////////////

	// Create result containers
	ICP_results icp_results;

	// If no odometry is given, use identity
	if (H_OdomToScanner.lpNorm<1>() < 0.001)
		H_OdomToScanner = Eigen::Matrix4d::Identity(4, 4);

	// Use odometry as init
	Eigen::Matrix4d H_scannerToMap_init = H_OdomToMap * H_OdomToScanner.inverse();
	
	// If no map is available, use init_H as first pose
	if (map.size() < 1 || params.icp_params.max_iter < 1)
	{
		icp_results.transform = H_scannerToMap_init;
	}
	else
	{
		// Perform ICP
		params.icp_params.init_transform = H_scannerToMap_init;
		PointToMapICP(sub_pts, icp_scores, map, params.icp_params, icp_results);
			

		// Safe Check
		if (icp_results.all_plane_rms.size() > 3 * params.icp_params.avg_steps)
			cout << "WARNING: ICP num_iter = " << icp_results.all_plane_rms.size() << endl;

		if (icp_results.all_plane_rms.size() > 0.9 * params.icp_params.max_iter)
			cout << "ERROR: ICP not converging = " << icp_results.all_plane_rms.size() << endl;
	}

	t.push_back(std::clock());


	////////////////////
	// Update the map //
	////////////////////

	if (params.motion_distortion)
	{
		// TODO			
		Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> pts_mat((float*)sub_pts.data(), 3, sub_pts.size());
		Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> norms_mat((float*)normals.data(), 3, normals.size());
		Eigen::Matrix3f R_tot = (icp_results.transform.block(0, 0, 3, 3)).cast<float>();
		Eigen::Vector3f T_tot = (icp_results.transform.block(0, 3, 3, 1)).cast<float>();
		pts_mat = (R_tot * pts_mat).colwise() + T_tot;
		norms_mat = R_tot * norms_mat;

		// TODO Here:	- Handle case of motion distortion
		//				- optimize by using the phis computed in ICP
	}
	else
	{
		Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> pts_mat((float*)sub_pts.data(), 3, sub_pts.size());
		Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> norms_mat((float*)normals.data(), 3, normals.size());
		Eigen::Matrix3f R_tot = (icp_results.transform.block(0, 0, 3, 3)).cast<float>();
		Eigen::Vector3f T_tot = (icp_results.transform.block(0, 3, 3, 1)).cast<float>();
		pts_mat = (R_tot * pts_mat).colwise() + T_tot;
		norms_mat = R_tot * norms_mat;

	}

	// // Check rotation modif
	// Eigen::Matrix3d R2 = icp_results.transform.block(0, 0, 3, 3);
	// Eigen::Matrix3d R1 = last_H.block(0, 0, 3, 3);
	// R1 = R2 * R1.transpose();
	// float R_error = acos((R1.trace() - 1) / 2);

	// if (R_error > 10.0 * M_PI / 180)
	// {
	// 	if (map.update_idx > 2)
	// 	{
	// 		char buffer[100];
	// 		sprintf(buffer, "cc_aligned_%05d.ply", (int)map.update_idx);
	// 		save_cloud(string(buffer), sub_pts, icp_scores);
	// 	}

	// 	if (map.update_idx > 2)
	// 	{

	// 		vector<PointXYZ> init_pts(f_pts);
	// 		Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> pts_mat((float*)init_pts.data(), 3, init_pts.size());
	// 		Eigen::Matrix3f R_tot = (last_H.block(0, 0, 3, 3)).cast<float>();
	// 		Eigen::Vector3f T_tot = (last_H.block(0, 3, 3, 1)).cast<float>();
	// 		pts_mat = (R_tot * pts_mat).colwise() + T_tot;

	// 		char buffer[100];
	// 		sprintf(buffer, "cc_last_%05d.ply", (int)map.update_idx);
	// 		save_cloud(string(buffer), init_pts);
	// 	}

	// 	if (map.update_idx > 2)
	// 	{

	// 		vector<PointXYZ> init_pts(f_pts);
	// 		Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> pts_mat((float*)init_pts.data(), 3, init_pts.size());
	// 		Eigen::Matrix3f R_tot = (H_scannerToMap_init.block(0, 0, 3, 3)).cast<float>();
	// 		Eigen::Vector3f T_tot = (H_scannerToMap_init.block(0, 3, 3, 1)).cast<float>();
	// 		pts_mat = (R_tot * pts_mat).colwise() + T_tot;

	// 		char buffer[100];
	// 		sprintf(buffer, "cc_init_%05d.ply", (int)map.update_idx);
	// 		save_cloud(string(buffer), init_pts);
	// 	}
	// }

	// The update function is called only on subsampled points as the others have no normal
	map.update(sub_pts, normals, norm_scores);

	// Update the pose correction from map to odom
	H_OdomToMap = icp_results.transform * H_OdomToScanner;

	// Update the last pose for future frames
	last_H = icp_results.transform;

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
		mapper.add_new_frame(f_pts, H_OdomToScanner, 0);

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





















