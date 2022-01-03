
#include "polar_processing.h"

void cart2pol_(vector<PointXYZ> &xyz)
{
	// In place modification to carthesian coordinates
	float tmp1, tmp2, phi;
	float pi_s_2 = M_PI / 2;
	for (auto& p : xyz)
	{
		tmp1 = p.x * p.x + p.y * p.y;
		tmp2 = tmp1 + p.z * p.z;
		phi = atan2(p.y, p.x); // azimuth angle
		p.y = atan2(p.z, sqrt(tmp1));
		p.z = phi + pi_s_2;
		p.x = sqrt(tmp2);
	}
}


void get_min_max_times(vector<float> &f_ts, float &t_min, float &t_max, float loop_ratio)
{
	t_min = f_ts[0];
	t_max = f_ts[0];
	for (int j = 0; (float)j < loop_ratio * (float)f_ts.size();  j++)
	{
		if (f_ts[j] < t_min)
			t_min = f_ts[j];
	}
	for (int j = (int)floor((1 - loop_ratio) * f_ts.size()); j < (int)f_ts.size();  j++)
	{
		if (f_ts[j] > t_max)
			t_max = f_ts[j];
	}
}


void pca_features(vector<PointXYZ> &points,
				  vector<float> &eigenvalues,
				  vector<PointXYZ> &eigenvectors)
{
	// Safe check
	if (points.size() < 4)
		return;

	// Compute PCA
	PointXYZ mean = accumulate(points.begin(), points.end(), PointXYZ());
	mean = mean * (1.0 / points.size());

	// Create centralized data
	for (auto &p : points)
		p -= mean;

	// Create a N by 3 matrix containing the points (same data in memory)
	Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> X_c((float *)points.data(), 3, points.size());

	// Compute covariance matrix
	Eigen::Matrix3f cov(X_c * X_c.transpose() / points.size());

	// Compute pca
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es;
	es.compute(cov);

	// Convert back to std containers
	eigenvalues = vector<float>(es.eigenvalues().data(), es.eigenvalues().data() + es.eigenvalues().size());
	eigenvectors = vector<PointXYZ>((PointXYZ *)es.eigenvectors().data(), (PointXYZ *)es.eigenvectors().data() + es.eigenvectors().rows());
}

void detect_outliers(vector<PointXYZ> &rtp,
					 vector<float> &scores,
					 int lidar_n_lines,
					 float lidar_angle_res,
					 float minTheta,
					 int n_pass,
					 float threshold)
{
	float theta0 = minTheta - 0.5 * lidar_angle_res;

	// Follow each scan line variables
	for (int pass = 0; pass < n_pass; pass++)
	{
		vector<float> last_r(lidar_n_lines, 0.0);
		vector<size_t> last_i(lidar_n_lines, 0);

		size_t i = 0;
		size_t l = 0;
		for (auto &p : rtp)
		{
			if (scores[i] > -0.5)
			{
				// Get line index
				l = (size_t)floor((p.y - theta0) / lidar_angle_res);

				// Get jumps
				float dl1 = p.x - last_r[l];

				// eliminate jumps
				if (abs(dl1) > threshold)
				{
					scores[last_i[l]] = -1.0;
					scores[i] = -1.0;
				}

				// Update saved variable
				last_r[l] = p.x;
				last_i[l] = i;
			}
			i++;
		}
	}
}

float get_lidar_angle_res(vector<PointXYZ> &rtp, float &minTheta, float &maxTheta, int lidar_n_lines)
{
	// Find lidar angle resolution automatically
	minTheta = 1000.0f;
	maxTheta = -1000.0f;
	for (size_t i = 1; i < rtp.size() - 1; i++)
	{
		if (rtp[i].y < minTheta)
			minTheta = rtp[i].y;
		if (rtp[i].y > maxTheta)
			maxTheta = rtp[i].y;
	}

	// Get line of scan inds
	return (maxTheta - minTheta) / (float)(lidar_n_lines - 1);
}

void get_lidar_angles(vector<PointXYZ>& rtp, vector<float>& ring_angles, int lidar_n_lines)
{
	// Find the angles for each lidar ring (only performed once at the start of the algorithm)

	float margin = 0.002;

	// Find lidar angle resolution automatically
	for (size_t i = 1; i < rtp.size(); i++)
	{
		bool new_ring = true;
		for (size_t l = 0; l < ring_angles.size(); l++)
		{
			if (abs(rtp[i].y - ring_angles[l]) < margin)
			{
				new_ring = false;
				break;
			}
		}
		if (new_ring)
		{
			ring_angles.push_back(rtp[i].y);
		}
	}
	
	stable_sort(ring_angles.begin(), ring_angles.end());

	if ((int)ring_angles.size() != lidar_n_lines)
	{
		cout << "WARNING: wrong number of lidar rings found: " << ring_angles.size() << " instead of " << lidar_n_lines << endl;
	}
	return;
}


void lidar_log_radius(vector<PointXYZ> &rtp, float polar_r, float r_scale)
{
	float r_factor = polar_r / (r_scale * log((1 + polar_r) / (1 - polar_r)));
	for (auto &p : rtp)
		p.x = log(p.x) * r_factor;
}

void lidar_horizontal_scale(vector<PointXYZ> &rtp, float h_scale)
{
	float h_factor = 1 / h_scale;
	for (auto &p : rtp)
		p.z *= h_factor;
}

void extract_features_multi_thread(vector<PointXYZ> &points,
								   vector<PointXYZ> &normals,
								   vector<float> &planarity,
								   vector<float> &linearity,
								   int lidar_n_lines,
								   float h_scale,
								   float r_scale,
								   int verbose)
{
	// Initialize variables
	// ********************

	// Number of points
	size_t N = points.size();

	// Result vectors
	normals = vector<PointXYZ>(points.size(), PointXYZ());
	planarity = vector<float>(points.size(), 0.0);
	linearity = vector<float>(points.size(), 0.0);

	// Cloud variable for KDTree
	PointCloud polar_cloud;
	polar_cloud.pts = vector<PointXYZ>(points);

	// Convert points to polar coordinates
	// ***********************************

	// In place modification of the data
	cart2pol_(polar_cloud.pts);

	// Find lidar angle resolution automatically
	float minTheta, maxTheta;
	float lidar_angle_res = get_lidar_angle_res(polar_cloud.pts, minTheta, maxTheta, lidar_n_lines);

	// Apply scaling
	// *************

	// Define search radius in chebichev metric. Vertical angular resolution of HDL32 is 1.29
	// lidar_angle_res = 1.29 * np.pi / 180;
	float polar_r = 1.5 * lidar_angle_res;

	// Apply horizontal and range scales
	// h_scale = 0.5 (smaller distance for the neighbor in horizontal direction)
	// r_scale = 4.0 (larger dist for neighbors in radius direction). Use log of range so that neighbor radius is proportional to the range
	float h_factor = 1 / h_scale;
	float r_factor = polar_r / (log((1 + polar_r) / (1 - polar_r)) * r_scale);
	for (auto &p : polar_cloud.pts)
	{
		p.z *= h_factor;
		p.x = log(p.x) * r_factor;
	}
	float r2 = polar_r * polar_r;

	// Outlier detection
	// *****************

	detect_outliers(polar_cloud.pts, planarity, lidar_n_lines, lidar_angle_res, minTheta, 2, 0.003);

	// Create KD Tree to search for neighbors
	// **************************************

	// Tree parameters
	nanoflann::KDTreeSingleIndexAdaptorParams tree_params(10 /* max leaf */);

	// Pointer to trees
	PointXYZ_KDTree *index;

	// Build KDTree for the first batch element
	index = new PointXYZ_KDTree(3, polar_cloud, tree_params);
	index->buildIndex();

	// Create search parameters
	nanoflann::SearchParams search_params;
	search_params.sorted = false;

	// Find neighbors and compute features
	// ***********************************

	// Variable for reserving memory
	size_t max_neighbs = 10;

	// Get all features in a parallel loop
	// #pragma omp parallel for shared(max_neighbs) schedule(dynamic, 10) num_threads(n_thread)
	for (size_t i = 0; i < N; i++)
	{
		if (planarity[i] < -0.5)
		{
			linearity[i] = -1.0f;
			continue;
		}
		vector<pair<size_t, float>> inds_dists;

		// Initial guess of neighbors size
		inds_dists.reserve(max_neighbs);

		// Find neighbors
		float query_pt[3] = {polar_cloud.pts[i].x, polar_cloud.pts[i].y, polar_cloud.pts[i].z};
		size_t n_neighbs = index->radiusSearch(query_pt, r2, inds_dists, search_params);

		// Update max count
		if (n_neighbs > max_neighbs)
		{
			// #pragma omp atomic
			max_neighbs = n_neighbs;
		}

		// Create Eigen matrix of neighbors (we use Eigen for PCA)
		vector<PointXYZ> neighbors;
		neighbors.reserve(n_neighbs);
		for (size_t j = 0; j < n_neighbs; j++)
			neighbors.push_back(points[inds_dists[j].first]);

		// Compute PCA
		vector<float> eigenvalues;
		vector<PointXYZ> eigenvectors;
		pca_features(neighbors, eigenvalues, eigenvectors);

		// Compute normals and score
		if (eigenvalues.size() < 3)
		{
			planarity[i] = -1.0f;
			linearity[i] = -1.0f;
		}
		else
		{
			// Score is 1 - sphericity equivalent to planarity + linearity
			planarity[i] = (eigenvalues[1] - eigenvalues[0]) / (eigenvalues[2] + 1e-9);
			linearity[i] = 1.0f - eigenvalues[1] / (eigenvalues[2] + 1e-9);
			if (eigenvectors[0].dot(points[i]) > 0)
				normals[i] = eigenvectors[0] * -1.0;
			else
				normals[i] = eigenvectors[0];
		}
	}
}

void smart_normal_score(vector<PointXYZ> &points,
						vector<PointXYZ> &polar_pts,
						vector<PointXYZ> &normals,
						vector<float> &scores)
{
	// Parameters
	float S0 = 0.4;
	float S1 = 1.0 - S0;
	float a0 = M_PI / 2;	  	// Max possible angle for which score is zero
	float a1 = a0 - M_PI / 24; 	// if angle > a1, whatever radius, score is better if angle is smaller (up to S0)
	float factor = S0 / (a0 - a1);
	float r0 = 4.0;
	float inv_sigma2 = 0.01f;

	// loop over all
	size_t i = 0;
	for (auto &s : scores)
	{
		if (s > -0.01)
		{
			float s2;
			float r = polar_pts[i].x;
			float angle = acos(min(abs(points[i].dot(normals[i]) / r), 1.0f));
			if (angle > a1)
				s2 = factor * (a0 - angle);
			else
				s2 = S0 + S1 * exp(-(pow(r - r0, 2)) * inv_sigma2);
			s = min(s * s2, 1.0f);
		}
		i++;
	}
}

void smart_icp_score(vector<PointXYZ> &polar_pts,
					 vector<PointXYZ> &normals,
					 vector<float> &heights,
					 vector<double> &scores)
{
	// There are more points close to the lidar, so we dont want to pick them to much.
	// Furthermore, points away carry more rotational information.

	// We also focus on picking more ground point (We can assume robot is nearly horinzontal)

	// Parameters
	double S0 = 1.0; 
	double S1 = 3.0; // -> prob to be picked when further away
	double r0 = 5.0;
	double H1 = 10.0; // -> prob to be picked for ground points
	double min_height = -0.4;
	double max_height = 1.8;
	double outlier_s = 0.01; // -> prob to be picked for outlier points
	double ground_dl = 0.15;
	double cos45 = cos(M_PI / 4);

	// Variables
	double S1m0 = S1 - S0;
	double inv_ro = 1 / r0;
	double cos20 = cos(20 * M_PI / 180);
	double H05 = 0.5 * (H1 - 1);

	// loop over all
	size_t i = 0;
	for (auto& s : scores)
	{
		// Score between 0 and 1 => mapped to [1.0, 2.0] (we take the higher score 2x more than the lower scores)
		s += 1.0; 
		
		// Then multiply by a distance score (We take furthest points 3x more often than closest one, this corrects density)
		s *= (S1 - S1m0 * exp(-pow((double)polar_pts[i].x * inv_ro, 2)));

		// Then multiply by a score based on normal angle. 10 more chance to pick if normal is within 20 degrees of vertical
		// But probability is never higher than 10
		if (normals[i].z > cos20)
		{
			s *= H05 * (cos(9 * acos(min((double)normals[i].z, 0.99999999))) + 1.0) + 1.0;
			if (s > H1)
				s = H1;
		}

		// Ignore points below ground and too high
		if (heights[i] < min_height || heights[i] > max_height)
			s = outlier_s;

		// Also ignor if vertical normal but not close to ground plane
		if (abs(heights[i]) > ground_dl && normals[i].z > cos45)
			s = outlier_s;


		i++;
	}
}

// void compare_map_to_frame(vector<PointXYZ> &frame_points,
// 						  vector<PointXYZ> &map_points,
// 						  vector<PointXYZ> &map_normals,
// 						  unordered_map<VoxKey, size_t> &map_samples,
// 						  Eigen::Matrix3d R_d,
// 						  Eigen::Vector3d T_d,
// 						  float theta_dl,
// 						  float phi_dl,
// 						  float map_dl,
// 						  vector<float> &movable_probs,
// 						  vector<int> &movable_counts)
// {
// 	int verbose = 0;
// 	vector<string> clock_str;
// 	vector<clock_t> t;
// 	if (verbose > 1)
// 	{
// 		clock_str.reserve(20);
// 		t.reserve(20);
// 		clock_str.push_back("Align frame ....... ");
// 		clock_str.push_back("Map limits ........ ");
// 		clock_str.push_back("Update full ....... ");
// 		clock_str.push_back("Polar frame ....... ");
// 		clock_str.push_back("Init grid ......... ");
// 		clock_str.push_back("Fill frustum ..... ");
// 		clock_str.push_back("Apply margin ...... ");
// 		clock_str.push_back("Cast frustum ..... ");
// 		clock_str.push_back("Test .............. ");
// 	}
// 	t.push_back(std::clock());

// 	////////////////
// 	// Parameters //
// 	////////////////

// 	float inv_theta_dl = 1.0 / theta_dl;
// 	float inv_phi_dl = 1.0 / phi_dl;
// 	float inv_map_dl = 1.0 / map_dl;
// 	float max_angle = 5 * M_PI / 12;
// 	float min_vert_cos = cos(M_PI / 3);

// 	// Convert alignment matrices to float
// 	Eigen::Matrix3f R = R_d.cast<float>();
// 	Eigen::Vector3f T = T_d.cast<float>();

// 	// Mask of the map point not updated yet
// 	vector<bool> not_updated(map_points.size(), true);

// 	///////////////////////////
// 	// Get map update limits //
// 	///////////////////////////

// 	// Align frame on map
// 	vector<PointXYZ> aligned_frame(frame_points);
// 	Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> aligned_mat((float *)aligned_frame.data(), 3, aligned_frame.size());
// 	aligned_mat = (R * aligned_mat).colwise() + T;

// 	t.push_back(std::clock());

// 	// Get limits
// 	PointXYZ min_P = min_point(aligned_frame) - PointXYZ(map_dl, map_dl, map_dl);
// 	PointXYZ max_P = max_point(aligned_frame) + PointXYZ(map_dl, map_dl, map_dl);

// 	t.push_back(std::clock());

// 	////////////////////////
// 	// Update full voxels //
// 	////////////////////////

// 	// Loop over aligned_frame
// 	VoxKey k0, k;
// 	for (auto &p : aligned_frame)
// 	{

// 		// Corresponding key
// 		k0.x = (int)floor(p.x * inv_map_dl);
// 		k0.y = (int)floor(p.y * inv_map_dl);
// 		k0.z = (int)floor(p.z * inv_map_dl);

// 		// Update the adjacent cells
// 		for (k.x = k0.x - 1; k.x < k0.x + 2; k.x++)
// 		{
// 			for (k.y = k0.y - 1; k.y < k0.y + 2; k.y++)
// 			{
// 				for (k.z = k0.z - 1; k.z < k0.z + 2; k.z++)
// 				{
// 					// Update count and movable at this point
// 					if (map_samples.count(k) > 0)
// 					{
// 						// Only update once
// 						size_t i0 = map_samples[k];
// 						if (not_updated[i0])
// 						{
// 							not_updated[i0] = false;
// 							movable_counts[i0] += 1;
// 							// movable_probs[i0] += 0; Useless line
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}

// 	t.push_back(std::clock());

// 	///////////////////////////////////
// 	// Create the free frustum grid //
// 	///////////////////////////////////

// 	// Get frame in polar coordinates
// 	vector<PointXYZ> polar_frame(frame_points);
// 	cart2pol_(polar_frame);

// 	t.push_back(std::clock());

// 	// Get grid limits
// 	PointXYZ minCorner = min_point(polar_frame);
// 	PointXYZ maxCorner = max_point(polar_frame);
// 	PointXYZ originCorner = minCorner - PointXYZ(0, 0.5 * theta_dl, 0.5 * phi_dl);

// 	// Dimensions of the grid
// 	size_t grid_n_theta = (size_t)floor((maxCorner.y - originCorner.y) / theta_dl) + 1;
// 	size_t grid_n_phi = (size_t)floor((maxCorner.z - originCorner.z) / phi_dl) + 1;

// 	// Initialize variables
// 	vector<float> frustum_radiuses(grid_n_theta * grid_n_phi, -1.0);
// 	size_t i_theta, i_phi, gridIdx;

// 	t.push_back(std::clock());

// 	// vector<PointXYZ> test_polar(polar_frame);
// 	// vector<float> test_itheta;
// 	// test_itheta.reserve(test_polar.size());
// 	// for (auto &p : test_polar)
// 	// {
// 	// 	p.y = (p.y - originCorner.y) * inv_theta_dl;
// 	// 	p.z = (p.z - originCorner.z) * inv_phi_dl;
// 	// 	test_itheta.push_back(floor(p.y));
// 	// }
// 	// save_cloud("test_polar.ply", test_polar, test_itheta);

// 	// Fill the frustum radiuses
// 	for (auto &p : polar_frame)
// 	{
// 		// Position of point in grid
// 		i_theta = (size_t)floor((p.y - originCorner.y) * inv_theta_dl);
// 		i_phi = (size_t)floor((p.z - originCorner.z) * inv_phi_dl);
// 		gridIdx = i_theta + grid_n_theta * i_phi;

// 		// Update the radius in cell
// 		if (frustum_radiuses[gridIdx] < 0)
// 			frustum_radiuses[gridIdx] = p.x;
// 		else if (p.x < frustum_radiuses[gridIdx])
// 			frustum_radiuses[gridIdx] = p.x;
// 	}

// 	t.push_back(std::clock());

// 	// Apply margin to free ranges
// 	float margin = map_dl;
// 	float frustum_alpha = theta_dl / 2;
// 	for (auto &r : frustum_radiuses)
// 	{
// 		float adapt_margin = r * frustum_alpha;
// 		if (margin < adapt_margin)
// 			r -= adapt_margin;
// 		else
// 			r -= margin;
// 	}

// 	t.push_back(std::clock());

// 	////////////////////////////
// 	// Apply frustum casting //
// 	////////////////////////////

// 	// Update free pixels
// 	float min_r = 2 * map_dl;
// 	size_t p_i = 0;
// 	Eigen::Matrix3f R_t = R.transpose();
// 	for (auto &p : map_points)
// 	{
// 		// Ignore points updated just now
// 		if (!not_updated[p_i])
// 		{
// 			p_i++;
// 			continue;
// 		}

// 		// Ignore points outside area of the frame
// 		if (p.x > max_P.x || p.y > max_P.y || p.z > max_P.z || p.x < min_P.x || p.y < min_P.y || p.z < min_P.z)
// 		{
// 			p_i++;
// 			continue;
// 		}

// 		// Align point in frame coordinates (and normal)
// 		PointXYZ xyz(p);
// 		PointXYZ nxyz(map_normals[p_i]);
// 		Eigen::Map<Eigen::Vector3f> p_mat((float *)&xyz, 3, 1);
// 		Eigen::Map<Eigen::Vector3f> n_mat((float *)&nxyz, 3, 1);
// 		p_mat = R_t * (p_mat - T);
// 		n_mat = R_t * n_mat;

// 		// Project in polar coordinates
// 		PointXYZ rtp = cart2pol(xyz);

// 		// Position of point in grid
// 		i_theta = (size_t)floor((rtp.y - originCorner.y) * inv_theta_dl);
// 		i_phi = (size_t)floor((rtp.z - originCorner.z) * inv_phi_dl);
// 		gridIdx = i_theta + grid_n_theta * i_phi;

// 		// Update movable prob
// 		if (rtp.x > min_r && rtp.x < frustum_radiuses[gridIdx])
// 		{
// 			// Do not update if normal is horizontal and perpendicular to ray (to avoid removing walls)
// 			if (abs(nxyz.z) > min_vert_cos)
// 			{
// 				movable_counts[p_i] += 1;
// 				movable_probs[p_i] += 1.0;
// 			}
// 			else
// 			{
// 				float angle = acos(min(abs(xyz.dot(nxyz) / rtp.x), 1.0f));
// 				if (angle < max_angle)
// 				{
// 					movable_counts[p_i] += 1;
// 					movable_probs[p_i] += 1.0;
// 				}
// 			}
			
// 		}
// 		p_i++;
// 	}

// 	t.push_back(std::clock());

// 	if (verbose > 1)
// 	{
// 		cout << endl
// 			 << "***********************" << endl;
// 		for (size_t i = 0; i < min(t.size() - 1, clock_str.size()); i++)
// 		{
// 			double duration = 1000 * (t[i + 1] - t[i]) / (double)CLOCKS_PER_SEC;
// 			cout << clock_str[i] << duration << " ms" << endl;
// 		}
// 		cout << "***********************" << endl
// 			 << endl;
// 	}
// }

void extract_lidar_frame_normals(vector<PointXYZ> &points,
								 vector<PointXYZ> &polar_pts,
								 vector<PointXYZ> &queries,
								 vector<PointXYZ> &polar_queries,
								 vector<int> &polar_rings,
								 vector<PointXYZ> &normals,
								 vector<float> &norm_scores,
								 vector<float> &polar_r2s)
{

	// Initialize variables
	// ********************

	// Result vectors
	normals = vector<PointXYZ>(polar_queries.size(), PointXYZ());
	norm_scores = vector<float>(polar_queries.size(), 0.0);
	// vector<float> visu_neighb(points.size(), 0.0);

	// Cloud variable for KDTree
	PointCloud polar_cloud;
	polar_cloud.pts = polar_pts;

	// Create KD Tree to search for neighbors
	// **************************************

	// Tree parameters
	nanoflann::KDTreeSingleIndexAdaptorParams tree_params(10 /* max leaf */);

	// Pointer to trees
	PointXYZ_KDTree *index;

	// Build KDTree for the first batch element
	index = new PointXYZ_KDTree(3, polar_cloud, tree_params);
	index->buildIndex();

	// Create search parameters
	nanoflann::SearchParams search_params;
	search_params.sorted = false;

	// Find neighbors and compute features
	// ***********************************

	// Variable for reserving memory
	size_t max_neighbs = 10;

	// Get all features in a parallel loop
	// #pragma omp parallel for shared(max_neighbs) schedule(dynamic, 10) num_threads(n_thread)
	for (size_t i = 0; i < polar_queries.size(); i++)
	{
		// Initial guess of neighbors size
		vector<pair<size_t, float>> inds_dists;
		inds_dists.reserve(max_neighbs);

		// Find neighbors
		float query_pt[3] = {polar_queries[i].x, polar_queries[i].y, polar_queries[i].z};
		size_t ring_i = polar_rings[i];
		size_t n_neighbs = index->radiusSearch(query_pt, polar_r2s[ring_i], inds_dists, search_params);

		// Update max count
		if (n_neighbs > max_neighbs)
		{
			// #pragma omp atomic
			max_neighbs = n_neighbs;
		}

		// Create a vector of the neighbors in carthesian coordinates
		vector<PointXYZ> neighbors;
		neighbors.reserve(n_neighbs);
		for (size_t j = 0; j < n_neighbs; j++)
			neighbors.push_back(points[inds_dists[j].first]);

		// // Save neighbors for debug
		// if (i % 43 == 1)
		// {
		// 	for (size_t j = 0; j < n_neighbs; j++)
		// 		visu_neighb[inds_dists[j].first] = (float)i;
		// }

		// Compute PCA
		vector<float> eigenvalues;
		vector<PointXYZ> eigenvectors;
		pca_features(neighbors, eigenvalues, eigenvectors);

		// Compute normals and score
		if (eigenvalues.size() < 3)
		{
			norm_scores[i] = -1.0f;
		}
		else
		{
			// Score is 1 - sphericity equivalent to planarity + linearity
			norm_scores[i] = 1.0f - eigenvalues[0] / (eigenvalues[2] + 1e-9);

			// Orient normal so that it always faces lidar origin
			if (eigenvectors[0].dot(queries[i]) > 0)
				normals[i] = eigenvectors[0] * -1.0;
			else
				normals[i] = eigenvectors[0];
		}
	}

	// string path000 = "/home/hth/Deep-Collison-Checker/SOGM-3D-2D-Net/results/";
	// string filepath00 = path000 + string("f_000_neighbs.ply");
	// string filepath01 = path000 + string("f_000_neighbs_polar.ply");
	// save_cloud(filepath00, points, visu_neighb);
	// save_cloud(filepath01, polar_pts, visu_neighb);

	// vector<float> f_1_2;
	// f_1_2.reserve(2 * polar_queries.size());
	// for (size_t i = 0; i < polar_queries.size(); i++)
	// 	f_1_2.push_back((float)polar_rings[i]);
	// for (size_t i = 0; i < polar_queries.size(); i++)
	// 	f_1_2.push_back(sqrt(polar_r2s[polar_rings[i]]));

	// string filepath02 = path000 + string("f_000_q_polar.ply");
	// save_cloud(filepath02, polar_queries, f_1_2);
}
