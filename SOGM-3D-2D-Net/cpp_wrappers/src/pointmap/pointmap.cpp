
#include "pointmap.h"


Eigen::Matrix4d pose_interp(float t, Eigen::Matrix4d const& H1, Eigen::Matrix4d const& H2, int verbose)
{
	// Assumes 0 < t < 1
	Eigen::Matrix3d R1 = H1.block(0, 0, 3, 3);
	Eigen::Matrix3d R2 = H2.block(0, 0, 3, 3);

	// Rotations to quaternions
	Eigen::Quaternion<double> rot1(R1);
	Eigen::Quaternion<double> rot2(R2);
	Eigen::Quaternion<double> rot3 = rot1.slerp(t, rot2);

	if (verbose > 0)
	{
		cout << R2.determinant() << endl;
		cout << R2 << endl;
		cout << "[" << rot1.x() << " " << rot1.y() << " " << rot1.z() << " " << rot1.w() << "] -> ";
		cout << "[" << rot2.x() << " " << rot2.y() << " " << rot2.z() << " " << rot2.w() << "] / " << t << endl;
		cout << "[" << rot3.x() << " " << rot3.y() << " " << rot3.z() << " " << rot3.w() << "]" << endl;
		cout << rot2.toRotationMatrix() << endl;
		cout << rot3.toRotationMatrix() << endl;
	}

	// Translations to vectors
	Eigen::Vector3d trans1 = H1.block(0, 3, 3, 1);
	Eigen::Vector3d trans2 = H2.block(0, 3, 3, 1);

	// Interpolation (not the real geodesic path, but good enough)
	Eigen::Affine3d result;
	result.translation() = (1.0 - t) * trans1 + t * trans2;
	result.linear() = rot1.slerp(t, rot2).normalized().toRotationMatrix();

	return result.matrix();
}

void get_ray_keys(PointXYZ &A,
				  PointXYZ &B,
				  float tile_dl,
				  float inv_tile_dl,
				  vector<VoxKey> &ray_keys)
{
	// Define the segment as (x, y, z) = A + t * AB with t in [0, 1]
	PointXYZ AB = B - A;
	PointXYZ inv_AB = PointXYZ(1 / AB.x, 1 / AB.y, 1 / AB.z);

	// Get min and max x, y, z from A and B
	PointXYZ minP = min_point(A, B);
	PointXYZ maxP = max_point(A, B);

	// Find intersections for x = n * dl
	float t;
	VoxKey k0;
	PointXYZ p0;
	for (size_t dim = 0; dim < 3; dim++)
	{
		k0[dim] = (int)floor(minP[dim] * inv_tile_dl);
		p0[dim] = (float)k0[dim] * tile_dl;
		while (p0[dim] < minP[dim])
		{
			k0[dim] += 1;
			p0[dim] += tile_dl;
		}
		while (p0[dim] <= maxP[dim])
		{
			// Get t, should always be between 0 and 1
			t = (p0[dim] - A[dim]) * inv_AB[dim];

			// Get the intersected Voxel
			size_t dim1 = (dim + 1) % 3;
			size_t dim2 = (dim + 2) % 3;
			p0[dim1] = A[dim1] + t * AB[dim1];
			p0[dim2] = A[dim2] + t * AB[dim2];
			k0[dim1] = (int)floor(p0[dim1] * inv_tile_dl);
			k0[dim2] = (int)floor(p0[dim2] * inv_tile_dl);
			ray_keys.push_back(k0);

			// Get the other intersected Voxel
			k0[dim] -= 1;
			ray_keys.push_back(k0);

			// Go to next intersection
			k0[dim] += 2;
			p0[dim] += tile_dl;
		}

	}

}

void PointMapPython::update(vector<PointXYZ>& points0,
	vector<PointXYZ>& normals0,
	vector<float>& scores0)
{
	// Initialize variables
	// ********************

	// New limits of the map
	PointXYZ minCorner = min_point(points0);
	PointXYZ maxCorner = max_point(points0);
	PointXYZ originCorner = floor(minCorner * (1 / dl) - PointXYZ(1, 1, 1)) * dl;

	// Check old limits of the map
	if (points.size() > 0)
	{
		PointXYZ oldMinCorner = min_point(points);
		PointXYZ oldMaxCorner = max_point(points);
		PointXYZ oldOriginCorner = floor(oldMinCorner * (1 / dl) - PointXYZ(1, 1, 1)) * dl;
		originCorner = min_point(originCorner, oldOriginCorner);
		maxCorner = max_point(maxCorner, oldMaxCorner);
	}

	// Dimensions of the grid
	size_t sampleNX = (size_t)floor((maxCorner.x - originCorner.x) / dl) + 2;
	size_t sampleNY = (size_t)floor((maxCorner.y - originCorner.y) / dl) + 2;

	// Create the sampled map
	// **********************

	unordered_map<size_t, MapVoxelData> samples;

	//// USe following to know if we need to reserve before inserting elements
	samples.reserve(points.size() + points0.size());
	//std::cout << "current max_load_factor: " << samples.max_load_factor() << std::endl;
	//std::cout << "current size: " << samples.size() << std::endl;
	//std::cout << "current bucket_count: " << samples.bucket_count() << std::endl;
	//std::cout << "current load_factor: " << samples.load_factor() << std::endl;

	// Add existing map points to the hashmap.
	if (points.size() > 0)
	{
		init_samples(originCorner, maxCorner, samples);
	}

	// Add new points to the hashmap
	add_samples(points0, normals0, scores0, originCorner, maxCorner, samples);

	// Convert hmap to vectors
	points.reserve(samples.size());
	normals.reserve(samples.size());
	scores.reserve(samples.size());
	counts.reserve(samples.size());
	size_t i = 0;
	size_t iX, iY, iZ, centroidIdx;
	for (auto& v : samples)
	{
		PointXYZ centroid = v.second.centroid * (1.0 / v.second.count);
		iX = (size_t)floor((centroid.x - originCorner.x) / dl);
		iY = (size_t)floor((centroid.y - originCorner.y) / dl);
		iZ = (size_t)floor((centroid.z - originCorner.z) / dl);
		centroidIdx = iX + sampleNX * iY + sampleNX * sampleNY * iZ;

		if (v.second.occupied && centroidIdx == v.first)
		{
			v.second.normal *= 1.0 / (sqrt(v.second.normal.sq_norm()) + 1e-6);
			if (i < points.size())
			{
				points[i] = centroid;
				normals[i] = v.second.normal;
				scores[i] = v.second.score;
				counts[i] = v.second.count;
			}
			else
			{
				points.push_back(v.second.centroid * (1.0 / v.second.count));
				normals.push_back(v.second.normal);
				scores.push_back(v.second.score);
				counts.push_back(v.second.count);
			}
			i++;
		}
	}
}


void PointMapPython::init_samples(const PointXYZ originCorner,
	const PointXYZ maxCorner,
	unordered_map<size_t, MapVoxelData>& samples)
{
	// Dimensions of the grid
	size_t sampleNX = (size_t)floor((maxCorner.x - originCorner.x) / dl) + 2;
	size_t sampleNY = (size_t)floor((maxCorner.y - originCorner.y) / dl) + 2;

	// Initialize variables
	size_t i = 0;
	size_t iX, iY, iZ, mapIdx;

	for (auto& p : points)
	{
		// Position of point in sample map
		iX = (size_t)floor((p.x - originCorner.x) / dl);
		iY = (size_t)floor((p.y - originCorner.y) / dl);
		iZ = (size_t)floor((p.z - originCorner.z) / dl);

		// Update the point cell
		mapIdx = iX + sampleNX * iY + sampleNX * sampleNY * iZ;
		samples.emplace(mapIdx, MapVoxelData(p * counts[i], normals[i] * counts[i], scores[i], counts[i]));
		i++;
	}
}


void PointMapPython::add_samples(const vector<PointXYZ>& points0,
	const vector<PointXYZ>& normals0,
	const vector<float>& scores0,
	const PointXYZ originCorner,
	const PointXYZ maxCorner,
	unordered_map<size_t, MapVoxelData>& samples)
{
	// Dimensions of the grid
	size_t sampleNX = (size_t)floor((maxCorner.x - originCorner.x) / dl) + 2;
	size_t sampleNY = (size_t)floor((maxCorner.y - originCorner.y) / dl) + 2;

	// Initialize variables
	float r2 = dl * 1.5;
	r2 *= r2;
	size_t i = 0;
	size_t iX, iY, iZ, mapIdx;

	for (auto& p : points0)
	{
		// Position of point in sample map
		iX = (size_t)floor((p.x - originCorner.x) / dl);
		iY = (size_t)floor((p.y - originCorner.y) / dl);
		iZ = (size_t)floor((p.z - originCorner.z) / dl);

		// Update the adjacent cells
		for (size_t ix = iX - 1; ix < iX + 2; ix++)
		{

			for (size_t iy = iY - 1; iy < iY + 2; iy++)
			{

				for (size_t iz = iZ - 1; iz < iZ + 2; iz++)
				{
					// Find distance to cell center
					mapIdx = ix + sampleNX * iy + sampleNX * sampleNY * iz;
					PointXYZ cellCenter(ix + 0.5, iy + 0.5, iz + 0.5);
					cellCenter = cellCenter * dl + originCorner;
					float d2 = (cellCenter - p).sq_norm();

					// Update barycenter if in range
					if (d2 < r2)
					{
						if (samples.count(mapIdx) < 1)
							samples.emplace(mapIdx, MapVoxelData(p));
						else
							samples[mapIdx].update_centroid(p);
					}
				}
			}
		}

		// Update the point cell
		mapIdx = iX + sampleNX * iY + sampleNX * sampleNY * iZ;
		samples[mapIdx].update_normal(scores0[i], normals0[i]);
		i++;
	}
}


void PointMap::update_movable_pts(vector<PointXYZ> &frame_points,
								  vector<float> &frame_alphas,
								  Eigen::Matrix4d &H0,
								  Eigen::Matrix4d &H1,
								  float theta_dl,
								  float phi_dl,
								  int n_slices,
								  vector<float> &ring_angles,
								  vector<float> &ring_mids,
								  vector<float> &ring_d_thetas,
								  vector<float> &movable_probs,
								  vector<int> &movable_counts)
{
	///////////////
	// Verbosity //
	///////////////

	int verbose = 0;
	vector<string> clock_str;
	vector<clock_t> t;
	if (verbose > 1)
	{
		clock_str.reserve(20);
		t.reserve(20);
		clock_str.push_back("Align frame ....... ");
		clock_str.push_back("Update full ....... ");
		clock_str.push_back("Get slices ........ ");
		clock_str.push_back("Get polar ......... ");
		clock_str.push_back("Slices phi ........ ");
		clock_str.push_back("Init frustums ..... ");
		clock_str.push_back("Fill frustums ..... ");
		clock_str.push_back("Apply amrgins ..... ");
		clock_str.push_back("Create tiles ...... ");
		clock_str.push_back("Frustum casting ... ");
		clock_str.push_back("Test .............. ");
	}
	t.push_back(std::clock());
	// cout << clock_str[t.size() - 1] << endl;
	if (verbose > 2)
		cout << clock_str[t.size() - 1] << endl;

	////////////////
	// Parameters //
	////////////////

	float inv_theta_dl = 1.0 / theta_dl;
	float inv_phi_dl = 1.0 / phi_dl;
	float inv_dl = 1.0 / dl;
	//float max_angle = 5 * M_PI / 12;
	//float min_vert_cos = cos(M_PI / 3);
	bool motion_distortion = n_slices > 1;

	// Convert alignment matrices to float
	Eigen::Matrix3f R = (H1.block(0, 0, 3, 3)).cast<float>();
	Eigen::Vector3f T = (H1.block(0, 3, 3, 1)).cast<float>();

	// Mask of the map point not updated yet
	vector<bool> not_updated(cloud.pts.size(), true);

	///////////////////////////
	// Get map update limits //
	///////////////////////////

	// Align frame on map
	vector<PointXYZ> aligned_frame(frame_points);
	vector<PointXYZ> aligned_origins;
	if (motion_distortion)
	{
		// Update map taking motion distortion into account
		size_t i_inds = 0;
		aligned_origins.reserve(frame_points.size());
		Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> aligned_mat((float *)aligned_frame.data(), 3, aligned_frame.size());
		for (auto& alpha : frame_alphas)
		{
			Eigen::Matrix4d H_rect = pose_interp(alpha, H0, H1, 0);
			Eigen::Matrix3f R_rect = (H_rect.block(0, 0, 3, 3)).cast<float>();
			Eigen::Vector3f T_rect = (H_rect.block(0, 3, 3, 1)).cast<float>();
			aligned_mat.col(i_inds) = (R_rect * aligned_mat.col(i_inds)) + T_rect;
			aligned_origins.push_back(PointXYZ(T_rect(0), T_rect(1), T_rect(2)));
			i_inds++;
		}
	}
	else
	{
		Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> aligned_mat((float *)aligned_frame.data(), 3, aligned_frame.size());
		aligned_mat = (R * aligned_mat).colwise() + T;
	}

	t.push_back(std::clock());
	if (verbose > 2)
		cout << clock_str[t.size() - 1] << endl;

	////////////////////////
	// Update full voxels //
	////////////////////////


	// Loop over aligned_frdlame
	VoxKey k0;
	unordered_map<VoxKey, size_t> test_keys;
	test_keys.reserve(aligned_frame.size());
	vector<VoxKey> unique_keys;
	unique_keys.reserve(aligned_frame.size());
	for (auto &p : aligned_frame)
	{
		// Corresponding key
		k0.x = (int)floor(p.x * inv_dl);
		k0.y = (int)floor(p.y * inv_dl);
		k0.z = (int)floor(p.z * inv_dl);

		// Update the point count
		if (test_keys.count(k0) < 1)
		{
			test_keys.emplace(k0, 0);
			unique_keys.push_back(k0);
		}
	}
	
	vector<VoxKey> grown_keys(unique_keys);
	VoxKey k;
	grown_keys.reserve(aligned_frame.size());
	for (auto &k1 : unique_keys)
	{
		// Update the adjacent cells
		for (k.x = k1.x - 1; k.x < k1.x + 2; k.x++)
		{
			for (k.y = k1.y - 1; k.y < k1.y + 2; k.y++)
			{
				for (k.z = k1.z - 1; k.z < k1.z + 2; k.z++)
				{
					if (test_keys.count(k) < 1)
					{
						test_keys.emplace(k, 0);
						grown_keys.push_back(k);
					}
				}
			}
		}
	}

	for (auto &k2 : grown_keys)
	{
		if (samples.count(k2) > 0)
		{	
			// Only update once
			size_t i0 = samples[k2];
			not_updated[i0] = false;
			movable_counts[i0] += 1;
			// movable_probs[i0] += 0; Useless line
		}
	}

	t.push_back(std::clock());
	if (verbose > 2)
		cout << clock_str[t.size() - 1] << endl;

	////////////////
	// Get slices //
	////////////////
	

	// We approximate buy considering alphas are increasing 
	vector<size_t> slice_inds;
	slice_inds.reserve(n_slices);
	float d_alpha = 1.0f / (float)n_slices;
	float current_alpha = d_alpha;
	slice_inds.push_back(0);
	if (motion_distortion)
	{
		for (int i = 0; i < (int)frame_alphas.size(); i++)
		{
			if (frame_alphas[i] >= current_alpha)
			{
				slice_inds.push_back(i);
				current_alpha += d_alpha;
				if (current_alpha > (1 - d_alpha / 2))
					break;
			}
		}
	}
	slice_inds.push_back(frame_alphas.size());

	// Get slices limits and poses
	vector<PointXYZ> min_P(n_slices);
	vector<PointXYZ> max_P(n_slices);
	vector<Eigen::Matrix3f> R_t_slices(n_slices);
	vector<Eigen::Vector3f> T_slices(n_slices);
	for (int s = 0; s < n_slices; s++)
	{
		vector<PointXYZ>::const_iterator first = aligned_frame.begin() + slice_inds[s];
		vector<PointXYZ>::const_iterator last = aligned_frame.begin() + slice_inds[s + 1];
		vector<PointXYZ> tmp_slice(first, last);	

		// Get limits
		min_P[s] = min_point(tmp_slice) - PointXYZ(dl, dl, dl);
		max_P[s] = max_point(tmp_slice) + PointXYZ(dl, dl, dl);
		
		float slice_alpha = ((float)s + 0.5f) / (float)n_slices;
		Eigen::Matrix4d H_rect = pose_interp(slice_alpha, H0, H1, 0);
		R_t_slices[s] = (H_rect.block(0, 0, 3, 3)).cast<float>().transpose();
		T_slices[s] = (H_rect.block(0, 3, 3, 1)).cast<float>();
	}


	t.push_back(std::clock());
	if (verbose > 2)
		cout << clock_str[t.size() - 1] << endl;

	///////////////////////////
	// Get polar coordinates //
	///////////////////////////

	// Get frame in polar coordinates
	vector<PointXYZ> polar_frame(frame_points);
	cart2pol_(polar_frame);

	// Arrange phi so that minimum corresponds to first slice
	for (auto& p : polar_frame)
			p.z = -p.z;

	float phi0 = polar_frame[0].z;
	float phi1 = polar_frame[0].z + M_PI / 8;
	float delta_phi = 2 * M_PI;
	int phi_i = 0;
	int phi_i0 = polar_frame.size() / 8;
	int s_phi_i0 = n_slices / 8 + 1;
	for (auto& p : polar_frame)
	{
		if (phi_i > phi_i0 && p.z < phi1)
			p.z = p.z - phi0 + delta_phi;
		else
			p.z = p.z - phi0;
		phi_i++;
	}
	
	t.push_back(std::clock());
	if (verbose > 2)
		cout << clock_str[t.size() - 1] << endl;

	////////////////////
	// Get slices phi //
	////////////////////

	// Get average phi angle for each slice
	vector<float> slices_phi(n_slices);
	for (int s = 0; s < n_slices; s++)
	{
		float sum = 0;
		for (int j = slice_inds[s]; j < (int)slice_inds[s + 1]; j++)
			sum += polar_frame[j].z;
		slices_phi[s] = sum / (float)(slice_inds[s + 1] - slice_inds[s]);
	}
	
	t.push_back(std::clock());
	if (verbose > 2)
		cout << clock_str[t.size() - 1] << endl;

	///////////////////////
	// Init frustum grid //
	///////////////////////

	// Get grid limits
	PointXYZ minCorner = min_point(polar_frame);
	PointXYZ maxCorner = max_point(polar_frame);
	PointXYZ originCorner = minCorner - PointXYZ(0, 0.5 * theta_dl, 0.5 * phi_dl);

	// string path000 = "/home/hth/Deep-Collison-Checker/SOGM-3D-2D-Net/results/";
	// save_cloud(path000 + string("f_polar.ply"), polar_frame);

	// Dimensions of the grid
	size_t grid_n_theta = (size_t)floor((maxCorner.y - originCorner.y) / theta_dl) + 1;
	size_t grid_n_phi = (size_t)floor((maxCorner.z - originCorner.z) / phi_dl) + 1;
	
	// Update max corner accordingly
	maxCorner.y = originCorner.y + ((float)grid_n_theta - 0.01) * theta_dl ;
	maxCorner.z = originCorner.z + ((float)grid_n_phi - 0.01) * phi_dl ;

	// Extension of theta indices given irregular rings
	vector<size_t> theta_to_ring(grid_n_theta);
	size_t ring_i = 0;
	for (size_t i_th = 0; i_th < grid_n_theta; i_th++)
	{
		float theta = originCorner.y + theta_dl * ((float)i_th + 0.5);
		if (ring_i < ring_mids.size() && theta > ring_mids[ring_i])
			ring_i++;
		theta_to_ring[i_th] = ring_i;
	}

	// Initialize variables
	size_t grid_n_ring = ring_mids.size() + 1;
	vector<float> frustum_radiuses(grid_n_ring * grid_n_phi, -1.0);

	t.push_back(std::clock());
	if (verbose > 2)
		cout << clock_str[t.size() - 1] << endl;

	///////////////////////
	// Fill frustum grid //
	///////////////////////

	// Fill the frustum radiuses
	size_t i_theta, i_phi, i_ring, gridIdx;
	for (auto &p : polar_frame)
	{
		// Position of point in grid
		i_theta = (size_t)floor((p.y - originCorner.y) * inv_theta_dl);
		i_phi = (size_t)floor((p.z - originCorner.z) * inv_phi_dl);
		i_ring = theta_to_ring[i_theta];
		gridIdx = i_ring + grid_n_ring * i_phi;

		// Update the radius in cell
		if (frustum_radiuses[gridIdx] < 0)
			frustum_radiuses[gridIdx] = p.x;
		else if (p.x < frustum_radiuses[gridIdx])
			frustum_radiuses[gridIdx] = p.x;
	}

	t.push_back(std::clock());
	if (verbose > 2)
		cout << clock_str[t.size() - 1] << endl;

	///////////////////
	// Apply margins //
	///////////////////

	// Apply margin to free ranges
	float margin = dl;
	for (int j = 0; j < (int)frustum_radiuses.size(); j++)
	{
		i_ring = j % (int)grid_n_ring;
		float frustum_alpha = ring_d_thetas[i_ring];
		float adapt_margin = frustum_radiuses[j] * frustum_alpha;
		if (margin < adapt_margin)
			frustum_radiuses[j] -= 2 * adapt_margin;
		else
			frustum_radiuses[j] -= 2 * margin;
	}

	// vector<PointXYZ> frustum_pts(grid_n_ring * grid_n_phi);
	// for (int j = 0; j < frustum_radiuses.size(); j++)
	// {
	// 	i_ring = j % (int)grid_n_ring;
	// 	i_phi = (j - (int)i_theta) / (int)grid_n_ring;

	// 	float theta = ring_angles[i_ring];

	// 	//float theta = originCorner.y + theta_dl * ((float)i_theta + 0.5);
	// 	float phi = originCorner.z + phi_dl * ((float)i_phi + 0.5);
	// 	frustum_pts[j] = PointXYZ(frustum_radiuses[j], theta, phi);
	// }
	// save_cloud(path000 + string("frustum_polar.ply"), frustum_pts);
	// cout << " OK" << endl;

	t.push_back(std::clock());
	if (verbose > 2)
		cout << clock_str[t.size() - 1] << endl;

	///////////////////////////////////////////
	// Create tiles for fast slice detection //
	///////////////////////////////////////////

	// Inverse of sample dl
	float tile_dl = 0.5;
	float inv_tile_dl = 1.0 / tile_dl;

	// Initialize variables
	VoxKey minK(1e6, 1e6, 1e6);
	VoxKey maxK(-1e6, -1e6, -1e6);
	for (auto& B : aligned_frame)
	{
		k0.x = (int)floor(B.x * inv_tile_dl);
		k0.y = (int)floor(B.y * inv_tile_dl);
		k0.z = (int)floor(B.z * inv_tile_dl);
		minK.update_min(k0);
		maxK.update_max(k0);
	}
	
	// Create a dense vector of the tile existence for faster access
	VoxKey nK = maxK - minK + VoxKey(1, 1, 1);
	vector<int> existing_tiles(nK.x * nK.y * nK.z, -1);
	vector<int> tile_data;
	tile_data.reserve(nK.x * nK.y * nK.z * n_slices);

	int fp_i = 0;
	int tile_s = 0;
	int current_tile_i = 0;
	for (auto& B : aligned_frame)
	{
		// Sample points along the ray to get all ray tiles
		vector<VoxKey> ray_keys;
		get_ray_keys(aligned_origins[fp_i], B, tile_dl, inv_tile_dl, ray_keys);

		for (auto &ray_k: ray_keys)
		{
			// Ensure inside limits
			if (ray_k.x >= minK.x && ray_k.x <= maxK.x &&
				ray_k.y >= minK.y && ray_k.y <= maxK.y &&
				ray_k.z >= minK.z && ray_k.z <= maxK.z)
			{
				// Get vectorized index
				VoxKey tile_key0 = ray_k - minK;
				int vec_key = tile_key0.x + nK.x * tile_key0.y + nK.x * nK.y * tile_key0.z;
				int tile_ind = existing_tiles[vec_key];
				
				if (tile_ind < 0)
				{
					existing_tiles[vec_key] = n_slices * current_tile_i;
					for (int s = 0; s < n_slices; s++)
						tile_data.push_back(0);
					tile_data[current_tile_i * n_slices + tile_s] = 1;
					current_tile_i++;
				}
				else
				{
					tile_data[existing_tiles[vec_key] + tile_s] = 1;
				}
			}

		}

		// Increment point index
		fp_i++;

		// Change slice
		if (fp_i >= (int)slice_inds[tile_s + 1])
			tile_s++;
	}



	// PointXYZ minP = min_point(aligned_frame) - PointXYZ(dl, dl, dl) * 5;
	// PointXYZ maxP = max_point(aligned_frame) + PointXYZ(dl, dl, dl) * 5;
	
	// vector<PointXYZ> slices_points;

	// for (float ix = minP.x; ix < maxP.x; ix += tile_dl/4)
	// {
	// 	for (float iy = minP.y; iy < maxP.y; iy += tile_dl/4)
	// 	{
	// 		for (float iz = minP.z; iz < maxP.z; iz += tile_dl/4)
	// 			slices_points.push_back(PointXYZ(ix, iy, iz));
	// 	}
	// }

	// vector<float> slices_features;
	// for (int s = 0; s < n_slices; s++)
	// {
	// 	for (auto &sp: slices_points)
	// 	{
	// 		k0.x = (int)floor(sp.x * inv_tile_dl);
	// 		k0.y = (int)floor(sp.y * inv_tile_dl);
	// 		k0.z = (int)floor(sp.z * inv_tile_dl);

	// 		// Ensure inside limits
	// 		if (k0.x < minK.x || k0.x > maxK.x ||
	// 			k0.y < minK.y || k0.y > maxK.y ||
	// 			k0.z < minK.z || k0.z > maxK.z)
	// 		{
	// 			slices_features.push_back(0);
	// 			continue;
	// 		}

	// 		// Get vectorized index
	// 		VoxKey tile_key0 = k0 - minK;
	// 		int vec_key = tile_key0.x + nK.x * tile_key0.y + nK.x * nK.y * tile_key0.z;
			
	// 		// Ignore points outside tiles
	// 		int tile_ind = existing_tiles[vec_key];
	// 		if (tile_ind < 0)
	// 		{
	// 			slices_features.push_back(1);
	// 			continue;
	// 		}

	// 		if (tile_data[tile_ind + s] > 0)
	// 			slices_features.push_back(3);
	// 		else
	// 			slices_features.push_back(2);

	// 	}
	// }

	// cout << slices_points.size() << endl;
	// cout << slices_features.size() << endl;
	// cout << slices_features.size() / n_slices << endl;
	
	// string path000 = "/home/hth/Deep-Collison-Checker/SOGM-3D-2D-Net/results/";
	// save_cloud(path000 + string("f_tiles.ply"), slices_points, slices_features);


	// vector<float> slices_features2;
	// tile_s = 0;
	// fp_i = 0;
	// for (auto& B : aligned_frame)
	// {
	// 	slices_features2.push_back(tile_s);

	// 	// Increment point index
	// 	fp_i++;

	// 	// Change slice
	// 	if (fp_i >= slice_inds[tile_s + 1])
	// 		tile_s++;
	// }

	// save_cloud(path000 + string("f_slices.ply"), aligned_frame, slices_features2);



	t.push_back(std::clock());
	if (verbose > 2)
		cout << clock_str[t.size() - 1] << endl;


	////////////////////////
	// Compute cloud_keys //
	////////////////////////

	// Only once
	if (cloud_keys.size() < 1)
	{
		cloud_keys.reserve(cloud.pts.size());
		for (auto &p : cloud.pts)
		{
			// Get which slice are candidates for this point
			k0.x = (int)floor(p.x * inv_tile_dl);
			k0.y = (int)floor(p.y * inv_tile_dl);
			k0.z = (int)floor(p.z * inv_tile_dl);
			cloud_keys.push_back(k0);
		}
	}

	///////////////////////////
	// Apply frustum casting //
	///////////////////////////

	// Update free pixels
	float min_r = 2 * dl;
	size_t p_i = 0;
	//Eigen::Matrix4d H_half = pose_interp(0.5, H0, H1, 0);
	//Eigen::Matrix3f R_half = (H_half.block(0, 0, 3, 3)).cast<float>();
	//Eigen::Vector3f T_half = (H_half.block(0, 3, 3, 1)).cast<float>();
	//Eigen::Matrix3f R_t = R.transpose();
	//Eigen::Matrix3f R_half_t = R_half.transpose();
	for (auto &tile_key : cloud_keys)
	{
		// Ignore points updated just now
		if (!not_updated[p_i])
		{
			p_i++;
			continue;
		}

		// // Ignore points outside area of the frame
		// if (p.x > max_P.x || p.y > max_P.y || p.z > max_P.z || p.x < min_P.x || p.y < min_P.y || p.z < min_P.z)
		// {
		// 	p_i++;
		// 	continue;
		// }

		// // Get which slice are candidates for this point
		// k0.x = (int)floor(p.x * inv_tile_dl);
		// k0.y = (int)floor(p.y * inv_tile_dl);
		// k0.z = (int)floor(p.z * inv_tile_dl);

		// Ensure inside limits
		if (tile_key.x < minK.x || tile_key.x > maxK.x ||
			tile_key.y < minK.y || tile_key.y > maxK.y ||
			tile_key.z < minK.z || tile_key.z > maxK.z)
		{
			p_i++;
			continue;
		}

		// Get vectorized index
		VoxKey tile_key0 = tile_key - minK;
		int vec_key = tile_key0.x + nK.x * tile_key0.y + nK.x * nK.y * tile_key0.z;
		
		// Ignore points outside tiles
		int tile_ind = existing_tiles[vec_key];
		if (tile_ind < 0)
		{
			p_i++;
			continue;
		}

		// Get the best slice according to phi angle
		int best_s = -1;
		float min_d_phi = 4 * M_PI;
		PointXYZ best_xyz;
		PointXYZ best_nxyz;
		PointXYZ best_rtp;
		for (int s = 0; s < n_slices; s++)
		{
			if (tile_data[tile_ind + s] > 0)
			{
				// Align point in frame coordinates (and normal)
				PointXYZ xyz(cloud.pts[p_i]);
				PointXYZ nxyz(normals[p_i]);
				Eigen::Map<Eigen::Vector3f> p_mat((float *)&xyz, 3, 1);
				Eigen::Map<Eigen::Vector3f> n_mat((float *)&nxyz, 3, 1);
				p_mat = R_t_slices[s] * (p_mat - T_slices[s]);
				n_mat = R_t_slices[s] * n_mat;

				// Project in polar coordinates
				PointXYZ rtp = cart2pol(xyz);

				// Arrange phi
				rtp.z = -rtp.z;
				if (s > s_phi_i0 && rtp.z < phi1)
					rtp.z = rtp.z - phi0 + delta_phi;
				else
					rtp.z = rtp.z - phi0;

				// Ensure rtp is in the frustum grid
				if (rtp.y > originCorner.y && rtp.y < maxCorner.y && rtp.z > originCorner.z && rtp.z < maxCorner.z)
				{
					// Update best_slice
					float d_phi = abs(rtp.z - slices_phi[s]);
					if (d_phi < min_d_phi)
					{
						best_s = s;
						min_d_phi = d_phi;
						best_xyz = xyz;
						best_nxyz = nxyz;
						best_rtp = rtp;
					}
				}
			}
		}

		// Perform ray_casting only if a tile was found
		if (best_s > -0.5)
		{
			// Position of point in grid
			i_theta = (size_t)floor((best_rtp.y - originCorner.y) * inv_theta_dl);
			i_phi = (size_t)floor((best_rtp.z - originCorner.z) * inv_phi_dl);

			i_ring = theta_to_ring[i_theta];
			gridIdx = i_ring + grid_n_ring * i_phi;

			// Update movable prob
			if (best_rtp.x > min_r && best_rtp.x < frustum_radiuses[gridIdx])
			{


				movable_counts[p_i] += 1;
				movable_probs[p_i] += 1.0;

				// // Do not update if normal is horizontal and perpendicular to ray (to avoid removing walls)
				// if (abs(best_nxyz.z) > min_vert_cos)
				// {
				// 	movable_counts[p_i] += 1;
				// 	movable_probs[p_i] += 1.0;
				// }
				// else
				// {
				// 	float angle = acos(min(abs(best_xyz.dot(best_nxyz) / best_rtp.x), 1.0f));
				// 	if (angle < max_angle)
				// 	{
				// 		movable_counts[p_i] += 1;
				// 		movable_probs[p_i] += 1.0;
				// 	}
				// }
			}
		}

		p_i++;
	}

	t.push_back(std::clock());

	if (verbose > 1)
	{
		cout << endl
			 << "***********************" << endl;
		for (size_t i = 0; i < min(t.size() - 1, clock_str.size()); i++)
		{
			double duration = 1000 * (t[i + 1] - t[i]) / (double)CLOCKS_PER_SEC;
			cout << clock_str[i] << duration << " ms" << endl;
		}
		cout << "***********************" << endl
			 << endl;
	}
}