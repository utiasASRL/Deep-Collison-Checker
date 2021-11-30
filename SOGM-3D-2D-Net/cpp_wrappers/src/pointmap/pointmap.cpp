
#include "pointmap.h"


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

void PointMap::update_movable(vector<PointXYZ> &frame_points,
							  Eigen::Matrix4d &H0,
							  Eigen::Matrix4d &H1,
							  float theta_dl,
							  float phi_dl,
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
		clock_str.push_back("Map limits ........ ");
		clock_str.push_back("Update full ....... ");
		clock_str.push_back("Polar frame ....... ");
		clock_str.push_back("Init grid ......... ");
		clock_str.push_back("Fill frustrum ..... ");
		clock_str.push_back("Apply margin ...... ");
		clock_str.push_back("Cast frustrum ..... ");
		clock_str.push_back("Test .............. ");
	}
	t.push_back(std::clock());
	// cout << clock_str[t.size() - 1] << endl;

	////////////////
	// Parameters //
	////////////////

	float inv_theta_dl = 1.0 / theta_dl;
	float inv_phi_dl = 1.0 / phi_dl;
	float inv_dl = 1.0 / dl;
	float max_angle = 5 * M_PI / 12;
	float min_vert_cos = cos(M_PI / 3);

	// Convert alignment matrices to float
	Eigen::Matrix3f R = (H1.block(0, 0, 3, 3)).cast<float>();
	Eigen::Vector3f T = (H1.block(0, 3, 3, 1)).cast<float>();

	bool motion_distortion = false;
	Eigen::Matrix3f R0;
	Eigen::Vector3f T0;
	if (H0.lpNorm<1>() > 0.001)
	{
		motion_distortion = true;
		R0 = (H1.block(0, 0, 3, 3)).cast<float>();
		T0 = (H1.block(0, 3, 3, 1)).cast<float>();
	}

	// Mask of the map point not updated yet
	vector<bool> not_updated(cloud.pts.size(), true);

	///////////////////////////
	// Get map update limits //
	///////////////////////////

	// Align frame on map
	vector<PointXYZ> aligned_frame(frame_points);
	Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> aligned_mat((float *)aligned_frame.data(), 3, aligned_frame.size());
	aligned_mat = (R * aligned_mat).colwise() + T;

	t.push_back(std::clock());

	// Get limits
	PointXYZ min_P = min_point(aligned_frame) - PointXYZ(dl, dl, dl);
	PointXYZ max_P = max_point(aligned_frame) + PointXYZ(dl, dl, dl);

	t.push_back(std::clock());

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

	///////////////////////////////////
	// Create the free frustrum grid //
	///////////////////////////////////

	// Get frame in polar coordinates
	vector<PointXYZ> polar_frame(frame_points);
	cart2pol_(polar_frame);

	t.push_back(std::clock());

	// Get grid limits
	PointXYZ minCorner = min_point(polar_frame);
	PointXYZ maxCorner = max_point(polar_frame);
	PointXYZ originCorner = minCorner - PointXYZ(0, 0.5 * theta_dl, 0.5 * phi_dl);

	// Dimensions of the grid
	size_t grid_n_theta = (size_t)floor((maxCorner.y - originCorner.y) / theta_dl) + 1;
	size_t grid_n_phi = (size_t)floor((maxCorner.z - originCorner.z) / phi_dl) + 1;

	// Initialize variables
	vector<float> frustrum_radiuses(grid_n_theta * grid_n_phi, -1.0);
	size_t i_theta, i_phi, gridIdx;

	t.push_back(std::clock());

	// vector<PointXYZ> test_polar(polar_frame);
	// vector<float> test_itheta;
	// test_itheta.reserve(test_polar.size());
	// for (auto &p : test_polar)
	// {
	// 	p.y = (p.y - originCorner.y) * inv_theta_dl;
	// 	p.z = (p.z - originCorner.z) * inv_phi_dl;
	// 	test_itheta.push_back(floor(p.y));
	// }
	// save_cloud("test_polar.ply", test_polar, test_itheta);

	// Fill the frustrum radiuses
	for (auto &p : polar_frame)
	{
		// Position of point in grid
		i_theta = (size_t)floor((p.y - originCorner.y) * inv_theta_dl);
		i_phi = (size_t)floor((p.z - originCorner.z) * inv_phi_dl);
		gridIdx = i_theta + grid_n_theta * i_phi;

		// Update the radius in cell
		if (frustrum_radiuses[gridIdx] < 0)
			frustrum_radiuses[gridIdx] = p.x;
		else if (p.x < frustrum_radiuses[gridIdx])
			frustrum_radiuses[gridIdx] = p.x;
	}

	t.push_back(std::clock());

	// Apply margin to free ranges
	float margin = dl;
	float frustrum_alpha = theta_dl / 2;
	for (auto &r : frustrum_radiuses)
	{
		float adapt_margin = r * frustrum_alpha;
		if (margin < adapt_margin)
			r -= adapt_margin;
		else
			r -= margin;
	}

	t.push_back(std::clock());

	////////////////////////////
	// Apply frustrum casting //
	////////////////////////////

	// Update free pixels
	float min_r = 2 * dl;
	size_t p_i = 0;
	Eigen::Matrix3f R_t = R.transpose();
	for (auto &p : cloud.pts)
	{
		// Ignore points updated just now
		if (!not_updated[p_i])
		{
			p_i++;
			continue;
		}

		// Ignore points outside area of the frame
		if (p.x > max_P.x || p.y > max_P.y || p.z > max_P.z || p.x < min_P.x || p.y < min_P.y || p.z < min_P.z)
		{
			p_i++;
			continue;
		}

		// Align point in frame coordinates (and normal)
		PointXYZ xyz(p);
		PointXYZ nxyz(normals[p_i]);
		Eigen::Map<Eigen::Vector3f> p_mat((float *)&xyz, 3, 1);
		Eigen::Map<Eigen::Vector3f> n_mat((float *)&nxyz, 3, 1);
		p_mat = R_t * (p_mat - T);
		n_mat = R_t * n_mat;

		// Project in polar coordinates
		PointXYZ rtp = cart2pol(xyz);

		// Position of point in grid
		i_theta = (size_t)floor((rtp.y - originCorner.y) * inv_theta_dl);
		i_phi = (size_t)floor((rtp.z - originCorner.z) * inv_phi_dl);
		gridIdx = i_theta + grid_n_theta * i_phi;

		// Update movable prob
		if (rtp.x > min_r && rtp.x < frustrum_radiuses[gridIdx])
		{
			// Do not update if normal is horizontal and perpendicular to ray (to avoid removing walls)
			if (abs(nxyz.z) > min_vert_cos)
			{
				movable_counts[p_i] += 1;
				movable_probs[p_i] += 1.0;
			}
			else
			{
				float angle = acos(min(abs(xyz.dot(nxyz) / rtp.x), 1.0f));
				if (angle < max_angle)
				{
					movable_counts[p_i] += 1;
					movable_probs[p_i] += 1.0;
				}
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