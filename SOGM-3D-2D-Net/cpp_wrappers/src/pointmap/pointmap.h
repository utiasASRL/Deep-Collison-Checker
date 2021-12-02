#pragma once


#include <cmath>
#include <algorithm>
#include <unordered_set>

#include "../grid_subsampling/grid_subsampling.h"
#include "../polar_processing/polar_processing.h"

using namespace std;

// KDTree type definition
typedef nanoflann::KDTreeSingleIndexAdaptorParams KDTree_Params;
typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, 3> PointXYZ_KDTree;
typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, 3> PointXYZ_Dynamic_KDTree;


Eigen::Matrix4d pose_interp(float t, Eigen::Matrix4d const& H1, Eigen::Matrix4d const& H2, int verbose);

//-------------------------------------------------------------------------------------------
//
// PointMapPython Class
// ********************
//
//	PointMap designed to be used in python. As it is hard to transfert unordered map to
//	python dict structure, we rebuild the hashmap every update (not very efficient).
//
//-------------------------------------------------------------------------------------------

class MapVoxelData
{
public:
	// Elements
	// ********

	bool occupied;
	int count;
	PointXYZ centroid;
	PointXYZ normal;
	float score;

	// Methods
	// *******

	// Constructor
	MapVoxelData()
	{
		occupied = false;
		count = 0;
		score = -1.0f;
		centroid = PointXYZ();
		normal = PointXYZ();
	}
	MapVoxelData(const PointXYZ p0, const PointXYZ n0, const float s0, const int c0)
	{
		occupied = true;
		count = c0;
		score = s0;
		centroid = p0;
		normal = n0;
	}

	MapVoxelData(const PointXYZ p0)
	{
		// We initiate only the centroid
		count = 1;
		centroid = p0;

		// Other varaible are kept null
		occupied = false;
		score = -1.0f;
		normal = PointXYZ();
	}

	void update_centroid(const PointXYZ p0)
	{
		count += 1;
		centroid += p0;
	}

	void update_normal(const float s0, const PointXYZ n0)
	{
		// We keep the worst normal
		occupied = true;

		// Rule for normal update:
		// IF current_score=2 : normal was computed with planarity in the map, do not modify
		// IF s0 < score - 0.1 : Too bad score dont update (This includes the condition above)
		// IF s0 > score + 0.1 : Better score, use new normal
		// IF abs(s0 - score) < 0.1 : Similar score, avergae normal
		// When averaging be careful of orientation. Dont worry about norm, we renormalize every normal in the end

		if (s0 > score + 0.1)
		{
			score = s0;
			normal = n0;
		}
		else if (s0 > score - 0.1)
		{
			if (s0 > score)
				score = s0;
			if (normal.dot(n0) > 0)
				normal += n0;
			else
				normal -= n0;
		}
	}
};

class PointMapPython
{
public:
	// Elements
	// ********

	float dl;
	vector<PointXYZ> points;
	vector<PointXYZ> normals;
	vector<float> scores;
	vector<int> counts;

	// Methods
	// *******

	// Constructor
	PointMapPython()
	{
		dl = 1.0f;
	}
	PointMapPython(const float dl0)
	{
		dl = dl0;
	}

	// Methods
	void update(vector<PointXYZ> &points0, vector<PointXYZ> &normals0, vector<float> &scores0);

	void init_samples(const PointXYZ originCorner,
					  const PointXYZ maxCorner,
					  unordered_map<size_t, MapVoxelData> &samples);

	void add_samples(const vector<PointXYZ> &points0,
					 const vector<PointXYZ> &normals0,
					 const vector<float> &scores0,
					 const PointXYZ originCorner,
					 const PointXYZ maxCorner,
					 unordered_map<size_t, MapVoxelData> &samples);

	size_t size() { return points.size(); }
};

//-------------------------------------------------------------------------------------------
//
// PointMap Class
// **************
//
//	PointMap designed to be used in C++. Everything should be more efficient here.
//
//-------------------------------------------------------------------------------------------

class PointMapOld
{
public:
	// Elements
	// ********

	// Voxel size
	float dl;

	// Containers for the data
	vector<PointXYZ> points;
	vector<PointXYZ> normals;
	vector<float> scores;

	// Containers for the data
	vector<PointXYZ> voxPoints;
	vector<PointXYZ> voxNormals;
	vector<float> voxScores;
	vector<int> voxCounts;

	// Sparse hashmap that contain voxels (each voxel data is in the contiguous vector containers)
	unordered_map<VoxKey, size_t> samples;

	// Methods
	// *******

	// Constructor
	PointMapOld()
	{
		dl = 1.0f;
	}
	PointMapOld(const float dl0)
	{
		dl = dl0;
	}
	PointMapOld(const float dl0,
				vector<PointXYZ> &init_points,
				vector<PointXYZ> &init_normals,
				vector<float> &init_scores)
	{
		dl = dl0;
		update(init_points, init_normals, init_scores);
	}

	// Size of the map (number of point/voxel in the map)
	size_t size() { return voxPoints.size(); }

	// Init of voxel centroid
	void init_sample_centroid(const VoxKey &k, const PointXYZ &p0)
	{
		// We place anew key in the hashmap
		samples.emplace(k, voxPoints.size());

		// We add new voxel data but initiate only the centroid
		voxPoints.push_back(p0);
		voxCounts.push_back(1);
		voxNormals.push_back(PointXYZ());
		voxScores.push_back(-1.0f);
	}

	// Update of voxel centroid
	void update_sample_centroid(const VoxKey &k, const PointXYZ &p0)
	{
		// Update count of points and centroid of the cell
		voxCounts[samples[k]] += 1;
		voxPoints[samples[k]] += p0;
	}

	// Update of voxel normal
	void update_sample_normal(PointXYZ &normal, float &score, const PointXYZ &n0, const float &s0)
	{
		// Rule for normal update:
		// IF current_score=2 : normal was computed with planarity in the map, do not modify
		// IF s0 < score - 0.1 : Too bad score dont update (This includes the condition above)
		// IF s0 > score + 0.1 : Better score, use new normal
		// IF abs(s0 - score) < 0.1 : Similar score, avergae normal
		// When averaging be careful of orientation. Dont worry about norm, we renormalize every normal in the end

		if (s0 > score + 0.1)
		{
			score = s0;
			normal = n0;
		}
		else if (s0 > score - 0.1)
		{
			if (s0 > score)
				score = s0;
			if (normal.dot(n0) > 0)
				normal += n0;
			else
				normal -= n0;
		}
	}

	// Update map with a set of new points
	void update(vector<PointXYZ> &points0, vector<PointXYZ> &normals0, vector<float> &scores0)
	{

		// Reserve new space if needed
		if (samples.size() < 1)
			samples.reserve(10 * points0.size());

		std::cout << std::endl
				  << "--------------------------------------" << std::endl;
		std::cout << "current max_load_factor: " << samples.max_load_factor() << std::endl;
		std::cout << "current size: " << samples.size() << std::endl;
		std::cout << "current bucket_count: " << samples.bucket_count() << std::endl;
		std::cout << "current load_factor: " << samples.load_factor() << std::endl;
		std::cout << "--------------------------------------" << std::endl
				  << std::endl;

		// Initialize variables
		float r = 1.5;
		float r2 = r * r;
		float inv_dl = 1 / dl;
		size_t i = 0;
		VoxKey k0, k;

		for (auto &p : points0)
		{
			// Position of point in sample map
			PointXYZ p_pos = p * inv_dl;

			// Corresponding key
			k0.x = (int)floor(p_pos.x);
			k0.y = (int)floor(p_pos.y);
			k0.z = (int)floor(p_pos.z);

			// Update the adjacent cells
			for (k.x = k0.x - 1; k.x < k0.x + 2; k.x++)
			{

				for (k.y = k0.y - 1; k.y < k0.y + 2; k.y++)
				{

					for (k.z = k0.z - 1; k.z < k0.z + 2; k.z++)
					{
						// Center of updated cell in grid coordinates
						PointXYZ cellCenter(k.x + 0.5, k.y + 0.5, k.z + 0.5);

						// Update barycenter if in range
						float d2 = (cellCenter - p_pos).sq_norm();
						if (d2 < r2)
						{
							if (samples.count(k) < 1)
								init_sample_centroid(k, p);
							else
								update_sample_centroid(k, p);
						}
					}
				}
			}

			// Update the point normal
			update_sample_normal(voxNormals[samples[k0]], voxScores[samples[k0]], normals0[i], scores0[i]);
			i++;
		}

		// Now update vector containers only with voxel whose centroid is in their voxel
		points.reserve(samples.size());
		normals.reserve(samples.size());
		scores.reserve(samples.size());
		i = 0;
		for (auto &v : samples)
		{
			// Check if centroid is in cell
			PointXYZ centroid = voxPoints[v.second] * (1.0 / voxCounts[v.second]);
			PointXYZ centroid_pos = centroid * inv_dl;
			k0.x = (int)floor(centroid_pos.x);
			k0.y = (int)floor(centroid_pos.y);
			k0.z = (int)floor(centroid_pos.z);

			float score = voxScores[v.second];
			if (score > -1.5 && k0 == v.first)
			{
				PointXYZ normal = voxNormals[v.second];
				normal *= 1.0 / (sqrt(normal.sq_norm()) + 1e-6);
				if (i < points.size())
				{
					points[i] = centroid;
					normals[i] = normal;
					scores[i] = score;
				}
				else
				{
					points.push_back(centroid);
					normals.push_back(normal);
					scores.push_back(score);
				}
				i++;
			}
		}
	}
};

class PointMap
{
public:
	// Elements
	// ********

	// Voxel size
	float dl;

	// Count the number of frames used tu update this map
	int update_idx;

	// Containers for the data
	PointCloud cloud;
	vector<PointXYZ> normals;
	vector<float> scores;
	vector<int> counts;
	vector<int> latest;

	// Container only used when ray tracing
	vector<VoxKey> cloud_keys;

	// Sparse hashmap that contain voxels (each voxel data is in the contiguous vector containers)
	unordered_map<VoxKey, size_t> samples;

	// KDTree for neighbors query
	PointXYZ_Dynamic_KDTree tree;

	// Methods
	// *******

	// Constructor
	PointMap() : tree(3, cloud, KDTree_Params(10 /* max leaf */))
	{
		dl = 1.0f;
		update_idx = 0;
	}
	PointMap(const float dl0) : tree(3, cloud, KDTree_Params(10 /* max leaf */))
	{
		dl = dl0;
		update_idx = 0;
	}
	PointMap(const float dl0,
			 vector<PointXYZ> &init_points,
			 vector<PointXYZ> &init_normals,
			 vector<float> &init_scores) : tree(3, cloud, KDTree_Params(10 /* max leaf */))
	{
		dl = dl0;
		update_idx = -1;
		update(init_points, init_normals, init_scores, -1);
	}

	PointMap(const PointMap &map0) : tree(3, cloud, KDTree_Params(10 /* max leaf */))
	{
		dl = map0.dl;
		update_idx = map0.update_idx;
		cloud = map0.cloud;
		normals = map0.normals;
		scores = map0.scores;
		counts = map0.counts;
		latest = map0.latest;
		samples = map0.samples;
		tree.addPoints(0, cloud.pts.size() - 1);
	}

	PointMap(const PointMap &map0,
			 const size_t max_ind) : tree(3, cloud, KDTree_Params(10 /* max leaf */))
	{
		dl = map0.dl;
		update_idx = max_ind;
		copy_until(map0, max_ind);
	}

	PointMap& operator=(const PointMap &map0)
	{
		dl = map0.dl;
		update_idx = map0.update_idx;
		cloud = map0.cloud;
		normals = map0.normals;
		scores = map0.scores;
		counts = map0.counts;
		latest = map0.latest;
		samples = map0.samples;
		tree.addPoints(0, cloud.pts.size() - 1);
      	return *this;
	}

	// Size of the map (number of point/voxel in the map)
	size_t size() { return cloud.pts.size(); }

	// Init of voxel centroid
	void init_sample(const VoxKey &k, const PointXYZ &p0, const PointXYZ &n0, const float &s0, const int &c0)
	{
		// We place anew key in the hashmap
		samples.emplace(k, cloud.pts.size());

		// We add new voxel data but initiate only the centroid
		cloud.pts.push_back(p0);
		normals.push_back(n0);
		scores.push_back(s0);

		// Count is useless, instead save index of first frame placing a point in this cell
		counts.push_back(c0);
		latest.push_back(c0);
	}

	// Update of voxel centroid
	void update_sample(const size_t idx, const PointXYZ &p0, const PointXYZ &n0, const float &s0, const int &c0)
	{
		// latest frame idx
		latest[idx] = c0;

		// Update normal if we have a clear view of it  and closer distance (see computation of score)
		if (s0 > scores[idx])
		{
			scores[idx] = s0;
			normals[idx] = n0;
		}
	}

	// Update map with a set of new points
	void update(vector<PointXYZ> &points0, vector<PointXYZ> &normals0, vector<float> &scores0, int ind0)
	{

		// Reserve new space if needed
		if (samples.size() < 1)
			samples.reserve(10 * points0.size());
		if (cloud.pts.capacity() < cloud.pts.size() + points0.size())
		{
			cloud.pts.reserve(cloud.pts.capacity() + points0.size());
			counts.reserve(counts.capacity() + points0.size());
			latest.reserve(latest.capacity() + points0.size());
			normals.reserve(normals.capacity() + points0.size());
			scores.reserve(scores.capacity() + points0.size());
		}

		//std::cout << std::endl << "--------------------------------------" << std::endl;
		//std::cout << "current max_load_factor: " << samples.max_load_factor() << std::endl;
		//std::cout << "current size: " << samples.size() << std::endl;
		//std::cout << "current bucket_count: " << samples.bucket_count() << std::endl;
		//std::cout << "current load_factor: " << samples.load_factor() << std::endl;
		//std::cout << "--------------------------------------" << std::endl << std::endl;

		// Initialize variables
		float inv_dl = 1 / dl;
		size_t i = 0;
		VoxKey k0;
		size_t num_added = 0;

		for (auto &p : points0)
		{
			// Position of point in sample map
			PointXYZ p_pos = p * inv_dl;

			// Corresponding key
			k0.x = (int)floor(p_pos.x);
			k0.y = (int)floor(p_pos.y);
			k0.z = (int)floor(p_pos.z);

			// Update the point count
			if (samples.count(k0) < 1)
			{
				init_sample(k0, p, normals0[i], scores0[i], ind0);
				num_added++;
			}
			else
			{
				update_sample(samples[k0], p, normals0[i], scores0[i], ind0);
			}
			i++;
		}

		// Update tree
		tree.addPoints(cloud.pts.size() - num_added, cloud.pts.size() - 1);

		// Update frame count
		update_idx++;
	}

	// Update map with a set of new points
	void copy_until(const PointMap &map0, const size_t max_ind)
	{
		// Reserve new space if needed
		if (samples.size() < 1)
			samples.reserve(10 * map0.cloud.pts.size());
		if (cloud.pts.capacity() < cloud.pts.size() + map0.cloud.pts.size())
		{
			cloud.pts.reserve(cloud.pts.capacity() + map0.cloud.pts.size());
			counts.reserve(counts.capacity() + map0.cloud.pts.size());
			latest.reserve(latest.capacity() + map0.cloud.pts.size());
			normals.reserve(normals.capacity() + map0.cloud.pts.size());
			scores.reserve(scores.capacity() + map0.cloud.pts.size());
		}

		// Initialize variables
		float inv_dl = 1 / dl;
		size_t i = 0;
		VoxKey k0;
		size_t num_added = 0;

		for (auto &p : map0.cloud.pts)
		{
			// Stop when reaching the current index
			if (map0.counts[i] > max_ind)
				break;

			// Position of point in sample map
			PointXYZ p_pos = p * inv_dl;

			// Corresponding key
			k0.x = (int)floor(p_pos.x);
			k0.y = (int)floor(p_pos.y);
			k0.z = (int)floor(p_pos.z);

			// check angle

			// Update the point count
			if (samples.count(k0) < 1)
			{
				init_sample(k0, p, map0.normals[i], map0.scores[i], map0.counts[i]);
				latest[samples[k0]] = map0.latest[i];
				num_added++;
			}
			i++;
		}	

		// Update tree
		tree.addPoints(cloud.pts.size() - num_added, cloud.pts.size() - 1);

		// Update frame count
		update_idx++;
	}

	// Compute movable probabilities

	void update_movable_pts(vector<PointXYZ> &frame_points,
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
							vector<int> &movable_counts);
};


class LaserRange2D
{

public:
	// Elements
	// ********

	// Angle resolution
	float angle_res;

	// Number of range angles
	size_t n_angles;

	// Maximum value of the counts
	int max_count;

	// Container for the ranges
	vector<float> range_table;

	// Methods
	// *******

	// Constructor
	LaserRange2D()
	{
		angle_res = 0.5 * M_PI / 180.0;
		n_angles = (size_t)floor(2.0 * M_PI / angle_res) + 1;
		range_table = vector<float>(n_angles, -1.0);
	}

	LaserRange2D(const float angle_res0)
	{
		angle_res = angle_res0;
		n_angles = (size_t)floor(2.0 * M_PI / angle_res) + 1;
		range_table = vector<float>(n_angles, -1.0);
	}

	LaserRange2D(const size_t n_angles0)
	{
		n_angles = n_angles0;
		angle_res = 2.0 * M_PI / (float)n_angles;
		range_table = vector<float>(n_angles, -1.0);
	}

	void compute_from_3D(vector<PointXYZ> &points3D, PointXYZ &center3D, Plane3D &ground_P, float zMin, float zMax)
	{
		//////////
		// Init //
		//////////
		//
		//	This function assumes the point cloud has been aligned in the world coordinates and the ground is close to a flat plane
		//

		// Get distances to ground
		vector<float> distances;
		ground_P.point_distances(points3D, distances);

		///////////////////
		// Update ranges //
		///////////////////

		// Loop over 3D points
		size_t p_i = 0;
		for (auto &p : points3D)
		{
			// Check height limits
			if (distances[p_i] < zMin || distances[p_i] > zMax)
			{
				p_i++;
				continue;
			}

			// Add the angle and its corresponding free_range
			PointXY diff2D(p - center3D);
			size_t angle_idx = (size_t)floor((atan2(diff2D.y, diff2D.x) + M_PI) / angle_res);
			float d2 = diff2D.sq_norm();
			if (range_table[angle_idx] < 0 || d2 < pow(range_table[angle_idx], 2))
				range_table[angle_idx] = sqrt(d2);

			p_i++;
		}

		///////////////////////////////
		// Interpolate the 2D ranges //
		///////////////////////////////

		// First find the last valid value
		int last_i, next_i;
		last_i = range_table.size() - 1;
		while (last_i >= 0)
		{
			if (range_table[last_i] > 0)
				break;
			last_i--;
		}

		// Interpolate
		next_i = 0;
		last_i -= range_table.size();
		while (next_i < range_table.size())
		{
			if (range_table[next_i] > 0)
			{
				if (last_i < 0)
				{
					int diff = next_i - last_i;
					if (diff > 1)
					{
						for (int i = last_i + 1; i < next_i; i++)
						{
							int real_i = i;
							if (real_i < 0)
								real_i += range_table.size();
							float t = (i - last_i) / diff;
							int real_last_i = last_i + range_table.size();
							range_table[real_i] = t * range_table[real_last_i] + (1 - t) * range_table[next_i];
						}
					}
				}
				else
				{
					int diff = next_i - last_i;
					if (diff > 1)
					{
						for (int i = last_i + 1; i < next_i; i++)
						{
							float t = (i - last_i) / diff;
							range_table[i] = t * range_table[last_i] + (1 - t) * range_table[next_i];
						}
					}
				}
				last_i = next_i;
			}
			next_i++;
		}
	}

	void get_costmap(vector<PointXYZ> &points3D, PointXYZ &center3D, Plane3D &ground_P, float zMin, float zMax, float dl_2D)
	{
		//////////
		// Init //
		//////////
		//
		//	This function assumes the point cloud has been aligned in the world coordinates and the ground is close to a flat plane
		//

		// Get distances to ground
		vector<float> distances;
		ground_P.point_distances(points3D, distances);

		// Create costmap
		vector<float> costmap;

		///////////////////
		// Update ranges //
		///////////////////

		// Loop over 3D points
		size_t p_i = 0;
		for (auto &p : points3D)
		{
			// Check height limits
			if (distances[p_i] < zMin || distances[p_i] > zMax)
			{
				p_i++;
				continue;
			}

			// Add the angle and its corresponding free_range
			PointXY diff2D(p - center3D);
			size_t angle_idx = (size_t)floor((atan2(diff2D.y, diff2D.x) + M_PI) / angle_res);
			float d2 = diff2D.sq_norm();
			if (range_table[angle_idx] < 0 || d2 < pow(range_table[angle_idx], 2))
				range_table[angle_idx] = sqrt(d2);

			p_i++;
		}

		///////////////////////////////
		// Interpolate the 2D ranges //
		///////////////////////////////

		// First find the last valid value
		int last_i, next_i;
		last_i = range_table.size() - 1;
		while (last_i >= 0)
		{
			if (range_table[last_i] > 0)
				break;
			last_i--;
		}

		// Interpolate
		next_i = 0;
		last_i -= range_table.size();
		while (next_i < range_table.size())
		{
			if (range_table[next_i] > 0)
			{
				if (last_i < 0)
				{
					int diff = next_i - last_i;
					if (diff > 1)
					{
						for (int i = last_i + 1; i < next_i; i++)
						{
							int real_i = i;
							if (real_i < 0)
								real_i += range_table.size();
							float t = (i - last_i) / diff;
							int real_last_i = last_i + range_table.size();
							range_table[real_i] = t * range_table[real_last_i] + (1 - t) * range_table[next_i];
						}
					}
				}
				else
				{
					int diff = next_i - last_i;
					if (diff > 1)
					{
						for (int i = last_i + 1; i < next_i; i++)
						{
							float t = (i - last_i) / diff;
							range_table[i] = t * range_table[last_i] + (1 - t) * range_table[next_i];
						}
					}
				}
				last_i = next_i;
			}
			next_i++;
		}
	}
};

class FullGrid2D
{
public:
	// Elements
	// ********

	// Voxel size
	float dl;
	vector<float> scores;
	vector<int> counts;
	PointXYZ originCorner;
	size_t sampleNX;
	size_t sampleNY;

	// Methods
	// *******

	// Constructor
	FullGrid2D()
	{
		dl = 1.0f;
	}
	FullGrid2D(const float dl0)
	{
		dl = dl0;
	}

	// Size of the map (number of point/pixel in the map)
	size_t size() { return scores.size(); }

	// Update map with a set of new points
	void update_from_3D(vector<PointXYZ> &points3D, PointXYZ &center3D, Plane3D &ground_P, float zMin, float zMax, float radius)
	{

		// Angle resolution
		float angle_res = 0.5 * M_PI / 180.0;
		float inv_angle_res = 1 / angle_res;
		size_t n_angles = (size_t)floor(2.0 * M_PI / angle_res) + 1;
		vector<size_t> angle_count(n_angles, 0);

		// Initialize variables
		float inv_dl = 1 / dl;
		size_t iX, iY, iZ, mapIdx;

		// Limits of the map
		PointXYZ minCorner = min_point(points3D);
		PointXYZ maxCorner = max_point(points3D);
		originCorner = floor(minCorner * inv_dl) * dl;

		// Dimensions of the grid
		sampleNX = (size_t)floor((maxCorner.x - originCorner.x) * inv_dl) + 1;
		sampleNY = (size_t)floor((maxCorner.y - originCorner.y) * inv_dl) + 1;

		// Init containers
		scores = vector<float>(sampleNX * sampleNY);
		counts = vector<int>(sampleNX * sampleNY);

		// Get distances to ground
		vector<float> distances;
		ground_P.point_distances(points3D, distances);

		// Get angle idx and dist for each point
		vector<size_t> angle_inds;
		vector<float> angle_d2s;
		angle_inds.reserve(points3D.size());
		angle_d2s.reserve(points3D.size());
		size_t p_i = 0;
		for (auto &p : points3D)
		{
			// Check height limits
			if (distances[p_i] < zMin || distances[p_i] > zMax)
			{
				p_i++;
				continue;
			}

			// Add the angle and its corresponding free_range
			PointXY diff2D(p - center3D);
			size_t angle_idx = (size_t)floor((atan2(diff2D.y, diff2D.x) + M_PI) * inv_angle_res);
			float d2 = diff2D.sq_norm();

			// Add angle idx and dist
			angle_inds.push_back(angle_idx);
			angle_d2s.push_back(d2);
			angle_count[angle_idx]++;
		}

		// Argsort angle dist with their inds
		vector<size_t> order(angle_inds.size());
		iota(order.begin(), order.end(), 0);
		stable_sort(order.begin(), order.end(), [&angle_inds](size_t i1, size_t i2)
					{ return angle_inds[i1] < angle_inds[i2]; });

		// Order angles
		vector<float> angle_d2s_ordered;
		angle_d2s_ordered.reserve(angle_d2s.size());
		vector<size_t> angle_starts;
		int current_angle_idx = -1;
		size_t c_i = 0;
		for (auto &i0 : order)
		{
			while ((int)angle_inds[i0] > current_angle_idx)
			{
				current_angle_idx++;
				angle_starts.push_back(c_i);
			}

			angle_d2s_ordered.push_back(angle_d2s[i0]);
			c_i++;
		}

		// Loop over grid pixels
		for (size_t iX = 0; iX < sampleNX; iX++)
		{
			for (size_t iY = 0; iY < sampleNY; iY++)
			{
				// Idx in the map containers
				mapIdx = iX + sampleNX * iY;

				// Position of pixel center in real world
				PointXYZ p_pixel;
				p_pixel.x = originCorner.x + ((float)iX + 0.5) * dl;
				p_pixel.y = originCorner.y + ((float)iY + 0.5) * dl;

				// Add the angle and its corresponding free_range
				PointXY diff2D(p_pixel - center3D);
				size_t angle_idx = (size_t)floor((atan2(diff2D.y, diff2D.x) + M_PI) * inv_angle_res);
				float d2 = diff2D.sq_norm();

				// Get score
				for (size_t a_i = angle_starts[angle_idx]; a_i < angle_starts[angle_idx] + angle_count[angle_idx]; a_i++)
				{
					if (angle_d2s_ordered[a_i] > d2)
						scores[mapIdx] += (1.0 - scores[mapIdx]) / ++counts[mapIdx];
					else
						scores[mapIdx] += (0.0 - scores[mapIdx]) / ++counts[mapIdx];
				}
			}
		}

		return;
	}
};

void get_ray_keys(PointXYZ &A,
				  PointXYZ &B,
				  float tile_dl,
				  float inv_tile_dl,
				  vector<VoxKey> &ray_keys);
