#pragma once

#include <set>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <queue>
#include <numeric>
#include <random>

#define _USE_MATH_DEFINES
#include <math.h>

#include <Eigen/Eigenvalues>
#include "../cloud/cloud.h"
#include "../polar_processing/polar_processing.h"
#include "../pointmap/pointmap.h"

using namespace std;

// ICP params and result classes
// *****************************

class Plane3D
{
public:

	// Elements
	// ********

	// The plane is define by the equation a*x + b*y + c*z = d. The values (a, b, c) are stored in a PointXYZ called u.
	PointXYZ u;
	float d;


	// Methods
	// *******

	// Constructor
	Plane3D() { u.x = 1; u.y = 0; u.z = 0; d = 0; }
	Plane3D(const float a0, const float b0, const float c0, const float d0) { u.x = a0; u.y = b0; u.z = c0; d = d0; }
	Plane3D(const PointXYZ P0, const PointXYZ N0)
	{
		// Init with point and normal
		u = N0;
		d = N0.dot(P0);
	}
	Plane3D(const PointXYZ A, const PointXYZ B, const PointXYZ C)
	{
		// Init with three points
		u = (B - A).cross(C - A);
		d = u.dot(A);
	}

	// Method getting distance to one point
	float point_distance(const PointXYZ P)
	{
		return abs((u.dot(P) - d) / sqrt(u.sq_norm()));
	}

	// Method getting square distance to one point
	float point_sq_dist(const PointXYZ P)
	{
		float tmp = u.dot(P) - d;
		return tmp * tmp / u.sq_norm();
	}

	// Method getting distances to some points
	void point_distances(vector<PointXYZ>& points, vector<float>& distances)
	{
		size_t i = 0;
		float inv_norm_u = 1 / sqrt(u.sq_norm());
		for (auto& p : points)
		{
			distances[i] = abs((u.dot(p) - d) * inv_norm_u);
			i++;
		}
	}

	// Method updating the plane by least square fitting
	void update(vector<PointXYZ>& points)
	{
		// Least square optimal plane :
		// ****************************

		// Instead of solving the least square problem (which is has singularities) We use PCA
		//
		// The best plane always intersect the points centroid, and then its normal is the third eigenvector

		// Safe check
		if (points.size() < 4)
			return;

		// Compute PCA
		PointXYZ mean = accumulate(points.begin(), points.end(), PointXYZ());
		mean = mean * (1.0 / points.size());

		// Create centralized data
		vector<PointXYZ> points_c(points);
		for (auto& p : points_c)
			p -= mean;

		// Create a N by 3 matrix containing the points (same data in memory)
		Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> X_c((float*)points_c.data(), 3, points_c.size());

		// Compute covariance matrix
		Eigen::Matrix3f cov(X_c * X_c.transpose() / points.size());

		// Compute pca
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es;
		es.compute(cov);

		// Convert back to std containers
		vector<float> eigenvalues(es.eigenvalues().data(), es.eigenvalues().data() + es.eigenvalues().size());
		vector<PointXYZ> eigenvectors((PointXYZ*)es.eigenvectors().data(), (PointXYZ*)es.eigenvectors().data() + es.eigenvectors().rows());

		// Define plane with point and normal
		u = eigenvectors[0];
		d = u.dot(mean);
	}

};


class RG_params
{
public:

	// Elements
	// ********

	// Threshold on normal variation (angle in radian)
	float norm_thresh;

	// Threshold on point to plane distance
	float dist_thresh;

	// Min number of points to keep a plane
	int min_points;

	// Maximum number of plane kept
	int max_planes;

	// Methods
	// *******

	// Constructor
	RG_params()
	{
		norm_thresh = 0.1;
		dist_thresh = 0.1;
		min_points = 500;
		max_planes = 50;
	}
};

void get_lidar_image(vector<PointXYZ>& rtp,
	vector<int>& image,
	int lidar_n_lines,
	float lidar_angle_res,
	float minTheta);

void lidar_plane_growing(vector<PointXYZ>& points,
	vector<PointXYZ>& normals,
	vector<int>& plane_inds,
	vector<Plane3D>& planes,
	int lidar_n_lines,
	RG_params params);

void pointmap_plane_growing(vector<PointXYZ>& points,
	vector<PointXYZ>& normals,
	vector<int>& plane_inds,
	vector<Plane3D>& planes,
	float dl,
	RG_params params);


