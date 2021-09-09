
#include "region_growing.h"



void get_lidar_image(vector<PointXYZ>& rtp,
	vector<int>& image,
	int lidar_n_lines,
	float lidar_angle_res,
	float minTheta)
{

	// Init variables
	float theta0 = minTheta - 0.5 * lidar_angle_res;
	image.reserve(rtp.size() * 2);

	// Follow each scan line varaibles
	vector<int> column_is(lidar_n_lines, -1);
	int i = 0;
	for (auto& p : rtp)
	{
		// Get line index
		int l = (int)floor((p.y - theta0) / lidar_angle_res);

		// Add the new phi column if we advanced.
		if (column_is[l] >= 0)
		{
			image.insert(image.end(), column_is.begin(), column_is.end());
			for (auto& j : column_is)
				j = -1;
		}

		// Add point index to current column
		column_is[l] = i;
		i++;
	}
}





void lidar_plane_growing(vector<PointXYZ>& points,
	vector<PointXYZ>& normals,
	vector<int>& plane_inds,
	vector<Plane3D>& planes,
	int lidar_n_lines,
	RG_params params)
{
	// Initialize variables
	// ********************

	// Number of points
	size_t N = points.size();

	// Result vectors
	plane_inds = vector<int>(N, -1);

	// Polar point cloud
	PointCloud polar_cloud;
	polar_cloud.pts = vector<PointXYZ>(points);


	// Convert points to polar coordinates
	// ***********************************

	// In place modification of the data
	cart2pol_(polar_cloud.pts);

	// Find lidar theta resolution
	float minTheta = 1000;
	float maxTheta = -1000;
	for (size_t i = 1; i < polar_cloud.pts.size() - 1; i++)
	{
		if (polar_cloud.pts[i].y < minTheta)
			minTheta = polar_cloud.pts[i].y;
		if (polar_cloud.pts[i].y > maxTheta)
			maxTheta = polar_cloud.pts[i].y;
	}

	// Get line of scan inds
	float lidar_angle_res = (maxTheta - minTheta) / (lidar_n_lines - 1);


	// Create the lidar depth image
	// ****************************

	// Image does not contain depth but index to the original point
	vector<int> image;
	get_lidar_image(polar_cloud.pts, image, lidar_n_lines, lidar_angle_res, minTheta);


	// Start region growing
	// ********************

	// Init variabels
	float theta0 = minTheta - 0.5 * lidar_angle_res;
	vector<int> unseen_mask(image);
	int region_ind = 0;

	// Random order
	vector<size_t> ordering(image.size());
	iota(ordering.begin(), ordering.end(), 0);
	shuffle(ordering.begin(), ordering.end(), default_random_engine());

	for (auto& i0 : ordering)
	{
		// Ignore empty image pixels
		if (unseen_mask[i0] < 0)
			continue;

		// Init region
		vector<size_t> region;
		vector<PointXYZ> region_points;

		// Init seed queue
		queue<size_t> candidates;
		candidates.push(i0);

		// Init plane
		Plane3D region_plane(points[image[i0]], normals[image[i0]]);
		PointXYZ region_normal = region_plane.u * (1 / (sqrt(region_plane.u.sq_norm()) + 1e-6));
		int last_n = 0;

		// Start growing
		while (candidates.size() > 0)
		{
			// Get corresponding point index and pop candidate
			size_t im_ind = candidates.front();
			size_t pt_ind = (size_t)image[im_ind];
			candidates.pop();

			// Check if candidate is in plane
			float angle_diff = acos(abs(normals[pt_ind].dot(region_normal)));
			float plane_dist = region_plane.point_distance(points[pt_ind]);
			if (angle_diff > params.norm_thresh || plane_dist > params.dist_thresh)
				continue;

			// Here, candidate is accepted, update region
			region.push_back(pt_ind);
			region_points.push_back(points[pt_ind]);
			plane_inds[pt_ind] = region_ind;

			// Remove point from image
			image[im_ind] = -1;

			// Update plane with square fitting if necessary
			if (region_points.size() - last_n > 50)
			{
				region_plane.update(region_points);
				region_normal = region_plane.u * (1 / sqrt(region_plane.u.sq_norm()));
				last_n += 50;
			}

			// Check neigbors for new candidates
			int iTheta0 = (int)floor((polar_cloud.pts[im_ind].y - theta0) / lidar_angle_res);
			for (int dTheta = -1; dTheta < 2; dTheta++)
			{
				if (iTheta0 + dTheta > 0 && iTheta0 + dTheta < lidar_n_lines)
				{
					for (int dPhi = -5; dPhi < 5; dPhi++)
					{
						int candidate_i = im_ind + dTheta + lidar_n_lines * dPhi;

						if (candidate_i > 0 && candidate_i < (int)image.size())
						{
							if (unseen_mask[candidate_i] > 0)
							{
								// Set index as seen and add to candidates
								unseen_mask[candidate_i] = -1;
								candidates.push(candidate_i);
							}
						}
					}
				}
			}
		}

		// Last update of the plane
		region_plane.update(region_points);

		// Fill result containers
		if (region_points.size() > (size_t)params.min_points)
		{
			//cout << " Plane " << region_ind << " accepted with  " << region_points.size() << " => " << region_plane.u << "  " << region_plane.d << endl;
			planes.push_back(region_plane);
			region_ind++;
		}
		else
		{
			//cout << " Plane " << region_ind << " rejected with  " << region_points.size() << " => " << region_plane.u << "  " << region_plane.d << endl;
			for (auto& pt_i : region)
				plane_inds[pt_i] = -1;
		}

		// Update unseen points (to remove candidates that were not added to the plane)
		unseen_mask = image;

		// Stop if we reached enough planes
		if (region_ind >= params.max_planes)
			break;
	}









}


void pointmap_plane_growing(vector<PointXYZ>& points,
	vector<PointXYZ>& normals,
	vector<int>& plane_inds,
	vector<Plane3D>& planes,
	float dl,
	RG_params params)
{
	// Initialize variables
	// ********************

	// Number of points
	size_t N = points.size();

	// Result vectors
	plane_inds = vector<int>(N, -2);


	// Create the sparse grid structure 
	// ********************************

	// Limits of the map
	PointXYZ minCorner = min_point(points);
	PointXYZ maxCorner = max_point(points);
	PointXYZ originCorner = floor(minCorner * (1 / dl) - PointXYZ(1, 1, 1)) * dl;

	// Dimensions of the grid
	size_t sampleNX = (size_t)floor((maxCorner.x - originCorner.x) / dl) + 2;
	size_t sampleNY = (size_t)floor((maxCorner.y - originCorner.y) / dl) + 2;

	// Init structure
	unordered_map<size_t, int> sparse_map;
	sparse_map.reserve(points.size());

	// Fill with point indices
	int i = 0;
	size_t iX, iY, iZ, mapIdx;
	for (auto& p : points)
	{
		// Position of point in sample map
		iX = (size_t)floor((p.x - originCorner.x) / dl);
		iY = (size_t)floor((p.y - originCorner.y) / dl);
		iZ = (size_t)floor((p.z - originCorner.z) / dl);

		// Update the point cell
		mapIdx = iX + sampleNX * iY + sampleNX * sampleNY * iZ;
		if (sparse_map.count(mapIdx) < 1)
			sparse_map.emplace(mapIdx, i);
		else
			sparse_map[mapIdx] = i;
		i++;
	}

	// Start region growing
	// ********************

	// Init variabels
	vector<int> unseen_mask(points.size(), 1);
	int region_ind = 0;
	size_t map_i0;


	// Random order
	vector<size_t> ordering(points.size());
	iota(ordering.begin(), ordering.end(), 0);
	shuffle(ordering.begin(), ordering.end(), default_random_engine());

	for (auto& i0 : ordering)
	{
		// Ignore point already in a plane
		if (plane_inds[i0] > -2)
			continue;

		// Get index in the map
		iX = (size_t)floor((points[i0].x - originCorner.x) / dl);
		iY = (size_t)floor((points[i0].y - originCorner.y) / dl);
		iZ = (size_t)floor((points[i0].z - originCorner.z) / dl);
		map_i0 = iX + sampleNX * iY + sampleNX * sampleNY * iZ;

		// Init region
		vector<size_t> region;
		vector<size_t> rejected_candidates;
		vector<PointXYZ> region_points;

		// Init candidates queue
		queue<size_t> candidates;
		candidates.push(map_i0);

		// Init plane
		Plane3D region_plane(points[i0], normals[i0]);
		PointXYZ region_normal = region_plane.u * (1 / (sqrt(region_plane.u.sq_norm()) + 1e-6));
		int last_n = 0;

		// Start growing
		while (candidates.size() > 0)
		{
			// Get corresponding point index and pop candidate
			size_t map_ind = candidates.front();
			size_t pt_ind = (size_t)sparse_map[map_ind];
			candidates.pop();

			// Check if candidate is in plane
			float angle_diff = acos(abs(normals[pt_ind].dot(region_normal)));
			float plane_dist = region_plane.point_distance(points[pt_ind]);
			if (angle_diff > params.norm_thresh || plane_dist > params.dist_thresh)
			{
				rejected_candidates.push_back(pt_ind);
				continue;
			}

			// Here, candidate is accepted, update region
			region.push_back(pt_ind);
			region_points.push_back(points[pt_ind]);
			plane_inds[pt_ind] = region_ind;

			// Update plane with square fitting if necessary
			if (region_points.size() - last_n > 50)
			{
				region_plane.update(region_points);
				region_normal = region_plane.u * (1 / sqrt(region_plane.u.sq_norm()));
				last_n += 50;
			}

			// Check neigbors for new candidates

			for (int dx = -1; dx < 2; dx++)
			{

				for (int dy = -1; dy < 2; dy++)
				{

					for (int dz = -1; dz < 2; dz++)
					{
						// Find distance to cell center
						int candidate_i = (int)map_ind + dx + (int)sampleNX * dy + (int)(sampleNX * sampleNY) * dz;

						if (candidate_i > 0 && sparse_map.count((size_t)candidate_i) > 0)
						{
							if (unseen_mask[sparse_map[(size_t)candidate_i]] > 0)
							{
								// Set index as seen and add to candidates
								unseen_mask[sparse_map[(size_t)candidate_i]] = -1;
								candidates.push(candidate_i);
							}
						}
					}
				}
			}
		}

		// Fill result containers
		if (region_points.size() > (size_t)params.min_points)
		{
			//cout << " Plane " << region_ind << " accepted with  " << region_points.size() << " => " << region_plane.u << "  " << region_plane.d << endl;
			region_plane.update(region_points);
			planes.push_back(region_plane);
			region_ind++;
		}
		else
		{
			//cout << " Plane " << region_ind << " rejected with  " << region_points.size() << " => " << region_plane.u << "  " << region_plane.d << endl;
			for (auto& pt_i : region)
				plane_inds[pt_i] = -1;
		}

		// Update unseen points. Make rejected candiadte unseen again
		for (auto& pi : rejected_candidates)
			unseen_mask[pi] = 1;

		// Stop if we reached enough planes
		if (region_ind >= params.max_planes)
			break;
	}

}
































