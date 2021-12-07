//
//
//		0==========================0
//		|    Local feature test    |
//		0==========================0
//
//		version 1.0 : 
//			> 
//
//---------------------------------------------------
//
//		Cloud source :
//		Define usefull Functions/Methods
//
//----------------------------------------------------
//
//		Hugues THOMAS - 10/02/2017
//


#include "cloud.h"


// Filter functions
// ****************

void filter_pointcloud(std::vector<PointXYZ>& pts, std::vector<float>& scores, float filter_value)
{
	// Remove every points whose score is < filter_value
	auto pts_address = pts.data();
	pts.erase(std::remove_if(pts.begin(), pts.end(),
		[&scores, pts_address, filter_value](const PointXYZ& p) { return scores[(size_t)(&p - pts_address)] < filter_value; }),
		pts.end());
}

void filter_floatvector(std::vector<float>& vec, std::vector<float>& scores, float filter_value)
{
	// Remove every element whose score is < filter_value
	auto vec_address = vec.data();
	vec.erase(std::remove_if(vec.begin(), vec.end(),
		[&scores, vec_address, filter_value](const float& f) { return scores[(size_t)(&f - vec_address)] < filter_value; }),
		vec.end());
}

void filter_floatvector(std::vector<float>& vec, float filter_value)
{
	vec.erase(std::remove_if(vec.begin(), vec.end(), [filter_value](const float s) { return s < filter_value; }), vec.end());
}


// Debug functions
// ***************

void save_cloud(std::string dataPath, 
	std::vector<PointXYZ>& points, 
	std::vector<PointXYZ>& normals, 
	std::vector<float>& features)
{
	// Variables
	uint64_t num_points = points.size();
	uint64_t num_normals = normals.size();
	uint64_t num_features = features.size() / num_points;

	// Safe check
	if (num_features * num_points != features.size())
	{
		std::cout << "Warning: features dimension do not match point cloud" << std::endl;
		std::cout << "         ply saving canceled" << std::endl;
		return;
	}
	if (num_normals != num_points && num_normals != 0)
	{
		std::cout << "Warning: normal dimension do not match point cloud" << std::endl;
		std::cout << "         ply saving canceled" << std::endl;
		return;
	}

	// Open file
	npm::PLYFileOut file(dataPath);

	// Push fields
	file.pushField(num_points, 3, npm::PLY_FLOAT, { "x", "y", "z" }, points);
	if (num_normals > 0)
		file.pushField(num_points, 3, npm::PLY_FLOAT, { "nx", "ny", "nz" }, normals);

	std::vector<std::vector<float>> fields(num_features);
	for (size_t i = 0; i < num_features; i++)
	{
		char buffer[100];
		sprintf(buffer, "f%d", (int)i);
		fields[i] = std::vector<float>(features.begin() + i * num_points, features.begin() + (i + 1) * num_points);
		file.pushField(num_points, 1, npm::PLY_FLOAT, { std::string(buffer) }, fields[i]);
	}
	file.write();
}

void save_cloud(std::string dataPath,
	std::vector<PointXYZ>& points,
	std::vector<PointXYZ>& normals)
{
	std::vector<float> no_f;
	save_cloud(dataPath, points, normals, no_f);
}

void save_cloud(std::string dataPath,
	std::vector<PointXYZ>& points,
	std::vector<float>& features)
{
	std::vector<PointXYZ> no_norms;
	save_cloud(dataPath, points, no_norms, features);
}

void save_cloud(std::string dataPath,
	std::vector<PointXYZ>& points)
{
	std::vector<float> no_f;
	save_cloud(dataPath, points, no_f);
}













void load_cloud(std::string& dataPath,
	std::vector<PointXYZ>& points)
{
	std::vector<float> float_scalar;
	std::string float_scalar_name = "";
	std::vector<int> int_scalar;
	std::string int_scalar_name = "";

	load_cloud(dataPath, points, float_scalar, float_scalar_name, int_scalar, int_scalar_name);

}


void load_cloud(std::string& dataPath, 
	std::vector<PointXYZ>& points, 
	std::vector<float>& float_scalar, 
	std::string& float_scalar_name,
	std::vector<int>& int_scalar,
	std::string& int_scalar_name)
{
	// Variables 
	uint64_t num_points(0);
	std::vector<npm::PLYType> types;
	std::vector<std::string> properties;

	size_t float_str_n = strlen(float_scalar_name.c_str());
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
	points.reserve(num_points);
	float_scalar.reserve(num_points);
	int_scalar.reserve(num_points);

	// Get the points
	for (size_t i = 0; i < properties.size(); i++)
	{
		if (properties[i].size() == 1 && strncmp(properties[i].c_str(), "x", 1) == 0)
			file.getField(i, 3, points);

		if (properties[i].size() == float_str_n && strncmp(properties[i].c_str(), float_scalar_name.c_str(), float_str_n) == 0)
			file.getField(i, 1, float_scalar);

		if (properties[i].size() == int_str_n && strncmp(properties[i].c_str(), int_scalar_name.c_str(), int_str_n) == 0)
			file.getField(i, 1, int_scalar);

	}
	
	return;
}




void load_annot(std::string& dataPath,
	std::vector<int>& int_scalar,
	std::string& int_scalar_name)
{
	// Variables 
	uint64_t num_points(0);
	std::vector<npm::PLYType> types;
	std::vector<std::string> properties;

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
				std::vector<PointXYZ> &f_pts,
				std::vector<float> &timestamps,
				std::vector<int> &rings,
				std::vector<int> &loc_labels,
				std::string &save_path,
				std::string &time_name,
				std::string &ring_name)
{

	if (loc_labels.size() > 0)
	{
		// Load annotations
		size_t i0 = dataPath.rfind(std::string("/"));
		std::string f_name = dataPath.substr(i0, dataPath.size() - i0);
		std::vector<int> categories;
		std::string int_scalar_name = "cat";
		std::string annot_path = save_path + "/tmp_frames" + f_name;

		load_annot(annot_path, categories, int_scalar_name);

		// Load raw points
		std::vector<PointXYZ> raw_f_pts;
		std::vector<float> raw_timestamps;
		std::vector<int> raw_rings;
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



void random_3_pick(int &A_i, int &B_i, int &C_i,
				   std::uniform_int_distribution<int> &distribution,
				   std::default_random_engine &generator)
{
	A_i = distribution(generator);
	B_i = distribution(generator);
	C_i = distribution(generator);
	while (B_i == A_i)
		B_i = distribution(generator);
	while (C_i == A_i || C_i == B_i)
		C_i = distribution(generator);
}

bool is_triplet_good(PointXYZ A, PointXYZ B, PointXYZ C, PointXYZ &u)
{
	PointXYZ AB = B - A;
	PointXYZ AC = C - A;

	float normAB = AB.sq_norm();
	if (normAB < 1e-6) // <=> ||AB|| < 1mm
		return false;

	float normAC = AC.sq_norm();
	if (normAC < 1e-6) // <=> ||AC|| < 1mm
		return false;

	u = (AB).cross(AC);
	if (u.sq_norm() / (normAB * normAC)  < 1e-4) // <=> angle BAC < 0.5 degres
		return false;

	return true;
}

Plane3D plane_ransac(std::vector<PointXYZ> &points,
					 float max_dist,
					 int max_steps)
{
	// Random generator
  	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::uniform_int_distribution<int> distribution(0, points.size() - 1);

	// signed distances to plane
	std::vector<float> distances(points.size());

	// Parameters
	float factor = 1.0 / (max_dist * 4);
	int A_i, B_i, C_i;
	PointXYZ u;

	// Initial values
    float best_vote = 0;
	Plane3D best_P(points[0], points[1], points[2]);

	for (int i = 0; i < max_steps; i++)
	{

		// Randomly pick three points (with ill-defined detection)
		random_3_pick(A_i, B_i, C_i, distribution, generator);
		bool good_triplet = is_triplet_good(points[A_i], points[B_i], points[C_i], u);
		while(!good_triplet)
		{
			random_3_pick(A_i, B_i, C_i, distribution, generator);
			good_triplet = is_triplet_good(points[A_i], points[B_i], points[C_i], u);
		}

		// Corresponding plane
		Plane3D P_ABC(points[A_i], u);

		// Votes for this plane
		P_ABC.point_distances(points, distances);
		float sum_vote = 0;
		for (auto &d: distances)
		{
			if (d < max_dist)
				sum_vote += 1.0 - d * factor;
		}

		// Update best
		if (sum_vote > best_vote)
		{
			best_vote = sum_vote;
			best_P = P_ABC;
		}
	}

	return best_P;
}

Plane3D frame_ground_ransac(std::vector<PointXYZ> &points,
							std::vector<PointXYZ> &normals,
							float vertical_thresh_deg,
							float max_dist)
{

	// Parameters
	float vertical_thresh = vertical_thresh_deg * M_PI / 180;

	// Variables
	float clip0 = -0.99999999;
	float clip1 = 0.99999999;

	// Get points with a vetical normal (we assume the lidar is horizontal)
	std::vector<float> vertical_angles;
	std::vector<PointXYZ> candidates;
	vertical_angles.reserve(points.size());
	candidates.reserve(points.size());
	size_t i = 0;
	for (auto &n : normals)
	{
		float clip_nz = std::max(std::min(n.z, clip1), clip0);
		float vertical_angle = acos(abs(clip_nz));
		vertical_angles.push_back(vertical_angle);
		if (vertical_angle < vertical_thresh)
			candidates.push_back(points[i]);

		i++;
	}

	// Get the ground plane with ransac
	Plane3D ground_P = plane_ransac(candidates, max_dist, 100);

	return ground_P;
}

// float tukey(float x)
// {
// 	float w = pow(1-pow(x,2),2);
// 	if (abs(x) > 1)
// 		w = 0;
// 	return w;
// }

// void IRLS_optimization(Mat points, Plane& P, int nb_iter, float standardDeviation)
// {
// 	//----------------------------------------------------------------------------------0
// 	//	Optimisation par IRLS :															|
// 	//																					|
// 	//		Repeter iter fois :															|
// 	//			1) (Re)calculer les poids des points par rapport au plan				|
// 	//			2) Trouver le plan optimal pour ces poids								|
// 	//																					|
// 	//		IRLS = iteratively reweighted least squares									|
// 	//		On peut choisir plusieurs fonctions (M-estimateurs) afin de jour le role	|
// 	//		de poids. Je choisis l'estimateur de Tukey afin d'ecarter les outliers		|
// 	//		efficacement.																|
// 	//																					|
// 	//----------------------------------------------------------------------------------0

// 	// Initialisation des variables :
// 	// ******************************

// 	Mat distances;
// 	Mat tukeyWeights;


// 	// Lancement de la boucle :
// 	// ************************

// 	for(int i=0; i<nb_iter; i++)
// 	{
// 		// Calcul des poids :
// 		// ******************

// 		// la fonction de Tuckey est :
// 		//			 / (1-(distance_i/standardDeviation).^2).^2		si (distance_i < standardDeviation)
// 		//	  w_i = <
// 		//			 \					0							sinon

// 		// si (distance_i < standardDeviation)
// 		P.distance(points,distances);
// 		Mat temp;
// 		pow(distances/standardDeviation,2,temp);
// 		pow(1-temp,2,tukeyWeights);

// 		// sinon
// 		Mat inliers;
// 		compare(distances, standardDeviation, inliers, CMP_LT);
// 		inliers.convertTo(inliers,CV_32F);
// 		multiply(inliers,tukeyWeights,tukeyWeights,(double)1/255);


// 		// Plan optimal par les moindres carres ponderes :
// 		// ***********************************************

// 		// Si on decrit le plan d'equation ax+by+cz=1 par le vecteur u tel que u = [a, b, c]
// 		// Alors on peut montrer en derivant la fonction objectif S = sum(wi*(a.xi+b.yi+c.zi)� que u verifie le systeme lineaire
// 		//
// 		//			A * u' = B      o�		A = (weightedPoints)' * points
// 		//									B = sum(weightedPoints)'
// 		//
// 		//		(voir rapport pour explications plus claires)

// 		// Points ponderes
// 		Mat weightedPoints;
// 		repeat(tukeyWeights,1,3,tukeyWeights);
// 		multiply(tukeyWeights,points,weightedPoints);

// 		// Matrice A
// 		Mat A = weightedPoints.t() * (points);

// 		// Matrice B
// 		weightedPoints = weightedPoints.reshape(3);
// 		Scalar b = sum(weightedPoints);
// 		Mat B(1,1,CV_32FC3,b);
// 		B = (B.reshape(1)).t();

// 		// Solution
// 		Mat result = (A.inv()*B);
// 		P = Plane(result.at<float>(0), result.at<float>(1), result.at<float>(2));

// 	}
// }


bool rot_u_to_v(PointXYZ u, PointXYZ v, Eigen::Matrix3d &R)
{
	//
	// Get rotation matrix from u to v
	//

	// Get cross product
	PointXYZ w = u.cross(v);
	float norm_w = sqrt(w.sq_norm());
	
	// Do not go through if norm of vector w is too small
	if (norm_w < 1e-9) // <=> ||w|| < 0.001mm
		return false;

	// Get cos of rot angle
	float inv_norm_uv = 1 / sqrt( u.sq_norm() * v.sq_norm());
	float C = u.dot(v) * inv_norm_uv;
	
	if (C < -0.99999)
		return false;

	// Get sin of rot angle
	float S = norm_w * inv_norm_uv;

	// Formula from https://math.stackexchange.com/questions/4155049/how-do-you-generate-a-rotation-matrix-in-3d-for-some-given-angle-and-axis
	Eigen::Matrix<double, 3, 1> a;
	a(0) = w.x / norm_w;
	a(1) = w.y / norm_w;
	a(2) = w.z / norm_w;

	Eigen::Matrix3d skew = Eigen::Matrix3d::Zero(3, 3);
	skew(0, 1) = - a(2);
	skew(0, 2) = a(1);
	skew(1, 2) = - a(0);
	skew(1, 0) = a(2);
	skew(2, 0) = - a(1);
	skew(2, 1) = a(0);

	Eigen::Matrix3d tmp = (1 - C) * (a * a.transpose());
	R = tmp + S * skew + Eigen::Matrix3d::Identity() * C;

	return true;
}
