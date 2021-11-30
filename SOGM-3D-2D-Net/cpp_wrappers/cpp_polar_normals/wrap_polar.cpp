#include <Python.h>
#include <numpy/arrayobject.h>
#include "../src/pointmap/pointmap.h"
#include <string>




// docstrings for our module
// *************************

static char module_docstring[] = "This module provides polar coordinates related functions";

static char polar_normals_docstring[] = "Gets normals from a lidar pointcloud";

static char map_frame_comp_docstring[] = "Gets difference between map and frame points in polar coordinates";


// Declare the functions
// *********************

static PyObject* polar_normals(PyObject* self, PyObject* args, PyObject* keywds);
static PyObject* map_frame_comp(PyObject* self, PyObject* args, PyObject* keywds);


// Specify the members of the module
// *********************************

static PyMethodDef module_methods[] = 
{
	{ "polar_normals", (PyCFunction)polar_normals, METH_VARARGS | METH_KEYWORDS, polar_normals_docstring },
	{ "map_frame_comp", (PyCFunction)map_frame_comp, METH_VARARGS | METH_KEYWORDS, map_frame_comp_docstring },
	{NULL, NULL, 0, NULL}
};


// Initialize the module
// *********************

static struct PyModuleDef moduledef = 
{
    PyModuleDef_HEAD_INIT,
    "polar_processing",		// m_name
    module_docstring,       // m_doc
    -1,                     // m_size
    module_methods,         // m_methods
    NULL,                   // m_reload
    NULL,                   // m_traverse
    NULL,                   // m_clear
    NULL,                   // m_free
};

PyMODINIT_FUNC PyInit_polar_processing(void)
{
    import_array();
	return PyModule_Create(&moduledef);
}


// Definition of the batch_subsample method
// **********************************

static PyObject* polar_normals(PyObject* self, PyObject* args, PyObject* keywds)
{

	// Manage inputs
	// *************

	// Args containers
	PyObject* queries_obj = NULL;

	// Keywords containers
	static char* kwlist[] = { "points", "radius", "lidar_n_lines", "h_scale", "r_scale", "verbose", NULL };
	float radius = 1.5;
	int lidar_n_lines = 32;
	float h_scale = 0.5f;
	float r_scale = 4.0f;
	int verbose = 0;


	// Parse the input  
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|$fiffi", kwlist, &queries_obj, &radius, &lidar_n_lines, &h_scale, &r_scale, &verbose))
	{
		PyErr_SetString(PyExc_RuntimeError, "Error parsing arguments");
		return NULL;
	}

	// Interpret the input objects as numpy arrays.
	PyObject* queries_array = PyArray_FROM_OTF(queries_obj, NPY_FLOAT, NPY_IN_ARRAY);

	// Verify data was load correctly.
	if (queries_array == NULL)
	{
		Py_XDECREF(queries_array);
		PyErr_SetString(PyExc_RuntimeError, "Error converting points to numpy arrays of type float32");
		return NULL;
	}
	// Check that the input array respect the dims
	if ((int)PyArray_NDIM(queries_array) != 2 || (int)PyArray_DIM(queries_array, 1) != 3)
	{
		Py_XDECREF(queries_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : points.shape is not (N, 3)");
		return NULL;
	}

	// Number of points
	int N = (int)PyArray_DIM(queries_array, 0);

	// Call the C++ function
	// *********************

	// Convert PyArray to Cloud C++ class
	vector<PointXYZ> queries;
	queries = vector<PointXYZ>((PointXYZ*)PyArray_DATA(queries_array), (PointXYZ*)PyArray_DATA(queries_array) + N);

	// Create result containers
	vector<PointXYZ> normals;
	vector<float> planarity;
	vector<float> linearity;

	// Compute results
	extract_features_multi_thread(queries, normals, planarity, linearity, lidar_n_lines, h_scale, r_scale, verbose);

	// Manage outputs
	// **************

	// Dimension of input containers
	npy_intp* normals_dims = new npy_intp[2];
	normals_dims[0] = normals.size();
	normals_dims[1] = 3;
	npy_intp* scores_dims = new npy_intp[1];
	scores_dims[0] = planarity.size();

	// Create output array
	PyObject* res_normals_obj = PyArray_SimpleNew(2, normals_dims, NPY_FLOAT);
	PyObject* res_plan_obj = PyArray_SimpleNew(1, scores_dims, NPY_FLOAT);
	PyObject* res_lin_obj = PyArray_SimpleNew(1, scores_dims, NPY_FLOAT);

	// Fill normals array with values
	size_t size_in_bytes = normals.size() * 3 * sizeof(float);
	memcpy(PyArray_DATA(res_normals_obj), normals.data(), size_in_bytes);

	// Fill scores array with values
	size_t size_in_bytes2 = planarity.size() * sizeof(float);
	memcpy(PyArray_DATA(res_plan_obj), planarity.data(), size_in_bytes2);
	memcpy(PyArray_DATA(res_lin_obj), linearity.data(), size_in_bytes2);

	// Merge results
	PyObject* ret = Py_BuildValue("NNN", res_normals_obj, res_plan_obj, res_lin_obj);

	// Clean up
	// ********

	Py_XDECREF(queries_array);

	return ret;
}


static PyObject* map_frame_comp(PyObject* self, PyObject* args, PyObject* keywds)
{

	// Manage inputs
	// *************

	// Args containers
	float map_dl = 0.1;
	float theta_dl = 0.1;
	float phi_dl = 1.0;
	float verbose_time = -1.0;
	bool motion_distortion = false;
	char* fnames_str;
	PyObject* map_p_obj = NULL;
	PyObject* map_n_obj = NULL;
	PyObject* H_obj = NULL;

	// Keywords containers
	static char* kwlist[] = { "frame_names", "map_points", "map_normals", "H_frames", "map_dl", "theta_dl", "phi_dl", "verbose_time", NULL };

	// Parse the input  
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "zOOO|$ffff", kwlist, 
		&fnames_str,
		&map_p_obj, 
		&map_n_obj, 
		&H_obj, 
		&map_dl, 
		&theta_dl, 
		&phi_dl, 
		&verbose_time))
	{
		PyErr_SetString(PyExc_RuntimeError, "Error parsing arguments");
		return NULL;
	}

	// Interpret the input objects as numpy arrays.
	PyObject* map_p_array = PyArray_FROM_OTF(map_p_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* map_n_array = PyArray_FROM_OTF(map_n_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* H_array = PyArray_FROM_OTF(H_obj, NPY_DOUBLE, NPY_IN_ARRAY);

	// Data verification
	// *****************

	vector<bool> conditions;
	vector<string> error_messages;

	// Verify data was load correctly.
	conditions.push_back(map_p_array == NULL);
	error_messages.push_back("Error converting map points to numpy arrays of type float32");
	conditions.push_back(map_n_array == NULL);
	error_messages.push_back("Error converting map normals to numpy arrays of type float32");
	conditions.push_back(H_array == NULL);
	error_messages.push_back("Error converting R to numpy arrays of type double");

	// Verify conditions
	for (size_t i = 0; i < conditions.size(); i++)
	{
		if (conditions[i])
		{
			Py_XDECREF(map_p_array);
			Py_XDECREF(map_n_array);
			Py_XDECREF(H_array);
			PyErr_SetString(PyExc_RuntimeError, error_messages[i].c_str());
			return NULL;
		}
	}

	// Check that the input array respect the dims
	conditions.push_back((int)PyArray_NDIM(map_p_array) != 2 || (int)PyArray_DIM(map_p_array, 1) != 3);
	error_messages.push_back("Error, wrong dimensions : map_points.shape is not (N, 3)");
	conditions.push_back((int)PyArray_NDIM(map_n_array) != 2 || (int)PyArray_DIM(map_n_array, 1) != 3);
	error_messages.push_back("Error, wrong dimensions : map_normals.shape is not (N, 3)");
	conditions.push_back((int)PyArray_NDIM(H_array) != 3 || (int)PyArray_DIM(H_array, 1) != 4 || (int)PyArray_DIM(H_array, 2) != 4);
	error_messages.push_back("Error, wrong dimensions : R.shape is not (N, 4, 4)");

	// Verify conditions
	for (size_t i = 0; i < conditions.size(); i++)
	{
		if (conditions[i])
		{
			Py_XDECREF(map_p_array);
			Py_XDECREF(map_n_array);
			Py_XDECREF(H_array);
			PyErr_SetString(PyExc_RuntimeError, error_messages[i].c_str());
			return NULL;
		}
	}

	// Check number of points
	size_t Nm = (size_t)PyArray_DIM(map_p_array, 0);
	size_t N_frames = (size_t)PyArray_DIM(H_array, 0);


	// Init variables
	// **************

	// Convert frame names to string
	string frame_names(fnames_str);

	// Convert PyArray to Cloud C++ class
	vector<PointXYZ> map_points, map_normals;
	map_points = vector<PointXYZ>((PointXYZ*)PyArray_DATA(map_p_array), (PointXYZ*)PyArray_DATA(map_p_array) + Nm);
	map_normals = vector<PointXYZ>((PointXYZ*)PyArray_DATA(map_n_array), (PointXYZ*)PyArray_DATA(map_n_array) + Nm);

	// Convert H to Eigen matrices
	vector<double> H_vec((double*)PyArray_DATA(H_array), (double*)PyArray_DATA(H_array) + N_frames * 4 * 4);
	Eigen::Map<Eigen::MatrixXd> all_H_t((double*)H_vec.data(), 4, N_frames * 4);
	Eigen::MatrixXd all_H = all_H_t.transpose();


	// Init point map
	// **************

	cout << "Creating point map" << endl;

	vector<float> map_scores(map_points.size(), 1.0);
	PointMap tmp_map(map_dl);
	tmp_map.cloud.pts = map_points;
	tmp_map.normals = map_normals;
	tmp_map.samples.reserve(map_points.size());
	
	// Create the pointmap voxels
	float inv_map_dl = 1.0 / map_dl;
	VoxKey k0;
	size_t p_i = 0;
	bool wrong_pointmap = false;
	for (auto &p : map_points)
	{
		// Corresponding key
		k0.x = (int)floor(p.x * inv_map_dl);
		k0.y = (int)floor(p.y * inv_map_dl);
		k0.z = (int)floor(p.z * inv_map_dl);

		// Update the sample map
		if (tmp_map.samples.count(k0) < 1)
			tmp_map.samples.emplace(k0, p_i);
		// else
		// {
		// 	cout << "[" << k0.x << ", " << k0.y << ", " << k0.z << "] /";
		// 	cout << " old index = " << tmp_map.samples[k0];
		// 	cout << " / new index = " << p_i << endl;
		// 	wrong_pointmap = true;
		// }
			
		p_i++;
	}

	if (wrong_pointmap)
	{
		Py_XDECREF(map_p_array);
		Py_XDECREF(map_n_array);
		Py_XDECREF(H_array);
		cout << "ERROR: multiple points in a single map voxel" << endl;
		return NULL;
	}

	cout << "OK" << endl;

	// Start movable detection
	// ***********************

	// Init map movable probabilities and counts
	vector<float> movable_probs(map_points.size(), 0);
	vector<int> movable_counts(map_points.size(), 0);

	// Parameters
	std::string time_name = "time";
	std::string ring_name = "ring";
	float last_t_max;

	// Timing
	clock_t t0 = std::clock();
	clock_t last_disp_t1 = std::clock();
	float fps = 0.0;
	float fps_regu = 0.9;

	// Loop on the lines of "frame_names" string
	istringstream iss(frame_names);
	size_t frame_i = 0;
	for (string line; getline(iss, line);)
	{

		// Load frame
		// **********

		// Load ply file
		
		vector<PointXYZ> f_pts;
		// vector<float> timestamps;
		// vector<int> rings;
		// load_cloud(line, f_pts, timestamps, time_name, rings, ring_name);
		load_cloud(line, f_pts);

		// // Get frame min and max times
		// float t_min, t_max;
		// float loop_ratio = 0.01;
		// get_min_max_times(timestamps, t_min, t_max, loop_ratio);
		
		// // Init last_time
		// if (frame_i < 1)
		// 	last_t_max = t_min;

		// Get the pose of the beginning and the end of the frame
		Eigen::Matrix4d H1 = all_H.block(frame_i * 4, 0, 4, 4);
		Eigen::Matrix4d H0;
		if (motion_distortion)
		{
			if (frame_i < 1)
				H0 = H1;
			else
				H0 = all_H.block((frame_i - 1) * 4, 0, 4, 4);
		}
		else
			H0 = Eigen::Matrix4d::Zero(4, 4);

		// Compute results
		tmp_map.update_movable(f_pts, H0, H1, theta_dl, phi_dl, movable_probs, movable_counts);
		// cout << "done" << endl;

		frame_i++;

		// Timing
		// ******
		clock_t t1 = std::clock();
		double duration = (t1 - t0) / (double)CLOCKS_PER_SEC;
		fps = fps_regu * fps + (1.0 - fps_regu) / duration;

		if (verbose_time > 0 && (t1 - last_disp_t1) / (double)CLOCKS_PER_SEC > verbose_time)
		{
			double remaining_sec = (N_frames - frame_i) / fps;
			int remaining_min = (int)floor(remaining_sec / 60.0);
			remaining_sec = remaining_sec - remaining_min * 60.0;
			char buffer[100];
			sprintf(buffer, "Annot %5d/%d at %5.1f fps - %d min %.0f sec remaining", (int)frame_i, N_frames, fps, remaining_min, remaining_sec);
			cout << string(buffer) << endl;
			last_disp_t1 = t1;
		}
		t0 = t1;
	}

	// Manage outputs
	// **************

	// Dimension of input containers
	npy_intp* res_dims = new npy_intp[1];
	res_dims[0] = movable_probs.size();

	// Create output array
	PyObject* movable_probs_obj = PyArray_SimpleNew(1, res_dims, NPY_FLOAT);
	PyObject* movable_counts_obj = PyArray_SimpleNew(1, res_dims, NPY_INT);

	// Fill normals array with values
	size_t size_in_bytes = movable_probs.size() * sizeof(float);
	memcpy(PyArray_DATA(movable_probs_obj), movable_probs.data(), size_in_bytes);

	// Fill scores array with values
	size_t size_in_bytes2 = movable_counts.size() * sizeof(int);
	memcpy(PyArray_DATA(movable_counts_obj), movable_counts.data(), size_in_bytes2);

	// Merge results
	PyObject* ret = Py_BuildValue("NN", movable_probs_obj, movable_counts_obj);

	// Clean up
	// ********

	Py_XDECREF(map_p_array);
	Py_XDECREF(map_n_array);
	Py_XDECREF(H_array);

	return ret;
}
