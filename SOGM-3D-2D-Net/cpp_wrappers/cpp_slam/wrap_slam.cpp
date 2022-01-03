#include <Python.h>
#include <numpy/arrayobject.h>
#include "../src/pointmap_slam/pointmap_slam.h"
#include <string>



// docstrings for our module
// *************************

static char module_docstring[] = "This module provides a method to compute a Pointmap SLAM";

static char map_sim_sequence_docstring[] = "Method that calls the SLAM on a list of frames from simulation";

static char map_real_sequence_docstring[] = "Method that calls the SLAM on a list of frames from real experimentation";

static char merge_pointmaps_docstring[] = "Method that merges one or two point maps";

//static char other_docstring[] = "Method that ???";


// Declare the functions
// *********************

static PyObject* map_sim_sequence(PyObject* self, PyObject* args, PyObject* keywds);
static PyObject* map_real_sequence(PyObject* self, PyObject* args, PyObject* keywds);
static PyObject* merge_pointmaps(PyObject* self, PyObject* args, PyObject* keywds);

// Specify the members of the module
// *********************************

static PyMethodDef module_methods[] =
{
	{ "map_sim_sequence", (PyCFunction)map_sim_sequence, METH_VARARGS | METH_KEYWORDS, map_sim_sequence_docstring },
	{ "map_real_sequence", (PyCFunction)map_real_sequence, METH_VARARGS | METH_KEYWORDS, map_real_sequence_docstring },
	{ "merge_pointmaps", (PyCFunction)merge_pointmaps, METH_VARARGS | METH_KEYWORDS, merge_pointmaps_docstring },
	//{ "OTHER", (PyCFunction)OTHER, METH_VARARGS | METH_KEYWORDS, OTHER_docstring },
	{NULL, NULL, 0, NULL}
};


// Initialize the module
// *********************

static struct PyModuleDef moduledef =
{
	PyModuleDef_HEAD_INIT,
	"pointmap_slam",		// m_name
	module_docstring,       // m_doc
	-1,                     // m_size
	module_methods,         // m_methods
	NULL,                   // m_reload
	NULL,                   // m_traverse
	NULL,                   // m_clear
	NULL,                   // m_free
};

PyMODINIT_FUNC PyInit_pointmap_slam(void)
{
	import_array();
	return PyModule_Create(&moduledef);
}


// Definition of the map_sim_sequence method
// *****************************************

static PyObject* map_sim_sequence(PyObject* self, PyObject* args, PyObject* keywds)
{

	// Manage inputs
	// *************

	// Args containers
	const char* fnames_str;
	PyObject* ftimes_obj = NULL;
	PyObject* gt_H_obj = NULL;
	PyObject* gt_t_obj = NULL;
	PyObject* init_points_obj = NULL;
	PyObject* init_normals_obj = NULL;
	PyObject* init_scores_obj = NULL;
	PyObject* velo_base_obj = NULL;
	PyObject* odom_obj = NULL;
	SLAM_params slam_params;
	const char* save_path;

	// Keywords containers
	static char* kwlist[] = {(char *)"frame_names", (char *)"frame_times",
							 (char *)"gt_poses", (char *)"gt_times", (char *)"save_path",
							 (char *)"init_points", (char *)"init_normals", (char *)"init_scores",
							 (char *)"map_voxel_size", (char *)"frame_voxel_size", (char *)"motion_distortion", 
							 (char *)"filtering", (char *)"saving_for_loop_closure", (char *)"verbose_time",
							 (char *)"icp_samples", (char *)"icp_pairing_dist", (char *)"icp_planar_dist",
							 (char *)"icp_avg_steps", (char *)"icp_max_iter",
							 (char *)"H_velo_base", (char *)"odom_H",
							 NULL};

	// Parse the input  
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "sOOOsOOO|$ffpppflffllOO", kwlist,
		&fnames_str,
		&ftimes_obj,
		&gt_H_obj,
		&gt_t_obj,
		&save_path,
		&init_points_obj,
		&init_normals_obj,
		&init_scores_obj,
		&slam_params.map_voxel_size,
		&slam_params.frame_voxel_size,
		&slam_params.motion_distortion,
		&slam_params.filtering,
		&slam_params.saving_for_loop_closure,
		&slam_params.verbose_time,
		&slam_params.icp_params.n_samples,
		&slam_params.icp_params.max_pairing_dist,
		&slam_params.icp_params.max_planar_dist,
		&slam_params.icp_params.avg_steps,
		&slam_params.icp_params.max_iter,
		&velo_base_obj,
		&odom_obj))
	{
		PyErr_SetString(PyExc_RuntimeError, "Error parsing arguments");
		return NULL;
	}

	slam_params.icp_params.motion_distortion = slam_params.motion_distortion;

	// Interpret the input objects as numpy arrays.
	PyObject* ftimes_array = PyArray_FROM_OTF(ftimes_obj, NPY_DOUBLE, NPY_IN_ARRAY);
	PyObject* gt_H_array = PyArray_FROM_OTF(gt_H_obj, NPY_DOUBLE, NPY_IN_ARRAY);
	PyObject* gt_t_array = PyArray_FROM_OTF(gt_t_obj, NPY_DOUBLE, NPY_IN_ARRAY);
	PyObject* init_points_array = PyArray_FROM_OTF(init_points_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* init_normals_array = PyArray_FROM_OTF(init_normals_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* init_scores_array = PyArray_FROM_OTF(init_scores_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* velo_base_array = PyArray_FROM_OTF(velo_base_obj, NPY_DOUBLE, NPY_IN_ARRAY);

	// Check if odometry is given
	bool use_odom = true;
	PyObject* odom_array = NULL;
	if (odom_obj == NULL)
		use_odom = false;
	else
		odom_array = PyArray_FROM_OTF(odom_obj, NPY_DOUBLE, NPY_IN_ARRAY);

	// Data verification
	// *****************

	vector<bool> conditions;
	vector<string> error_messages;

	// Verify data was load correctly.
	conditions.push_back(ftimes_array == NULL);
	error_messages.push_back("Error converting frame times to numpy arrays of type int64");
	conditions.push_back(gt_H_array == NULL);
	error_messages.push_back("Error converting gt poses to numpy arrays of type float64 (double)");
	conditions.push_back(gt_t_array == NULL);
	error_messages.push_back("Error converting gt times to numpy arrays of type int64");
	conditions.push_back(init_points_array == NULL);
	error_messages.push_back("Error converting initial map points to numpy arrays of type float32");
	conditions.push_back(init_normals_array == NULL);
	error_messages.push_back("Error converting initial map normals to numpy arrays of type float32");
	conditions.push_back(init_scores_array == NULL);
	error_messages.push_back("Error converting initial map scores to numpy arrays of type float32");
	conditions.push_back(velo_base_array == NULL);
	error_messages.push_back("Error converting H_velo_base to numpy arrays of type double");
	conditions.push_back(use_odom && (odom_array == NULL));
	error_messages.push_back("Error converting odom_H to numpy arrays of type double");

	// Verify conditions
	for (size_t i = 0; i < conditions.size(); i++)
	{
		if (conditions[i])
		{
			Py_XDECREF(ftimes_array);
			Py_XDECREF(gt_H_array);
			Py_XDECREF(gt_t_array);
			Py_XDECREF(init_points_array);
			Py_XDECREF(init_normals_array);
			Py_XDECREF(init_scores_array);
			Py_XDECREF(velo_base_array);
			Py_XDECREF(odom_array);
			PyErr_SetString(PyExc_RuntimeError, error_messages[i].c_str());
			return NULL;
		}
	}

	// Check that the input array respect the dims
	conditions.push_back((int)PyArray_NDIM(ftimes_array) != 1);
	error_messages.push_back("Error, wrong dimensions : ftimes.shape is not (N1,)");
	conditions.push_back((int)PyArray_NDIM(gt_H_array) != 3 || (int)PyArray_DIM(gt_H_array, 1) != 4 || (int)PyArray_DIM(gt_H_array, 2) != 4);
	error_messages.push_back("Error, wrong dimensions : weights.shape is not (N2, 4, 4)");
	conditions.push_back((int)PyArray_NDIM(gt_t_array) != 1);
	error_messages.push_back("Error, wrong dimensions : map_points.shape is not (N2,)");
	conditions.push_back((int)PyArray_NDIM(init_points_array) != 2 || (int)PyArray_DIM(init_points_array, 1) != 3);
	error_messages.push_back("Error, wrong dimensions : init_points.shape is not (N3, 3)");
	conditions.push_back((int)PyArray_NDIM(init_normals_array) != 2 || (int)PyArray_DIM(init_normals_array, 1) != 3);
	error_messages.push_back("Error, wrong dimensions : init_normals.shape is not (N3, 3)");
	conditions.push_back((int)PyArray_NDIM(init_scores_array) != 1);
	error_messages.push_back("Error, wrong dimensions : init_scores.shape is not (N3,)");
	conditions.push_back((int)PyArray_NDIM(velo_base_array) != 2 || (int)PyArray_DIM(velo_base_array, 0) != 4 || (int)PyArray_DIM(velo_base_array, 1) != 4);
	error_messages.push_back("Error, wrong dimensions : H_velo_base.shape is not (4, 4)");
	conditions.push_back(use_odom && ((int)PyArray_NDIM(odom_array) != 3 || (int)PyArray_DIM(odom_array, 1) != 4 || (int)PyArray_DIM(odom_array, 2) != 4));
	error_messages.push_back("Error, wrong dimensions : odom_H.shape is not (N1, 4, 4)");

	// Check number of points
	size_t N_f = (size_t)PyArray_DIM(ftimes_array, 0);
	size_t N_gt = (size_t)PyArray_DIM(gt_t_array, 0);
	conditions.push_back((size_t)PyArray_DIM(gt_H_array, 0) != N_gt);
	error_messages.push_back("Error: number of gt_H not equal to the number of gt_t");
	conditions.push_back(use_odom && ((size_t)PyArray_DIM(odom_array, 0) != N_f));
	error_messages.push_back("Error: number of odom_H not equal to the number of frames");

	// Dimension of the features
	size_t N_map = (size_t)PyArray_DIM(init_points_array, 0);
	conditions.push_back((size_t)PyArray_DIM(init_normals_array, 0) != N_map);
	error_messages.push_back("Error: number of map normals not equal to the number of map points");
	conditions.push_back((size_t)PyArray_DIM(init_scores_array, 0) != N_map);
	error_messages.push_back("Error: number of map scores not equal to the number of map points");

	// Verify conditions
	for (size_t i = 0; i < conditions.size(); i++)
	{
		if (conditions[i])
		{
			Py_XDECREF(ftimes_array);
			Py_XDECREF(gt_H_array);
			Py_XDECREF(gt_t_array);
			Py_XDECREF(init_points_array);
			Py_XDECREF(init_normals_array);
			Py_XDECREF(init_scores_array);
			Py_XDECREF(velo_base_array);
			Py_XDECREF(odom_array);
			PyErr_SetString(PyExc_RuntimeError, error_messages[i].c_str());
			return NULL;
		}
	}

	// Call the C++ function
	// *********************

	// Convert frame names to string
	string frame_names(fnames_str);

	// Convert data to std vectors
	vector<double> f_t((double*)PyArray_DATA(ftimes_array), (double*)PyArray_DATA(ftimes_array) + N_f);
	vector<double> gt_t((double*)PyArray_DATA(gt_t_array), (double*)PyArray_DATA(gt_t_array) + N_gt);
	vector<double> gt_H_vec((double*)PyArray_DATA(gt_H_array), (double*)PyArray_DATA(gt_H_array) + N_gt * 4 * 4);
	vector<PointXYZ> init_points((PointXYZ*)PyArray_DATA(init_points_array), (PointXYZ*)PyArray_DATA(init_points_array) + N_map);
	vector<PointXYZ> init_normals((PointXYZ*)PyArray_DATA(init_normals_array), (PointXYZ*)PyArray_DATA(init_normals_array) + N_map);
	vector<float> init_scores((float*)PyArray_DATA(init_scores_array), (float*)PyArray_DATA(init_scores_array) + N_map);
	vector<double> velo_base_vec((double*)PyArray_DATA(velo_base_array), (double*)PyArray_DATA(velo_base_array) + 4 * 4);
	vector<double> odom_H_vec;
	if (use_odom)
		odom_H_vec = vector<double>((double*)PyArray_DATA(odom_array), (double*)PyArray_DATA(odom_array) + N_f * 4 * 4);
	else
		odom_H_vec = vector<double>(N_f * 4 * 4, 0.0);
	

	// Convert gt poses to Eigen matrices
	Eigen::Map<Eigen::MatrixXd> gt_H_T((double*)gt_H_vec.data(), 4, N_gt * 4);
	Eigen::MatrixXd gt_H = gt_H_T.transpose();
	
	// Convert odometry to Eigen matrices
	Eigen::Map<Eigen::MatrixXd> odom_H_T((double*)odom_H_vec.data(), 4, N_f * 4);
	Eigen::MatrixXd odom_H = odom_H_T.transpose();

	// Convert H_velo_base to Eigen matrices
	Eigen::Map<Eigen::Matrix4d> H_velo_base((double*)velo_base_vec.data(), 4, 4);
	slam_params.H_velo_base = H_velo_base.transpose();

	// Call main function
	Eigen::MatrixXd res_H;
	res_H = call_on_sim_sequence(frame_names, f_t, gt_H, gt_t, odom_H, init_points, init_normals, init_scores, slam_params, string(save_path));



	// Manage outputs
	// **************

	// Dimension of output containers
	npy_intp* H_dims = new npy_intp[2];
	H_dims[0] = 4;
	H_dims[1] = 4 * N_f;

	// Create output array
	PyObject* res_H_obj = PyArray_SimpleNew(2, H_dims, NPY_DOUBLE);

	// cout << res_H.block(0, 0, 4, 4) << endl;
	// cout << res_H.block(4, 0, 4, 4) << endl;

	//// Fill transform array with values
	size_t size_in_bytes = N_f * 4 * 4 * sizeof(double);
	memcpy(PyArray_DATA(res_H_obj), res_H.data(), size_in_bytes);


	// Merge results
	PyObject* ret = Py_BuildValue("N", res_H_obj);

	// Clean up
	// ********

	Py_XDECREF(ftimes_array);
	Py_XDECREF(gt_H_array);
	Py_XDECREF(gt_t_array);
	Py_XDECREF(init_points_array);
	Py_XDECREF(init_normals_array);
	Py_XDECREF(init_scores_array);
	Py_XDECREF(velo_base_array);
	Py_XDECREF(odom_array);


	return ret;
}


// Definition of the map_real_sequence method
// ******************************************

static PyObject* map_real_sequence(PyObject* self, PyObject* args, PyObject* keywds)
{

	// Manage inputs
	// *************

	// Args containers
	const char* fnames_str;
	PyObject* ftimes_obj = NULL;
	PyObject* init_points_obj = NULL;
	PyObject* init_normals_obj = NULL;
	PyObject* init_scores_obj = NULL;
	PyObject* velo_base_obj = NULL;
	PyObject* odom_obj = NULL;
	SLAM_params slam_params;
	const char* save_path;

	// Keywords containers
	static char* kwlist[] = {(char *)"frame_names", (char *)"frame_times", (char *)"save_path",
							 (char *)"init_points", (char *)"init_normals", (char *)"init_scores",
							 (char *)"map_voxel_size", (char *)"frame_voxel_size", (char *)"motion_distortion", 
							 (char *)"filtering", (char *)"saving_for_loop_closure", (char *)"force_flat_ground", 
							 (char *)"barycenter_map", (char *)"update_init_map", 
							 (char *)"verbose_time", (char *)"icp_samples", (char *)"icp_pairing_dist", 
							 (char *)"icp_planar_dist", (char *)"icp_avg_steps", (char *)"icp_max_iter",
							 (char *)"H_velo_base", (char *)"odom_H",
							 NULL};

	// Parse the input  
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "sOsOOO|$ffppppppflffllOO", kwlist,
		&fnames_str,
		&ftimes_obj,
		&save_path,
		&init_points_obj,
		&init_normals_obj,
		&init_scores_obj,
		&slam_params.map_voxel_size,
		&slam_params.frame_voxel_size,
		&slam_params.motion_distortion,
		&slam_params.filtering,
		&slam_params.saving_for_loop_closure,
		&slam_params.force_flat_ground,
		&slam_params.barycenter_map,
		&slam_params.update_init_map,
		&slam_params.verbose_time,
		&slam_params.icp_params.n_samples,
		&slam_params.icp_params.max_pairing_dist,
		&slam_params.icp_params.max_planar_dist,
		&slam_params.icp_params.avg_steps,
		&slam_params.icp_params.max_iter,
		&velo_base_obj,
		&odom_obj))
	{
		PyErr_SetString(PyExc_RuntimeError, "Error parsing arguments");
		return NULL;
	}

	slam_params.icp_params.motion_distortion = slam_params.motion_distortion;

	// Interpret the input objects as numpy arrays.
	PyObject* ftimes_array = PyArray_FROM_OTF(ftimes_obj, NPY_DOUBLE, NPY_IN_ARRAY);
	PyObject* init_points_array = PyArray_FROM_OTF(init_points_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* init_normals_array = PyArray_FROM_OTF(init_normals_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* init_scores_array = PyArray_FROM_OTF(init_scores_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* velo_base_array = PyArray_FROM_OTF(velo_base_obj, NPY_DOUBLE, NPY_IN_ARRAY);

	// Check if odometry is given
	bool use_odom = true;
	PyObject* odom_array = NULL;
	if (odom_obj == NULL)
		use_odom = false;
	else
		odom_array = PyArray_FROM_OTF(odom_obj, NPY_DOUBLE, NPY_IN_ARRAY);

	// Data verification
	// *****************

	vector<bool> conditions;
	vector<string> error_messages;

	// Verify data was load correctly.
	conditions.push_back(ftimes_array == NULL);
	error_messages.push_back("Error converting frame times to numpy arrays of type int64");
	conditions.push_back(init_points_array == NULL);
	error_messages.push_back("Error converting initial map points to numpy arrays of type float32");
	conditions.push_back(init_normals_array == NULL);
	error_messages.push_back("Error converting initial map normals to numpy arrays of type float32");
	conditions.push_back(init_scores_array == NULL);
	error_messages.push_back("Error converting initial map scores to numpy arrays of type float32");
	conditions.push_back(velo_base_array == NULL);
	error_messages.push_back("Error converting H_velo_base to numpy arrays of type double");
	conditions.push_back(use_odom && (odom_array == NULL));
	error_messages.push_back("Error converting odom_H to numpy arrays of type double");

	// Verify conditions
	for (size_t i = 0; i < conditions.size(); i++)
	{
		if (conditions[i])
		{
			Py_XDECREF(ftimes_array);
			Py_XDECREF(init_points_array);
			Py_XDECREF(init_normals_array);
			Py_XDECREF(init_scores_array);
			Py_XDECREF(velo_base_array);
			Py_XDECREF(odom_array);
			PyErr_SetString(PyExc_RuntimeError, error_messages[i].c_str());
			return NULL;
		}
	}

	// Check that the input array respect the dims
	conditions.push_back((int)PyArray_NDIM(ftimes_array) != 1);
	error_messages.push_back("Error, wrong dimensions : ftimes.shape is not (N1,)");
	conditions.push_back((int)PyArray_NDIM(init_points_array) != 2 || (int)PyArray_DIM(init_points_array, 1) != 3);
	error_messages.push_back("Error, wrong dimensions : init_points.shape is not (N3, 3)");
	conditions.push_back((int)PyArray_NDIM(init_normals_array) != 2 || (int)PyArray_DIM(init_normals_array, 1) != 3);
	error_messages.push_back("Error, wrong dimensions : init_normals.shape is not (N3, 3)");
	conditions.push_back((int)PyArray_NDIM(init_scores_array) != 1);
	error_messages.push_back("Error, wrong dimensions : init_scores.shape is not (N3,)");
	conditions.push_back((int)PyArray_NDIM(velo_base_array) != 2 || (int)PyArray_DIM(velo_base_array, 0) != 4 || (int)PyArray_DIM(velo_base_array, 1) != 4);
	error_messages.push_back("Error, wrong dimensions : H_velo_base.shape is not (4, 4)");
	conditions.push_back(use_odom && ((int)PyArray_NDIM(odom_array) != 3 || (int)PyArray_DIM(odom_array, 1) != 4 || (int)PyArray_DIM(odom_array, 2) != 4));
	error_messages.push_back("Error, wrong dimensions : odom_H.shape is not (N1, 4, 4)");

	// Check number of points
	size_t N_f = (size_t)PyArray_DIM(ftimes_array, 0);
	conditions.push_back(use_odom && ((size_t)PyArray_DIM(odom_array, 0) != N_f));
	error_messages.push_back("Error: number of odom_H not equal to the number of frames");

	// Dimension of the features
	size_t N_map = (size_t)PyArray_DIM(init_points_array, 0);
	conditions.push_back((size_t)PyArray_DIM(init_normals_array, 0) != N_map);
	error_messages.push_back("Error: number of map normals not equal to the number of map points");
	conditions.push_back((size_t)PyArray_DIM(init_scores_array, 0) != N_map);
	error_messages.push_back("Error: number of map scores not equal to the number of map points");

	// Verify conditions
	for (size_t i = 0; i < conditions.size(); i++)
	{
		if (conditions[i])
		{
			Py_XDECREF(ftimes_array);
			Py_XDECREF(init_points_array);
			Py_XDECREF(init_normals_array);
			Py_XDECREF(init_scores_array);
			Py_XDECREF(velo_base_array);
			Py_XDECREF(odom_array);
			PyErr_SetString(PyExc_RuntimeError, error_messages[i].c_str());
			return NULL;
		}
	}

	// Call the C++ function
	// *********************

	// Convert frame names to string
	string frame_names(fnames_str);

	// Convert data to std vectors
	vector<double> f_t((double*)PyArray_DATA(ftimes_array), (double*)PyArray_DATA(ftimes_array) + N_f);
	vector<PointXYZ> init_points((PointXYZ*)PyArray_DATA(init_points_array), (PointXYZ*)PyArray_DATA(init_points_array) + N_map);
	vector<PointXYZ> init_normals((PointXYZ*)PyArray_DATA(init_normals_array), (PointXYZ*)PyArray_DATA(init_normals_array) + N_map);
	vector<float> init_scores((float*)PyArray_DATA(init_scores_array), (float*)PyArray_DATA(init_scores_array) + N_map);
	vector<double> velo_base_vec((double*)PyArray_DATA(velo_base_array), (double*)PyArray_DATA(velo_base_array) + 4 * 4);
	vector<double> odom_H_vec;
	if (use_odom)
		odom_H_vec = vector<double>((double*)PyArray_DATA(odom_array), (double*)PyArray_DATA(odom_array) + N_f * 4 * 4);
	else
		odom_H_vec = vector<double>(N_f * 4 * 4, 0.0);
	
	// Convert odometry to Eigen matrices
	Eigen::Map<Eigen::MatrixXd> odom_H_T((double*)odom_H_vec.data(), 4, N_f * 4);
	Eigen::MatrixXd odom_H = odom_H_T.transpose();

	// Convert H_velo_base to Eigen matrices
	Eigen::Map<Eigen::Matrix4d> H_velo_base((double*)velo_base_vec.data(), 4, 4);
	slam_params.H_velo_base = H_velo_base.transpose();

	// Call main function
	Eigen::MatrixXd res_H;
	res_H = call_on_real_sequence(frame_names, f_t, odom_H, init_points, init_normals, init_scores, slam_params, string(save_path));


	// Manage outputs
	// **************

	// Dimension of output containers
	npy_intp* H_dims = new npy_intp[2];
	H_dims[0] = 4;
	H_dims[1] = 4 * N_f;

	// Create output array
	PyObject* res_H_obj = PyArray_SimpleNew(2, H_dims, NPY_DOUBLE);

	// cout << res_H.block(0, 0, 4, 4) << endl;
	// cout << res_H.block(4, 0, 4, 4) << endl;

	//// Fill transform array with values
	size_t size_in_bytes = N_f * 4 * 4 * sizeof(double);
	memcpy(PyArray_DATA(res_H_obj), res_H.data(), size_in_bytes);


	// Merge results
	PyObject* ret = Py_BuildValue("N", res_H_obj);

	// Clean up
	// ********

	Py_XDECREF(ftimes_array);
	Py_XDECREF(init_points_array);
	Py_XDECREF(init_normals_array);
	Py_XDECREF(init_scores_array);
	Py_XDECREF(velo_base_array);
	Py_XDECREF(odom_array);


	return ret;
}


// Definition of the merge_pointmaps method
// ******************************************

static PyObject* merge_pointmaps(PyObject* self, PyObject* args, PyObject* keywds)
{

	// Manage inputs
	// *************

	// Args containers
	PyObject* points_obj = NULL;
	PyObject* normals_obj = NULL;
	PyObject* scores_obj = NULL;
	PyObject* add_points_obj = NULL;
	PyObject* add_normals_obj = NULL;
	PyObject* add_scores_obj = NULL;
	SLAM_params slam_params;
	float map_voxel_size = 0.1;
	float merge_dist = 0.1;
	bool barycenter_map = false;

	// Keywords containers
	static char* kwlist[] = {(char *)"points", (char *)"normals", (char *)"scores",
							 (char *)"add_points", (char *)"add_normals", (char *)"add_scores",
							 (char *)"map_voxel_size", (char *)"merge_dist", (char *)"barycenter_map",
							 NULL};

	// Parse the input  
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOOOO|$ffp", kwlist,
		&points_obj,
		&normals_obj,
		&scores_obj,
		&add_points_obj,
		&add_normals_obj,
		&add_scores_obj,
		&map_voxel_size,
		&merge_dist,
		&barycenter_map))
	{
		PyErr_SetString(PyExc_RuntimeError, "Error parsing arguments");
		return NULL;
	}

	slam_params.icp_params.motion_distortion = slam_params.motion_distortion;

	// Interpret the input objects as numpy arrays.
	PyObject* points_array = PyArray_FROM_OTF(points_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* normals_array = PyArray_FROM_OTF(normals_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* scores_array = PyArray_FROM_OTF(scores_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* add_points_array = PyArray_FROM_OTF(add_points_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* add_normals_array = PyArray_FROM_OTF(add_normals_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* add_scores_array = PyArray_FROM_OTF(add_scores_obj, NPY_FLOAT, NPY_IN_ARRAY);

	// Data verification
	// *****************

	vector<bool> conditions;
	vector<string> error_messages;

	// Verify data was load correctly.
	conditions.push_back(points_array == NULL);
	error_messages.push_back("Error converting initial map points to numpy arrays of type float32");
	conditions.push_back(normals_array == NULL);
	error_messages.push_back("Error converting initial map normals to numpy arrays of type float32");
	conditions.push_back(scores_array == NULL);
	error_messages.push_back("Error converting initial map scores to numpy arrays of type float32");
	conditions.push_back(add_points_array == NULL);
	error_messages.push_back("Error converting add map points to numpy arrays of type float32");
	conditions.push_back(add_normals_array == NULL);
	error_messages.push_back("Error converting add map normals to numpy arrays of type float32");
	conditions.push_back(add_scores_array == NULL);
	error_messages.push_back("Error converting add map scores to numpy arrays of type float32");

	// Verify conditions
	for (size_t i = 0; i < conditions.size(); i++)
	{
		if (conditions[i])
		{
			Py_XDECREF(points_array);
			Py_XDECREF(normals_array);
			Py_XDECREF(scores_array);
			Py_XDECREF(add_points_array);
			Py_XDECREF(add_normals_array);
			Py_XDECREF(add_scores_array);
			PyErr_SetString(PyExc_RuntimeError, error_messages[i].c_str());
			return NULL;
		}
	}

	// Check that the input array respect the dims
	conditions.push_back((int)PyArray_NDIM(points_array) != 2 || (int)PyArray_DIM(points_array, 1) != 3);
	error_messages.push_back("Error, wrong dimensions : points.shape is not (N, 3)");
	conditions.push_back((int)PyArray_NDIM(normals_array) != 2 || (int)PyArray_DIM(normals_array, 1) != 3);
	error_messages.push_back("Error, wrong dimensions : normals.shape is not (N, 3)");
	conditions.push_back((int)PyArray_NDIM(scores_array) != 1);
	error_messages.push_back("Error, wrong dimensions : scores.shape is not (N,)");

	conditions.push_back((int)PyArray_NDIM(add_points_array) != 2 || (int)PyArray_DIM(add_points_array, 1) != 3);
	error_messages.push_back("Error, wrong dimensions : add_points.shape is not (N, 3)");
	conditions.push_back((int)PyArray_NDIM(add_normals_array) != 2 || (int)PyArray_DIM(add_normals_array, 1) != 3);
	error_messages.push_back("Error, wrong dimensions : add_normals.shape is not (N, 3)");
	conditions.push_back((int)PyArray_NDIM(add_scores_array) != 1);
	error_messages.push_back("Error, wrong dimensions : add_scores.shape is not (N,)");


	// Dimension of the features
	size_t N_map = (size_t)PyArray_DIM(points_array, 0);
	conditions.push_back((size_t)PyArray_DIM(normals_array, 0) != N_map);
	error_messages.push_back("Error: number of map normals not equal to the number of map points");
	conditions.push_back((size_t)PyArray_DIM(scores_array, 0) != N_map);
	error_messages.push_back("Error: number of map scores not equal to the number of map points");

	// Dimension of the features
	size_t N_add = (size_t)PyArray_DIM(add_points_array, 0);
	conditions.push_back((size_t)PyArray_DIM(add_normals_array, 0) != N_add);
	error_messages.push_back("Error: number of map normals not equal to the number of map points");
	conditions.push_back((size_t)PyArray_DIM(add_scores_array, 0) != N_add);
	error_messages.push_back("Error: number of map scores not equal to the number of map points");

	// Verify conditions
	for (size_t i = 0; i < conditions.size(); i++)
	{
		if (conditions[i])
		{
			Py_XDECREF(points_array);
			Py_XDECREF(normals_array);
			Py_XDECREF(scores_array);
			Py_XDECREF(add_points_array);
			Py_XDECREF(add_normals_array);
			Py_XDECREF(add_scores_array);
			PyErr_SetString(PyExc_RuntimeError, error_messages[i].c_str());
			return NULL;
		}
	}

	// Call the C++ function
	// *********************

	// Convert data to std vectors
	vector<PointXYZ> points((PointXYZ*)PyArray_DATA(points_array), (PointXYZ*)PyArray_DATA(points_array) + N_map);
	vector<PointXYZ> normals((PointXYZ*)PyArray_DATA(normals_array), (PointXYZ*)PyArray_DATA(normals_array) + N_map);
	vector<float> scores((float*)PyArray_DATA(scores_array), (float*)PyArray_DATA(scores_array) + N_map);
	
	vector<PointXYZ> add_points((PointXYZ*)PyArray_DATA(add_points_array), (PointXYZ*)PyArray_DATA(add_points_array) + N_add);
	vector<PointXYZ> add_normals((PointXYZ*)PyArray_DATA(add_normals_array), (PointXYZ*)PyArray_DATA(add_normals_array) + N_add);
	vector<float> add_scores((float*)PyArray_DATA(add_scores_array), (float*)PyArray_DATA(add_scores_array) + N_add);

	// Barycenter_map not handled
	if (barycenter_map)
		throw std::invalid_argument(string("\nERROR: Barycenter_map not handled\n"));


	// Create a point map from the first points
	// ****************************************

	cout << "\nCreating first Pointmap" << endl;
	PointMap map(map_voxel_size);
	if (points.size() > 0)
	{
		map.update_idx = -1;
		map.update(points, normals, scores, -1);
	}
	else
		throw std::invalid_argument(string("\nERROR: Zero initial points given\n"));
	cout << "OK\n" << endl;

	if (add_points.size() > 0)
	{

		// Computing distances to original map
		// ***********************************
		
		// Container for the points that are far enough from the map
		PointCloud far_cloud;
		far_cloud.pts.reserve(add_points.size());
		
		// Corresponding normals and scores
		vector<PointXYZ> far_normals;
		vector<float> far_scores;
		far_normals.reserve(add_points.size());
		far_scores.reserve(add_points.size());

		// Indices of the points that are currently too close
		vector<size_t> close_inds;
		close_inds.reserve(close_inds.size());

		cout << "\nComputing distances to original map" << endl;
		nanoflann::SearchParams search_params;
		float merge_d2 = merge_dist * merge_dist;
		for (size_t i = 0; i < add_points.size(); i++)
		{
			// get nearest neighbor
			float nn_d2;
			size_t nn_i;
			nanoflann::KNNResultSet<float> resultSet(1);
			resultSet.init(&nn_i, &nn_d2);
			map.tree.findNeighbors(resultSet, (float*)&add_points[i], search_params);

			if (nn_d2 > merge_d2)
			{
				// Add to far points
				far_cloud.pts.push_back(add_points[i]);
				far_normals.push_back(add_normals[i]);
				far_scores.push_back(add_scores[i]);
			}
			else
			{
				close_inds.push_back(i);
			}
			
		}
		cout << "OK\n" << endl;


		// Computing distances from eliminated to far points
		// *************************************************

		cout << "\nComputing distances from eliminated to far points" << endl;

		// Container for added points
		vector<PointXYZ> close_points;
		close_points.reserve(close_inds.size());
		
		// Corresponding normals and scores
		vector<PointXYZ> close_normals;
		vector<float> close_scores;
		close_normals.reserve(close_inds.size());
		close_scores.reserve(close_inds.size());

		// Create a KD Tree on our far_cloud
		PointXYZ_KDTree far_tree(3, far_cloud, KDTree_Params(10 /* max leaf */));
		far_tree.buildIndex();

		float remerge_dist = 0.95 * merge_dist;
		float remerge_d2 = remerge_dist * remerge_dist;

		for (auto &i : close_inds)
		{
			// get nearest neighbor
			float nn_d2;
			size_t nn_i;
			nanoflann::KNNResultSet<float> resultSet(1);
			resultSet.init(&nn_i, &nn_d2);
			far_tree.findNeighbors(resultSet, (float*)&add_points[i], search_params);

			if (nn_d2 < remerge_d2)
			{
				// Set far points as valid
				close_points.push_back(add_points[i]);
				close_normals.push_back(add_normals[i]);
				close_scores.push_back(add_scores[i]);
			}
		}
		cout << "OK\n" << endl;

		// Adding points to the map
		// ************************

		cout << "\nAdding points to the map" << endl;
		map.update(far_cloud.pts, far_normals, far_scores, 1);
		map.update(close_points, close_normals, close_scores, 2);
		cout << "OK\n" << endl;
	}

	// Manage outputs
	// **************

	// Dimension of output containers
	npy_intp* p_dims = new npy_intp[2];
	p_dims[0] = map.size();
	p_dims[1] = 3;

	npy_intp* s_dims = new npy_intp[1];
	s_dims[0] = map.size();

	// Create output array
	PyObject* res_points = PyArray_SimpleNew(2, p_dims, NPY_FLOAT);
	PyObject* res_normals = PyArray_SimpleNew(2, p_dims, NPY_FLOAT);
	PyObject* res_scores = PyArray_SimpleNew(1, s_dims, NPY_FLOAT);
	PyObject* res_latest = PyArray_SimpleNew(1, s_dims, NPY_INT);

	// Fill transform array with values
	size_t size_in_bytes = map.size() * 3 * sizeof(float);
	memcpy(PyArray_DATA(res_points), map.cloud.pts.data(), size_in_bytes);
	memcpy(PyArray_DATA(res_normals), map.normals.data(), size_in_bytes);
	size_in_bytes = map.size() * sizeof(float);
	memcpy(PyArray_DATA(res_scores), map.scores.data(), size_in_bytes);
	size_in_bytes = map.size() * sizeof(int);
	memcpy(PyArray_DATA(res_latest), map.latest.data(), size_in_bytes);

	// Merge results
	PyObject* ret = Py_BuildValue("NNNN", res_points, res_normals, res_scores, res_latest);

	// Clean up
	// ********

	Py_XDECREF(points_array);
	Py_XDECREF(normals_array);
	Py_XDECREF(scores_array);
	Py_XDECREF(add_points_array);
	Py_XDECREF(add_normals_array);
	Py_XDECREF(add_scores_array);

	return ret;
}

