#include <Python.h>
#include <numpy/arrayobject.h>
#include "../src/icp/icp.h"
#include <string>



// docstrings for our module
// *************************

static char module_docstring[] = "This module provides a method to compute point to plane icp";

static char map_pt2pl_docstring[] = "Method that performs ICP that alugns a clouds on a map";

static char bundle_pt2pl_docstring[] = "Method that jointly optimize a sequence of frames";


// Declare the functions
// *********************

static PyObject* map_pt2pl(PyObject* self, PyObject* args, PyObject* keywds);
static PyObject* bundle_pt2pl(PyObject* self, PyObject* args, PyObject* keywds);

// Specify the members of the module
// *********************************

static PyMethodDef module_methods[] =
{
	{ "map_pt2pl", (PyCFunction)map_pt2pl, METH_VARARGS | METH_KEYWORDS, map_pt2pl_docstring },
	{ "bundle_pt2pl", (PyCFunction)bundle_pt2pl, METH_VARARGS | METH_KEYWORDS, bundle_pt2pl_docstring },
	{NULL, NULL, 0, NULL}
};


// Initialize the module
// *********************

static struct PyModuleDef moduledef =
{
	PyModuleDef_HEAD_INIT,
	"icp",					// m_name
	module_docstring,       // m_doc
	-1,                     // m_size
	module_methods,         // m_methods
	NULL,                   // m_reload
	NULL,                   // m_traverse
	NULL,                   // m_clear
	NULL,                   // m_free
};

PyMODINIT_FUNC PyInit_icp(void)
{
	import_array();
	return PyModule_Create(&moduledef);
}


// Definition of the map_pt2pl method
// **********************************

static PyObject* map_pt2pl(PyObject* self, PyObject* args, PyObject* keywds)
{

	// Manage inputs
	// *************

	// Args containers
	PyObject* tgt_obj = NULL;
	PyObject* tgtw_obj = NULL;
	PyObject* ref_obj = NULL;
	PyObject* refn_obj = NULL;
	PyObject* refw_obj = NULL;
	PyObject* init_obj = NULL;
	ICP_params params;

	// Keywords containers
	static char *kwlist[] = {(char *)"targets", (char *)"weights",
							 (char *)"map_points", (char *)"map_normals", (char *)"map_weights",
							 (char *)"init_H", (char *)"init_phi",
							 (char *)"n_samples", (char *)"max_pairing_dist", (char *)"max_iter",
							 (char *)"rotDiffThresh", (char *)"transDiffThresh", (char *)"avg_steps", (char *)"motion_distortion", NULL};

	// Parse the input
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOOO|$Oflflfflp", kwlist,
									 &tgt_obj,
									 &tgtw_obj,
									 &ref_obj,
									 &refn_obj,
									 &refw_obj,
									 &init_obj,
									 &params.init_phi,
									 &params.n_samples,
									 &params.max_pairing_dist,
									 &params.max_iter,
									 &params.rotDiffThresh,
									 &params.transDiffThresh,
									 &params.avg_steps,
									 &params.motion_distortion))
	{
		PyErr_SetString(PyExc_RuntimeError, "Error parsing arguments");
		return NULL;
	}

	// Interpret the input objects as numpy arrays.
	PyObject* tgt_array = PyArray_FROM_OTF(tgt_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* tgtw_array = PyArray_FROM_OTF(tgtw_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* ref_array = PyArray_FROM_OTF(ref_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* refn_array = PyArray_FROM_OTF(refn_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* refw_array = PyArray_FROM_OTF(refw_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* init_array = PyArray_FROM_OTF(init_obj, NPY_DOUBLE, NPY_IN_ARRAY);

	// Data verification
	// *****************

	vector<bool> conditions;
	vector<string> error_messages;

	// Verify data was load correctly.
	conditions.push_back(tgt_array == NULL);
	error_messages.push_back("Error converting targets to numpy arrays of type float32");
	conditions.push_back(ref_array == NULL);
	error_messages.push_back("Error converting map_points to numpy arrays of type float32");
	conditions.push_back(tgtw_array == NULL);
	error_messages.push_back("Error converting weights to numpy arrays of type float32");
	conditions.push_back(refn_array == NULL);
	error_messages.push_back("Error converting map_normals to numpy arrays of type float32");
	conditions.push_back(refw_array == NULL);
	error_messages.push_back("Error converting map_weights to numpy arrays of type float32");
	conditions.push_back(init_array == NULL);
	error_messages.push_back("Error converting initial transform to numpy arrays of type double");

	// Verify conditions
	for (size_t i = 0; i < conditions.size(); i++)
	{
		if (conditions[i])
		{
			Py_XDECREF(tgt_array);
			Py_XDECREF(tgtw_array);
			Py_XDECREF(ref_array);
			Py_XDECREF(refn_array);
			Py_XDECREF(refw_array);
			Py_XDECREF(init_array);
			PyErr_SetString(PyExc_RuntimeError, error_messages[i].c_str());
			return NULL;
		}
	}

	// Check that the input array respect the dims
	conditions.push_back((int)PyArray_NDIM(tgt_array) != 2 || (int)PyArray_DIM(tgt_array, 1) != 3);
	error_messages.push_back("Error, wrong dimensions : targets.shape is not (N, 3)");
	conditions.push_back((int)PyArray_NDIM(ref_array) != 2 || (int)PyArray_DIM(ref_array, 1) != 3);
	error_messages.push_back("Error, wrong dimensions : map_points.shape is not (N, 3)");
	conditions.push_back((int)PyArray_NDIM(tgtw_array) != 1);
	error_messages.push_back("Error, wrong dimensions : weights.shape is not (N,)");
	conditions.push_back((int)PyArray_NDIM(refn_array) != 2 || (int)PyArray_DIM(refn_array, 1) != 3);
	error_messages.push_back("Error, wrong dimensions : map_normals.shape is not (N, 3)");
	conditions.push_back((int)PyArray_NDIM(refw_array) != 1);
	error_messages.push_back("Error, wrong dimensions : map_weights.shape is not (N,)");
	conditions.push_back((int)PyArray_NDIM(init_array) != 2 || (int)PyArray_DIM(init_array, 0) != 4 || (int)PyArray_DIM(init_array, 1) != 4);
	error_messages.push_back("Error, wrong dimensions : init_H.shape is not (4, 4)");

	// Check number of points
	int Nt = (int)PyArray_DIM(tgt_array, 0);
	int Nr = (int)PyArray_DIM(ref_array, 0);
	conditions.push_back((int)PyArray_DIM(refn_array, 0) != Nr);
	error_messages.push_back("Error: number of map_normals not equal to the number of map_points");
	conditions.push_back((int)PyArray_DIM(tgtw_array, 0) != Nt);
	error_messages.push_back("Error: number of weights not equal to the number of targets");
	conditions.push_back((int)PyArray_DIM(refw_array, 0) != Nr);
	error_messages.push_back("Error: number of map_weights not equal to the number of map_points");

	// Verify conditions
	for (size_t i = 0; i < conditions.size(); i++)
	{
		if (conditions[i])
		{
			Py_XDECREF(tgt_array);
			Py_XDECREF(tgtw_array);
			Py_XDECREF(ref_array);
			Py_XDECREF(refn_array);
			Py_XDECREF(refw_array);
			Py_XDECREF(init_array);
			PyErr_SetString(PyExc_RuntimeError, error_messages[i].c_str());
			return NULL;
		}
	}

	// Call the C++ function
	// *********************

	// Convert PyArray to Cloud C++ class
	vector<PointXYZ> tgt_points, map_points, map_normals;
	vector<float> tgt_weights, map_weights;
	tgt_points = vector<PointXYZ>((PointXYZ*)PyArray_DATA(tgt_array), (PointXYZ*)PyArray_DATA(tgt_array) + Nt);
	tgt_weights = vector<float>((float*)PyArray_DATA(tgtw_array), (float*)PyArray_DATA(tgtw_array) + Nt);

	PointMapPython ptmap(1.0);
	ptmap.points = vector<PointXYZ>((PointXYZ*)PyArray_DATA(ref_array), (PointXYZ*)PyArray_DATA(ref_array) + Nr);
	ptmap.normals = vector<PointXYZ>((PointXYZ*)PyArray_DATA(refn_array), (PointXYZ*)PyArray_DATA(refn_array) + Nr);
	ptmap.scores = vector<float>((float*)PyArray_DATA(refw_array), (float*)PyArray_DATA(refw_array) + Nr);

	vector<double> init_vec = vector<double>((double*)PyArray_DATA(init_array), (double*)PyArray_DATA(init_array) +  4 * 4);
	Eigen::Map<Eigen::Matrix4d> init_transform((double*)init_vec.data(), 4, 4);
	params.init_transform = init_transform.transpose();

	// Create result containers
	ICP_results results;

	// Compute results
	PointToMapICPDebug(tgt_points, tgt_weights, ptmap.points, ptmap.normals, ptmap.scores, params, results);

	// Manage outputs
	// **************

	// Dimension of output containers
	size_t I = results.all_rms.size();
	npy_intp* H_dims = new npy_intp[2];
	H_dims[0] = 4;
	H_dims[1] = I * 4;
	npy_intp* rms_dims = new npy_intp[1];
	rms_dims[0] = I;

	// Create output array
	PyObject* res_H_obj = PyArray_SimpleNew(2, H_dims, NPY_DOUBLE);
	PyObject* res_rms_obj = PyArray_SimpleNew(1, rms_dims, NPY_FLOAT);
	PyObject* res_prms_obj = PyArray_SimpleNew(1, rms_dims, NPY_FLOAT);

	// Fill transform array with values
	size_t size_in_bytes = I * 4 * 4 * sizeof(double);
	memcpy(PyArray_DATA(res_H_obj), results.all_transforms.data(), size_in_bytes);
	size_t size_in_bytes2 = results.all_rms.size() * sizeof(float);
	memcpy(PyArray_DATA(res_rms_obj), results.all_rms.data(), size_in_bytes2);
	memcpy(PyArray_DATA(res_prms_obj), results.all_plane_rms.data(), size_in_bytes2);

	// Merge results
	PyObject* ret = Py_BuildValue("NNN", res_H_obj, res_rms_obj, res_prms_obj);

	// Clean up
	// ********

	Py_XDECREF(tgt_array);
	Py_XDECREF(tgtw_array);
	Py_XDECREF(ref_array);
	Py_XDECREF(refn_array);
	Py_XDECREF(refw_array);

	return ret;
}


// Definition of the bumdle_pt2pl method
// *************************************

static PyObject* bundle_pt2pl(PyObject* self, PyObject* args, PyObject* keywds)
{

	// Manage inputs
	// *************

	// Args containers
	PyObject* pts_obj = NULL;
	PyObject* norm_obj = NULL;
	PyObject* w_obj = NULL;
	PyObject* l_obj = NULL;
	ICP_params params;

	// Keywords containers
	static char* kwlist[] = { (char*)"targets", (char*)"normals", (char*)"weights", (char*)"lengths",
		(char*)"n_samples", (char*)"max_pairing_dist", (char*)"max_iter",
		(char*)"rotDiffThresh", (char*)"transDiffThresh", (char*)"avg_steps", NULL };

	// Parse the input  
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOO|$lflffl", kwlist,
		&pts_obj,
		&norm_obj,
		&w_obj,
		&l_obj,
		&params.n_samples,
		&params.max_pairing_dist,
		&params.max_iter,
		&params.rotDiffThresh,
		&params.transDiffThresh,
		&params.avg_steps))
	{
		PyErr_SetString(PyExc_RuntimeError, "Error parsing arguments");
		return NULL;
	}

	// Interpret the input objects as numpy arrays.
	PyObject* pts_array = PyArray_FROM_OTF(pts_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* norm_array = PyArray_FROM_OTF(norm_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* w_array = PyArray_FROM_OTF(w_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* l_array = PyArray_FROM_OTF(l_obj, NPY_INT, NPY_IN_ARRAY);

	// Data verification
	// *****************

	vector<bool> conditions;
	vector<string> error_messages;

	// Verify data was load correctly.
	conditions.push_back(pts_array == NULL);
	error_messages.push_back("Error converting targets to numpy arrays of type float32");
	conditions.push_back(norm_array == NULL);
	error_messages.push_back("Error converting normals to numpy arrays of type float32");
	conditions.push_back(w_array == NULL);
	error_messages.push_back("Error converting weights to numpy arrays of type float32");
	conditions.push_back(l_array == NULL);
	error_messages.push_back("Error converting references to numpy arrays of type float32");

	// Check that the input array respect the dims
	conditions.push_back((int)PyArray_NDIM(pts_array) != 2 || (int)PyArray_DIM(pts_array, 1) != 3);
	error_messages.push_back("Error, wrong dimensions : targets.shape is not (N, 3)");
	conditions.push_back((int)PyArray_NDIM(norm_array) != 2 || (int)PyArray_DIM(norm_array, 1) != 3);
	error_messages.push_back("Error, wrong dimensions : normals.shape is not (N, 3)");
	conditions.push_back((int)PyArray_NDIM(w_array) != 2);
	error_messages.push_back("Error, wrong dimensions : weights.shape is not (N, W)");
	conditions.push_back((int)PyArray_NDIM(l_array) != 1);
	error_messages.push_back("Error, wrong dimensions : lengths.shape is not (B,)");

	// Check number of points
	int B = (int)PyArray_DIM(l_array, 0);
	int N = (int)PyArray_DIM(pts_array, 0);
	int W = (int)PyArray_DIM(w_array, 1);

	conditions.push_back((int)PyArray_DIM(norm_array, 0) != N);
	error_messages.push_back("Error: number of normals not equal to the number of points");
	conditions.push_back((int)PyArray_DIM(w_array, 0) != N);
	error_messages.push_back("Error: number of weights not equal to the number of points");

	// Verify conditions
	for (size_t i = 0; i < conditions.size(); i++)
	{
		if (conditions[i])
		{
			Py_XDECREF(pts_array);
			Py_XDECREF(norm_array);
			Py_XDECREF(w_array);
			Py_XDECREF(l_array);
			PyErr_SetString(PyExc_RuntimeError, error_messages[i].c_str());
			return NULL;
		}
	}

	// Call the C++ function
	// *********************

	// Convert PyArray to Cloud C++ class
	vector<PointXYZ> points, normals;
	vector<float> weights;
	vector<int> lengths;
	points = vector<PointXYZ>((PointXYZ*)PyArray_DATA(pts_array), (PointXYZ*)PyArray_DATA(pts_array) + N);
	normals = vector<PointXYZ>((PointXYZ*)PyArray_DATA(norm_array), (PointXYZ*)PyArray_DATA(norm_array) + N);
	weights = vector<float>((float*)PyArray_DATA(w_array), (float*)PyArray_DATA(w_array) + N * W);
	lengths = vector<int>((int*)PyArray_DATA(l_array), (int*)PyArray_DATA(l_array) + B);

	// Create result containers
	BundleIcpResults results(B);


	// Compute results
	BundleICP(points, normals, weights, lengths, params, results);

	// Manage outputs
	// **************

	// Dimension of output containers
	size_t I = results.all_rms[0].size();
	npy_intp* H_dims = new npy_intp[3];
	H_dims[0] = B;
	H_dims[1] = 4;
	H_dims[2] = 4;
	npy_intp* rms_dims = new npy_intp[2];
	rms_dims[0] = B;
	rms_dims[1] = I;
	npy_intp* allH_dims = new npy_intp[2];
	allH_dims[0] = B * 4;
	allH_dims[1] = I * 4;

	// Convert result containers into contiguous memory containers
	vector<float> result_rms(B * I);
	for (size_t b = 0; b < (size_t)B; b++)
	{
		for (size_t i = 0; i < I; i++)
			result_rms[i + I * b] = results.all_rms[b][i];
	}

	// Create output array
	PyObject* res_H_obj = PyArray_SimpleNew(3, H_dims, NPY_DOUBLE);
	PyObject* res_rms_obj = PyArray_SimpleNew(2, rms_dims, NPY_FLOAT);
	PyObject* res_allH_obj = PyArray_SimpleNew(2, allH_dims, NPY_DOUBLE);

	// Fill transform array with values
	size_t size_in_bytes = B * 4 * 4 * sizeof(double);
	memcpy(PyArray_DATA(res_H_obj), results.transforms.data(), size_in_bytes);
	size_t size_in_bytes2 = B * I * sizeof(float);
	memcpy(PyArray_DATA(res_rms_obj), result_rms.data(), size_in_bytes2);
	size_t size_in_bytes3 = B * I * 4 * 4 * sizeof(double);
	memcpy(PyArray_DATA(res_allH_obj), results.all_transforms.data(), size_in_bytes3);

	// Merge results
	PyObject* ret = Py_BuildValue("NNN", res_H_obj, res_rms_obj, res_allH_obj);

	// Clean up
	// ********

	Py_XDECREF(pts_array);
	Py_XDECREF(norm_array);
	Py_XDECREF(w_array);
	Py_XDECREF(l_array);

	return ret;
}


