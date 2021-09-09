#include <Python.h>
#include <numpy/arrayobject.h>
#include "../src/pointmap/pointmap.h"
#include <string>



// docstrings for our module
// *************************

static char module_docstring[] = "This module provides a method to create and update a pointmap";

static char update_map_docstring[] = "Method that updates the map with new points, or create it if no previous map is given";


// Declare the functions
// *********************

static PyObject* update_map(PyObject* self, PyObject* args, PyObject* keywds);

// Specify the members of the module
// *********************************

static PyMethodDef module_methods[] =
{
	{ "update_map", (PyCFunction)update_map, METH_VARARGS | METH_KEYWORDS, update_map_docstring },
	{NULL, NULL, 0, NULL}
};


// Initialize the module
// *********************

static struct PyModuleDef moduledef =
{
	PyModuleDef_HEAD_INIT,
	"pointmap",					// m_name
	module_docstring,       // m_doc
	-1,                     // m_size
	module_methods,         // m_methods
	NULL,                   // m_reload
	NULL,                   // m_traverse
	NULL,                   // m_clear
	NULL,                   // m_free
};

PyMODINIT_FUNC PyInit_pointmap(void)
{
	import_array();
	return PyModule_Create(&moduledef);
}

// Definition of the update_map method
// ***********************************

static PyObject* update_map(PyObject* self, PyObject* args, PyObject* keywds)
{

	// Manage inputs
	// *************

	// Args containers
	PyObject* pts_obj = NULL;
	PyObject* norm_obj = NULL;
	PyObject* s_obj = NULL;
	PyObject* map_pts_obj = NULL;
	PyObject* map_norm_obj = NULL;
	PyObject* map_s_obj = NULL;
	PyObject* map_n_obj = NULL;
	float map_dl;

	// Keywords containers
	static char* kwlist[] = { (char*)"points", (char*)"normals", (char*)"scores", 
		(char*)"map_points", (char*)"map_normals", (char*)"map_scores", (char*)"map_counts",
		(char*)"map_dl", NULL };

	// Parse the input  
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOO|$OOOOf", kwlist,
		&pts_obj,
		&norm_obj,
		&s_obj,
		&map_pts_obj,
		&map_norm_obj,
		&map_s_obj,
		&map_n_obj,
		&map_dl))
	{
		PyErr_SetString(PyExc_RuntimeError, "Error parsing arguments");
		return NULL;
	}

	// Check if we have a previous map
	bool previous_map = !(map_pts_obj == NULL);

	// Interpret the input objects as numpy arrays.
	PyObject* pts_array = PyArray_FROM_OTF(pts_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* norm_array = PyArray_FROM_OTF(norm_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* s_array = PyArray_FROM_OTF(s_obj, NPY_FLOAT, NPY_IN_ARRAY);

	PyObject* map_pts_array = NULL;
	PyObject* map_norm_array = NULL;
	PyObject* map_s_array = NULL;
	PyObject* map_n_array = NULL;
	if (previous_map)
	{
		map_pts_array = PyArray_FROM_OTF(map_pts_obj, NPY_FLOAT, NPY_IN_ARRAY);
		map_norm_array = PyArray_FROM_OTF(map_norm_obj, NPY_FLOAT, NPY_IN_ARRAY);
		map_s_array = PyArray_FROM_OTF(map_s_obj, NPY_FLOAT, NPY_IN_ARRAY);
		map_n_array = PyArray_FROM_OTF(map_n_obj, NPY_INT, NPY_IN_ARRAY);
	}


	// Data verification
	// *****************

	vector<bool> conditions;
	vector<string> error_messages;

	// Verify data was load correctly.
	conditions.push_back(pts_array == NULL);
	error_messages.push_back("Error converting points to numpy arrays of type float32");
	conditions.push_back(norm_array == NULL);
	error_messages.push_back("Error converting normals to numpy arrays of type float32");
	conditions.push_back(s_array == NULL);
	error_messages.push_back("Error converting scores to numpy arrays of type float32");

	// Check that the input array respect the dims
	conditions.push_back((int)PyArray_NDIM(pts_array) != 2 || (int)PyArray_DIM(pts_array, 1) != 3);
	error_messages.push_back("Error, wrong dimensions : points.shape is not (N, 3)");
	conditions.push_back((int)PyArray_NDIM(norm_array) != 2 || (int)PyArray_DIM(norm_array, 1) != 3);
	error_messages.push_back("Error, wrong dimensions : normals.shape is not (N, 3)");
	conditions.push_back((int)PyArray_NDIM(s_array) != 1);
	error_messages.push_back("Error, wrong dimensions : scores.shape is not (N,)");

	// Check number of points
	int N = (int)PyArray_DIM(pts_array, 0);
	conditions.push_back((int)PyArray_DIM(norm_array, 0) != N);
	error_messages.push_back("Error: number of normals not equal to the number of points");
	conditions.push_back((int)PyArray_DIM(s_array, 0) != N);
	error_messages.push_back("Error: number of scores not equal to the number of points");

	int N_map = 0;
	if (previous_map)
	{
		N_map = (int)PyArray_DIM(map_pts_array, 0);

		// Verify data was load correctly.
		conditions.push_back(map_pts_array == NULL);
		error_messages.push_back("Error converting map_points to numpy arrays of type float32");
		conditions.push_back(map_norm_array == NULL);
		error_messages.push_back("Error converting map_normals to numpy arrays of type float32");
		conditions.push_back(map_s_array == NULL);
		error_messages.push_back("Error converting map_scores to numpy arrays of type float32");
		conditions.push_back(map_n_array == NULL);
		error_messages.push_back("Error converting map_counts to numpy arrays of type int2");

		// Check that the input array respect the dims
		conditions.push_back((int)PyArray_NDIM(map_pts_array) != 2 || (int)PyArray_DIM(map_pts_array, 1) != 3);
		error_messages.push_back("Error, wrong dimensions : map_points.shape is not (N, 3)");
		conditions.push_back((int)PyArray_NDIM(map_norm_array) != 2 || (int)PyArray_DIM(map_norm_array, 1) != 3);
		error_messages.push_back("Error, wrong dimensions : map_normals.shape is not (N, 3)");
		conditions.push_back((int)PyArray_NDIM(map_s_array) != 1);
		error_messages.push_back("Error, wrong dimensions : map_scores.shape is not (N,)");
		conditions.push_back((int)PyArray_NDIM(map_n_array) != 1);
		error_messages.push_back("Error, wrong dimensions : map_counts.shape is not (N,)");

		// Check number of points
		conditions.push_back((int)PyArray_DIM(map_norm_array, 0) != N_map);
		error_messages.push_back("Error: number of map_normals not equal to the number of map_points");
		conditions.push_back((int)PyArray_DIM(map_s_array, 0) != N_map);
		error_messages.push_back("Error: number of map_scores not equal to the number of map_points");
		conditions.push_back((int)PyArray_DIM(map_n_array, 0) != N_map);
		error_messages.push_back("Error: number of map_counts not equal to the number of map_points");
	}

	// Verify conditions
	for (size_t i = 0; i < conditions.size(); i++)
	{
		if (conditions[i])
		{
			Py_XDECREF(pts_array);
			Py_XDECREF(norm_array);
			Py_XDECREF(s_array);
			Py_XDECREF(map_pts_array);
			Py_XDECREF(map_norm_array);
			Py_XDECREF(map_s_array);
			Py_XDECREF(map_n_array);
			PyErr_SetString(PyExc_RuntimeError, error_messages[i].c_str());
			return NULL;
		}
	}

	// Call the C++ function
	// *********************

	// Convert PyArray to Cloud C++ class
	vector<PointXYZ> points, normals;
	vector<float> scores;
	points = vector<PointXYZ>((PointXYZ*)PyArray_DATA(pts_array), (PointXYZ*)PyArray_DATA(pts_array) + N);
	normals = vector<PointXYZ>((PointXYZ*)PyArray_DATA(norm_array), (PointXYZ*)PyArray_DATA(norm_array) + N);
	scores = vector<float>((float*)PyArray_DATA(s_array), (float*)PyArray_DATA(s_array) + N);


	// Create PointMap containers
	PointMapPython resMap(map_dl);

	if (previous_map)
	{
		resMap.points = vector<PointXYZ>((PointXYZ*)PyArray_DATA(map_pts_array), (PointXYZ*)PyArray_DATA(map_pts_array) + N_map);
		resMap.normals = vector<PointXYZ>((PointXYZ*)PyArray_DATA(map_norm_array), (PointXYZ*)PyArray_DATA(map_norm_array) + N_map);
		resMap.scores = vector<float>((float*)PyArray_DATA(map_s_array), (float*)PyArray_DATA(map_s_array) + N_map);
		resMap.counts = vector<int>((int*)PyArray_DATA(map_n_array), (int*)PyArray_DATA(map_n_array) + N_map);
	}

	// Compute results
	resMap.update(points, normals, scores);

	// Manage outputs
	// **************

	// Dimension of output containers
	size_t Nres = resMap.points.size();
	npy_intp* p_dims = new npy_intp[2];
	p_dims[0] = Nres;
	p_dims[1] = 3;
	npy_intp* s_dims = new npy_intp[1];
	s_dims[0] = Nres;

	// Create output array
	PyObject* res_pts_obj = PyArray_SimpleNew(2, p_dims, NPY_FLOAT);
	PyObject* res_norm_obj = PyArray_SimpleNew(2, p_dims, NPY_FLOAT);
	PyObject* res_s_obj = PyArray_SimpleNew(1, s_dims, NPY_FLOAT);
	PyObject* res_n_obj = PyArray_SimpleNew(1, s_dims, NPY_INT);

	// Fill transform array with values
	size_t size_in_bytes = Nres * 3 * sizeof(float);
	memcpy(PyArray_DATA(res_pts_obj), resMap.points.data(), size_in_bytes);
	memcpy(PyArray_DATA(res_norm_obj), resMap.normals.data(), size_in_bytes);


	size_t size_in_bytes2 = Nres * sizeof(float);
	memcpy(PyArray_DATA(res_s_obj), resMap.scores.data(), size_in_bytes2);

	size_t size_in_bytes3 = Nres * sizeof(int);
	memcpy(PyArray_DATA(res_n_obj), resMap.counts.data(), size_in_bytes3);

	// Merge results
	PyObject* ret = Py_BuildValue("NNNN", res_pts_obj, res_norm_obj, res_s_obj, res_n_obj);


	// Clean up
	// ********

	Py_XDECREF(pts_array);
	Py_XDECREF(norm_array);
	Py_XDECREF(s_array);
	Py_XDECREF(map_pts_array);
	Py_XDECREF(map_norm_array);
	Py_XDECREF(map_s_array);
	Py_XDECREF(map_n_array);

	return ret;
}


