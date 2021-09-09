#include <Python.h>
#include <numpy/arrayobject.h>
#include "../src/pointmap/pointmap.h"
#include <string>




// docstrings for our module
// *************************

static char module_docstring[] = "This module provides lidar related functions";

static char get_visibility_docstring[] = "Gets the visibility mask as 1D ranges or 2D costmap from a lidar pointcloud";

static char other_utils_docstring[] = "TODO";


// Declare the functions
// *********************

static PyObject* get_visibility(PyObject* self, PyObject* args, PyObject* keywds);
static PyObject* other_utils(PyObject* self, PyObject* args, PyObject* keywds);


// Specify the members of the module
// *********************************

static PyMethodDef module_methods[] = 
{
	{ "get_visibility", (PyCFunction)get_visibility, METH_VARARGS | METH_KEYWORDS, get_visibility_docstring },
	{ "other_utils", (PyCFunction)other_utils, METH_VARARGS | METH_KEYWORDS, other_utils_docstring },
	{NULL, NULL, 0, NULL}
};


// Initialize the module
// *********************

static struct PyModuleDef moduledef = 
{
    PyModuleDef_HEAD_INIT,
    "lidar_utils",		// m_name
    module_docstring,       // m_doc
    -1,                     // m_size
    module_methods,         // m_methods
    NULL,                   // m_reload
    NULL,                   // m_traverse
    NULL,                   // m_clear
    NULL,                   // m_free
};

PyMODINIT_FUNC PyInit_lidar_utils(void)
{
    import_array();
	return PyModule_Create(&moduledef);
}


// Definition of the batch_subsample method
// **********************************

static PyObject* get_visibility(PyObject* self, PyObject* args, PyObject* keywds)
{

	// Manage inputs
	// *************

	// Args containers
	PyObject* points_obj = NULL;
	PyObject* center_obj = NULL;
	PyObject* ground_obj = NULL;
	int n_angles = 360;
	float zMin = 0.4;
	float zMax = 1.5;
	float dl_2D = -1.0;

	// Keywords containers
	static char* kwlist[] = { "points", "center", "ground", "n_angles", "z_min", "z_max", "dl_2D", NULL };

	// Parse the input  
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOO|$ifff", kwlist, &points_obj, &center_obj, &ground_obj, &n_angles, &zMin, &zMax, &dl_2D))
	{
		PyErr_SetString(PyExc_RuntimeError, "Error parsing arguments");
		return NULL;
	}

	// Interpret the input objects as numpy arrays.
	PyObject* points_array = PyArray_FROM_OTF(points_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* center_array = PyArray_FROM_OTF(center_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* ground_array = PyArray_FROM_OTF(ground_obj, NPY_FLOAT, NPY_IN_ARRAY);

	// Verify data was load correctly.
	if (points_array == NULL)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(center_array);
		Py_XDECREF(ground_array);
		PyErr_SetString(PyExc_RuntimeError, "Error converting points to numpy arrays of type float32");
		return NULL;
	}
	
	// Check that the input array respect the dims
	if ((int)PyArray_NDIM(points_array) != 2 || (int)PyArray_DIM(points_array, 1) != 3)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(center_array);
		Py_XDECREF(ground_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : points.shape is not (N, 3)");
		return NULL;
	}
	
	// Check that the input array respect the dims
	if ((int)PyArray_NDIM(center_array) != 1 || (int)PyArray_DIM(center_array, 0) != 3)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(center_array);
		Py_XDECREF(ground_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : center.shape is not (3,)");
		return NULL;
	}
	
	// Check that the input array respect the dims
	if ((int)PyArray_NDIM(ground_array) != 1 || (int)PyArray_DIM(ground_array, 0) != 4)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(center_array);
		Py_XDECREF(ground_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : ground.shape is not (4,)");
		return NULL;
	}

	// Number of points
	int N = (int)PyArray_DIM(points_array, 0);

	// Call the C++ function
	// *********************
	
	// Convert PyArray to Cloud C++ class
	vector<PointXYZ> points;
	points = vector<PointXYZ>((PointXYZ*)PyArray_DATA(points_array), (PointXYZ*)PyArray_DATA(points_array) + N);

	PointXYZ center3D = *((PointXYZ*)PyArray_DATA(center_array));

	vector<float> plane_values;
	plane_values = vector<float>((float*)PyArray_DATA(ground_array), (float*)PyArray_DATA(ground_array) + 4);
	Plane3D ground(plane_values[0], plane_values[1], plane_values[2], plane_values[3]);


	if (dl_2D < 0)
	{
		// Create result containers
		LaserRange2D ranges((size_t)n_angles);

		// Compute results
		ranges.compute_from_3D(points, center3D, ground, zMin, zMax);

		// Manage outputs
		// **************

		// Dimension of input containers
		npy_intp* ranges_dims = new npy_intp[1];
		ranges_dims[0] = n_angles;

		// Create output array
		PyObject* res_ranges_obj = PyArray_SimpleNew(1, ranges_dims, NPY_FLOAT);

		// Fill scores array with values
		size_t size_in_bytes2 = n_angles * sizeof(float);
		memcpy(PyArray_DATA(res_ranges_obj), ranges.range_table.data(), size_in_bytes2);

		// Merge results
		PyObject* ret = Py_BuildValue("N", res_ranges_obj);

		// Clean up
		// ********

		Py_XDECREF(points_array);
		Py_XDECREF(center_array);
		Py_XDECREF(ground_array);

		return ret;
		
	}
	else
	{
		cout << dl_2D << endl;

		// Create occupancy grid
		FullGrid2D costmap(dl_2D);

		// Update occupancy
		float radius = 10.0;
		costmap.update_from_3D(points, center3D, ground, zMin, zMax, radius);
		

		// Manage outputs
		// **************

		// Dimension of input containers
		npy_intp* ranges_dims = new npy_intp[2];
		ranges_dims[0] = costmap.sampleNY;
		ranges_dims[1] = costmap.sampleNX;

		// Create output array
		PyObject* res_ranges_obj = PyArray_SimpleNew(2, ranges_dims, NPY_FLOAT);

		// Fill scores array with values
		size_t size_in_bytes2 = costmap.sampleNX * costmap.sampleNY * sizeof(float);
		memcpy(PyArray_DATA(res_ranges_obj), costmap.scores.data(), size_in_bytes2);

		// Merge results
		PyObject* ret = Py_BuildValue("N", res_ranges_obj);

		// Clean up
		// ********

		Py_XDECREF(points_array);
		Py_XDECREF(center_array);
		Py_XDECREF(ground_array);

		return ret;


	}
}


static PyObject* other_utils(PyObject* self, PyObject* args, PyObject* keywds)
{

	// Manage inputs
	// *************

	// Args containers
	float tmp = 0.1;

	// Keywords containers
	static char* kwlist[] = { "tmp", NULL };

	// Parse the input  
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "f", kwlist, 
		&tmp))
	{
		PyErr_SetString(PyExc_RuntimeError, "Error parsing arguments");
		return NULL;
	}

	// Computations
	// ************

	vector<float> results;
	results.push_back(tmp);

	// Manage outputs
	// **************

	// Dimension of input containers
	npy_intp* res_dims = new npy_intp[1];
	res_dims[0] = 1;

	// Create output array
	PyObject* res_obj = PyArray_SimpleNew(1, res_dims, NPY_FLOAT);

	// Fill normals array with values
	size_t size_in_bytes = 1 * sizeof(float);
	memcpy(PyArray_DATA(res_obj), results.data(), size_in_bytes);

	// Merge results
	PyObject* ret = Py_BuildValue("N", res_obj);

	// Clean up
	// ********

	return ret;
}
