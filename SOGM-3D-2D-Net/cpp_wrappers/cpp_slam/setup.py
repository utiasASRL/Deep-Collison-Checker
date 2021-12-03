from distutils.core import setup, Extension
import numpy.distutils.misc_util

# Adding Eigen to project
# ***********************

EIGEN_INCLUDE = ["/usr/include/eigen3"]
CERES_LIBS = ["glog", "ceres", "gflags", "pthread", "spqr", "cholmod", "ccolamd", "camd", "colamd", "amd", "lapack", "f77blas", "cxsparse", "rt", "stdc++", "m"]
CERES_INCLUDE = []
CERES_LIB = []

# CERES_MACRO = 'CERES_USE_CXX11_THREADS'
# CERES_MACRO = 'CERES_USE_OPENMP'
# CERES_MACRO = 'CERES_NO_THREADS'

# Adding sources of the project
# *****************************

SOURCES = ["../src/cloud/cloud.cpp",
           "../src/cloud/points.cpp",
           "../src/npm_ply/ply_file_in.cc",
           "../src/npm_ply/ply_file_out.cc",
           "../src/grid_subsampling/grid_subsampling.cpp",
           "../src/polar_processing/polar_processing.cpp",
           "../src/icp/icp.cpp",
           "../src/pointmap/pointmap.cpp",
           "../src/pointmap_slam/pointmap_slam.cpp",
           "wrap_slam.cpp"]

module = Extension(name="pointmap_slam",
                   sources=SOURCES,
                   include_dirs=EIGEN_INCLUDE,
                   libraries=CERES_LIBS,
                   extra_compile_args=['-std=c++17'])


setup(ext_modules=[module], include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs())
