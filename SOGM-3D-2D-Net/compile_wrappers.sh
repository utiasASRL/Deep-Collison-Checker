#!/bin/bash

cd cpp_wrappers

# # Compile cpp icp normals
# cd cpp_pointmap
# python3 setup.py build_ext --inplace
# cd ..

# # Compile cpp subsampling
# cd cpp_subsampling
# python3 setup.py build_ext --inplace
# cd ..

# # Compile cpp neighbors
# cd cpp_neighbors
# python3 setup.py build_ext --inplace
# cd ..

# # Compile cpp polar normals
# cd cpp_polar_normals
# python3 setup.py build_ext --inplace
# cd ..

# # Compile cpp icp normals
# cd cpp_icp
# python3 setup.py build_ext --inplace
# cd ..

# # Compile cpp region growing
# # cd cpp_region_growing
# # python3 setup.py build_ext --inplace
# # cd ..


# Compile cpp icp normals
cd cpp_slam
python3 setup.py build_ext --inplace
cd ..

# # Compile cpp icp normals
# cd cpp_lidar_utils
# python3 setup.py build_ext --inplace
# cd ..