# This represents a 3-D grid map, in which each cell represents the probability of
# occupancy at different times.
std_msgs/Header  header 
# Dimensions of the grid, depth for time. [cells]
uint32 height
uint32 width
uint32 depth
# The map resolutions in meters and seconds [m/cell] [s/cell]
float32 dl
float32 dt
# Origin of the grid in the world: position of the corner of voxel and time origin of the first prediction (0,0,0). [m, m, s]
geometry_msgs/Point origin
# Orientation of the map in the 2D plane [rad]
float32 theta
# The map data, in row-width-depth order, starting with (0,0,0).  Occupancy
# probabilities are in the range [0,255].
uint8[] data