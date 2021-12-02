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
//		Cloud header
//
//----------------------------------------------------
//
//		Hugues THOMAS - 10/02/2017
//


# pragma once

#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdio.h>
#include <string.h>


#include <time.h>

#include "../npm_ply/ply_types.h"
#include "../npm_ply/ply_file_in.h"
#include "../npm_ply/ply_file_out.h"


//------------------------------------------------------------------------------------------------------------
// Point class
// ***********
//
//------------------------------------------------------------------------------------------------------------

class PointXYZ
{
public:

	// Elements
	// ********

	float x, y, z;


	// Methods
	// *******
	
	// Constructor
	PointXYZ() { x = 0; y = 0; z = 0; }
	PointXYZ(float x0, float y0, float z0) { x = x0; y = y0; z = z0; }
	
	// array type accessor
	float& operator[](std::size_t idx)       
	{
		if (idx == 2) return z;
		else if (idx == 1) return y;
		else  return x;
	}
    const float& operator[](std::size_t idx) const
	{
		if (idx == 2) return z;
		else if (idx == 1) return y;
		else return x;
	}

	// opperations
	float dot(const PointXYZ P) const
	{
		return x * P.x + y * P.y + z * P.z;
	}

	float sq_norm()
	{
		return x*x + y*y + z*z;
	}

	PointXYZ cross(const PointXYZ P) const
	{
		return PointXYZ(y*P.z - z*P.y, z*P.x - x*P.z, x*P.y - y*P.x);
	}	

	PointXYZ& operator+=(const PointXYZ& P)
	{
		x += P.x;
		y += P.y;
		z += P.z;
		return *this;
	}

	PointXYZ& operator-=(const PointXYZ& P)
	{
		x -= P.x;
		y -= P.y;
		z -= P.z;
		return *this;
	}

	PointXYZ& operator*=(const float& a)
	{
		x *= a;
		y *= a;
		z *= a;
		return *this;
	}
	
};


// Point Opperations
// *****************

inline PointXYZ operator + (const PointXYZ A, const PointXYZ B)
{
	return PointXYZ(A.x + B.x, A.y + B.y, A.z + B.z);
}

inline PointXYZ operator - (const PointXYZ A, const PointXYZ B)
{
	return PointXYZ(A.x - B.x, A.y - B.y, A.z - B.z);
}

inline PointXYZ operator * (const PointXYZ P, const float a)
{
	return PointXYZ(P.x * a, P.y * a, P.z * a);
}

inline PointXYZ operator * (const float a, const PointXYZ P)
{
	return PointXYZ(P.x * a, P.y * a, P.z * a);
}

inline std::ostream& operator << (std::ostream& os, const PointXYZ P)
{
	return os << "[" << P.x << ", " << P.y << ", " << P.z << "]";
}

inline bool operator == (const PointXYZ A, const PointXYZ B)
{
	return A.x == B.x && A.y == B.y && A.z == B.z;
}

inline PointXYZ floor(const PointXYZ P)
{
	return PointXYZ(std::floor(P.x), std::floor(P.y), std::floor(P.z));
}


PointXYZ max_point(const std::vector<PointXYZ>& points);
PointXYZ min_point(const std::vector<PointXYZ>& points);
PointXYZ max_point(const PointXYZ A, const PointXYZ B);
PointXYZ min_point(const PointXYZ A, const PointXYZ B);

//------------------------------------------------------------------------------------------------------------
// Point class 2D
// **************
//
//------------------------------------------------------------------------------------------------------------

class PointXY
{
public:

	// Elements
	// ********

	float x, y;


	// Methods
	// *******
	
	// Constructor
	PointXY() { x = 0; y = 0;}
	PointXY(float x0, float y0) { x = x0; y = y0;}
	PointXY(PointXYZ P) { x = P.x; y = P.y;}
	
	// array type accessor
	float operator [] (int i) const
	{
		if (i == 0) return x;
		else return y;
	}

	// opperations
	float dot(const PointXY P) const
	{
		return x * P.x + y * P.y;
	}

	float sq_norm() const
	{
		return x*x + y*y;
	}

	float cross(const PointXY P) const
	{
		return x*P.y - y*P.x;
	}	

	PointXY& operator+=(const PointXY& P)
	{
		x += P.x;
		y += P.y;
		return *this;
	}

	PointXY& operator-=(const PointXY& P)
	{
		x -= P.x;
		y -= P.y;
		return *this;
	}

	PointXY& operator*=(const float& a)
	{
		x *= a;
		y *= a;
		return *this;
	}
};


// Point Opperations
// *****************

inline PointXY operator + (const PointXY A, const PointXY B)
{
	return PointXY(A.x + B.x, A.y + B.y);
}

inline PointXY operator - (const PointXY A, const PointXY B)
{
	return PointXY(A.x - B.x, A.y - B.y);
}

inline PointXY operator * (const PointXY P, const float a)
{
	return PointXY(P.x * a, P.y * a);
}

inline PointXY operator * (const float a, const PointXY P)
{
	return PointXY(P.x * a, P.y * a);
}

inline std::ostream& operator << (std::ostream& os, const PointXY P)
{
	return os << "[" << P.x << ", " << P.y << "]";
}

inline bool operator == (const PointXY A, const PointXY B)
{
	return A.x == B.x && A.y == B.y;
}

inline PointXY floor(const PointXY P)
{
	return PointXY(std::floor(P.x), std::floor(P.y));
}


//------------------------------------------------------------------------------------------------------------
// Plane3D class
// *************
//
//------------------------------------------------------------------------------------------------------------

class Plane3D
{
public:

	// Elements
	// ********

	// The plane is define by the equation a*x + b*y + c*z = d. The values (a, b, c) are stored in a PointXYZ called u.
	PointXYZ u;
	float d;


	// Methods
	// *******

	// Constructor
	Plane3D() { u.x = 1; u.y = 0; u.z = 0; d = 0; }
	Plane3D(const float a0, const float b0, const float c0, const float d0) { u.x = a0; u.y = b0; u.z = c0; d = d0; }
	Plane3D(const PointXYZ P0, const PointXYZ N0)
	{
		// Init with point and normal
		u = N0;
		d = N0.dot(P0);
	}
	Plane3D(const PointXYZ A, const PointXYZ B, const PointXYZ C)
	{
		// Init with three points
		u = (B - A).cross(C - A);
		d = u.dot(A);
	}

	// Method getting distance to one point
	float point_distance(const PointXYZ P)
	{
		return std::abs((u.dot(P) - d) / std::sqrt(u.sq_norm()));
	}

	// Method getting square distance to one point
	float point_sq_dist(const PointXYZ P)
	{
		float tmp = u.dot(P) - d;
		return tmp * tmp / u.sq_norm();
	}

	// Method for reversing the normal of the Plane
	void reverse()
	{
		u *= -1;
		d *= -1;
	}


	// Method getting distances to some points
	void point_distances_signed(std::vector<PointXYZ>& points, std::vector<float>& distances)
	{
		if (distances.size() != points.size())
			distances = std::vector<float>(points.size());
		size_t i = 0;
		float inv_norm_u = 1 / std::sqrt(u.sq_norm());
		for (auto& p : points)
		{
			distances[i] = (u.dot(p) - d) * inv_norm_u;
			i++;
		}
	}

	// Method getting distances to some points
	void point_distances(std::vector<PointXYZ>& points, std::vector<float>& distances)
	{
		if (distances.size() != points.size())
			distances = std::vector<float>(points.size());
		size_t i = 0;
		float inv_norm_u = 1 / std::sqrt(u.sq_norm());
		for (auto& p : points)
		{
			distances[i] = std::abs((u.dot(p) - d) * inv_norm_u);
			i++;
		}
	}
	
	int in_range(std::vector<PointXYZ>& points, float threshold)
	{
		int count = 0;
		float inv_norm_u = 1 / std::sqrt(u.sq_norm());
		for (auto& p : points)
		{
			if (std::abs((u.dot(p) - d) * inv_norm_u) < threshold)
				count++;
		}
		return count;
	}
};


//-------------------------------------------------------------------------------------------
//
// VoxKey
// ******
//
//	Here we define a struct that will be used as key in our hash map. It contains 3 integers.
//  Then we specialize the std::hash function for this class.
//
//-------------------------------------------------------------------------------------------

class VoxKey
{
public:
	int x;
	int y;
	int z;

	VoxKey()
	{
		x = 0;
		y = 0;
		z = 0;
	}
	VoxKey(int x0, int y0, int z0)
	{
		x = x0;
		y = y0;
		z = z0;
	}

	bool operator==(const VoxKey &other) const
	{
		return (x == other.x && y == other.y && z == other.z);
	}

	int& operator[](std::size_t idx)       
	{
		if (idx == 2) return z;
		else if (idx == 1) return y;
		else  return x;
	}
    const int& operator[](std::size_t idx) const
	{
		if (idx == 2) return z;
		else if (idx == 1) return y;
		else return x;
	}

	void update_min(const VoxKey &k0)
	{
		if (k0.x < x)
			x = k0.x;
		if (k0.y < y)
			y = k0.y;
		if (k0.z < z)
			z = k0.z;
	}

	void update_max(const VoxKey &k0)
	{
		if (k0.x > x)
			x = k0.x;
		if (k0.y > y)
			y = k0.y;
		if (k0.z > z)
			z = k0.z;
	}
};

inline VoxKey operator+(const VoxKey A, const VoxKey B)
{
	return VoxKey(A.x + B.x, A.y + B.y, A.z + B.z);
}

inline bool operator<(const VoxKey A, const VoxKey B)
{
	if (A.x == B.x)
	{
		if (A.y == B.y)
			return (A.z < B.z);
		else
			return (A.y < B.y);
	}
	else
		return (A.x < B.x);
}

inline VoxKey operator-(const VoxKey A, const VoxKey B)
{
	return VoxKey(A.x - B.x, A.y - B.y, A.z - B.z);
}

// Simple utility function to combine hashtables
template <typename T, typename... Rest>
void hash_combine(std::size_t &seed, const T &v, const Rest &...rest)
{
	seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	(hash_combine(seed, rest), ...);
}

// Specialization of std:hash function
namespace std
{
	template <>
	struct hash<VoxKey>
	{
		std::size_t operator()(const VoxKey &k) const
		{
			std::size_t ret = 0;
			hash_combine(ret, k.x, k.y, k.z);
			return ret;
		}
	};
}


//-------------------------------------------------------------------------------------------
//
// PixKey
// ******
//
//	Same as VoxKey but in 2D
//
//-------------------------------------------------------------------------------------------

class PixKey
{
public:
	int x;
	int y;

	PixKey()
	{
		x = 0;
		y = 0;
	}
	PixKey(int x0, int y0)
	{
		x = x0;
		y = y0;
	}

	bool operator==(const PixKey &other) const
	{
		return (x == other.x && y == other.y);
	}
};

inline PixKey operator+(const PixKey A, const PixKey B)
{
	return PixKey(A.x + B.x, A.y + B.y);
}

inline PixKey operator-(const PixKey A, const PixKey B)
{
	return PixKey(A.x - B.x, A.y - B.y);
}

// Specialization of std:hash function
namespace std
{
	template <>
	struct hash<PixKey>
	{
		std::size_t operator()(const PixKey &k) const
		{
			std::size_t ret = 0;
			hash_combine(ret, k.x, k.y);
			return ret;
		}
	};
} // namespace std

