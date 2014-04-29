#include "Patch.h"

cv::RNG CPatch::rng_z0_A(cv::getTickCount() + 1);
cv::RNG CPatch::rng_nx_A(cv::getTickCount() + 10);
cv::RNG CPatch::rng_ny_A(cv::getTickCount() + 100);
cv::RNG CPatch::rng_nz_A(cv::getTickCount() + 1000);
cv::RNG CPatch::rng_z0_B(cv::getTickCount() + 10000);
cv::RNG CPatch::rng_nx_B(cv::getTickCount() + 100000);
cv::RNG CPatch::rng_ny_B(cv::getTickCount() + 1000000);
cv::RNG CPatch::rng_nz_B(cv::getTickCount() + 10000000);


CPatch::CPatch(int px, int py, int maxdisp, IMAGE_SIDE i_side): 
_px(px), _py(py), _max_disp(maxdisp), side(i_side)
{
	Init();
}

CPatch::CPatch(float a, float b, float c, int maxdisp) : _a(a), _b(b), _c(c)
{}

float CPatch::disparity()
{
	return _a * _px + _b * _py + _c;
}

float CPatch::disparity(int ix, int iy)
{
	return _a * ix + _b * iy + _c;
}

cv::Point3f CPatch::normalForm()
{
	cv::Point3f temp_normal;
	float norm = sqrt(_a*_a + _b*_b + _c*_c);
	temp_normal.x = _a / norm;
	temp_normal.y = _b / norm;
	temp_normal.z = _c / norm;
	return temp_normal;
}

cv::Point2i CPatch::pixelCoord()
{
	return cv::Point2i(_px, _py);
}

void CPatch::setPatch(double nx, double ny, double nz, double z0)
{
	// Convert normal vector to a, b, c
	_a = -nx / nz;
	_b = -ny / nz;
	_c = (nx*_px + ny*_py + nz*z0) / nz;
}

void CPatch::Init()
{	
	bool flag = false;
	double nx, ny, nz, z0;
	if (side == A)
	{
		z0 = rng_z0_A.uniform(0.0, _max_disp);
		nx = rng_nx_A.gaussian(1);
		ny = rng_ny_A.gaussian(1);
		nz = rng_nz_A.gaussian(1);
	}
	else
	{
		z0 = rng_z0_B.uniform(0.0, _max_disp);
		nx = rng_nx_B.gaussian(1);
		ny = rng_ny_B.gaussian(1);
		nz = rng_nz_B.gaussian(1);
	}

	// Generate unit vector pointing to uniformly distributed direction
	// Method 1
// 	double k1, k2;
// 	cv::RNG rng_k1(cv::getTickCount());
// 	cv::RNG rng_k2(cv::getTickCount());
// 	while (flag)
// 	{
// 		// Marsaglia (1972) Ref:http://mathworld.wolfram.com/SpherePointPicking.html
// 		k1 = rng_k1.uniform(-1.0, 1.0);
// 		k2 = rng_k2.uniform(-1.0, 1.0);
// 		if (k1*k1 + k2*k2 >= 1)
// 			continue;
// 		nx = 2 * k1 * sqrt(1 - k1*k1 - k2*k2);
// 		ny = 2 * k2 * sqrt(1 - k1*k1 - k2*k2);
// 		nz = 1 - 2 * (k1*k1 + k2*k2);
// 		flag = true;
// 	}
	// Method 2

	
	double length = sqrt(nx*nx + ny*ny + nz*nz);
	nx /= length;
	ny /= length;
	nz /= length;

	// Convert normal vector to a, b, c
	setPatch(nx, ny, nz, z0);
}