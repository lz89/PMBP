#pragma once
#ifndef _PATCH_H_
#define _PATCH_H_





#include <opencv2/opencv.hpp>

class CPatch
{
	
public:
	enum IMAGE_SIDE { A, B };
	IMAGE_SIDE side;

	CPatch(int px, int py, int maxdisp, IMAGE_SIDE side);
	CPatch(float a, float b, float c, int maxdisp);
	CPatch(int px, int py, double initZ, int maxdisp, IMAGE_SIDE side);
	// Overload assignment operator
	CPatch& operator=(CPatch copy_Patch);
	void swap(CPatch& other);

	static CPatch fromCoarser(const CPatch &coarse_patch, int currPx, int currPy);
	void setPlaneParams(float a, float b, float c) {_a = a; _b = b; _c = c;}
	float disparity();
	float disparity(int ix, int iy);
	cv::Point3f normalForm();
	cv::Point2i pixelCoord();
	void setPatch(double nx, double ny, double nz, double z0);
	// Plane params
	float _a, _b, _c;
	int _px, _py;
	double _max_disp;

private:
	static cv::RNG rng_z0_A, rng_nx_A, rng_ny_A, rng_nz_A;
	static cv::RNG rng_z0_B, rng_nx_B, rng_ny_B, rng_nz_B;

	// Randomly initialize the patch parameters
	void Init(double iz = -1);

};





#endif