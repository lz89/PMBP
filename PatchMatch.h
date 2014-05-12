#pragma once
#ifndef _PATCH_MATCH_H_
#define _PATCH_MATCH_H_

#include <opencv2/opencv.hpp>
#include "Patch.h"
#include "CostFunction.h"

class CStereoPM
{
public:
	CStereoPM();

	void operator()(cv::Mat &left, cv::Mat &right, cv::Mat &disp);
	void operator()(cv::Mat &left, cv::Mat &right, cv::Mat &disp, int layer);
	int Iter;
	int WindowSize;
	// Maximum disparity
	int maxDisparity;

private:
	// Input left/right images
	cv::Mat Left_Img, Right_Img;
	// Current processing (down-sampled) image
	cv::Mat currLImage, currRImage;
	// Patches for image
	std::vector<CPatch> Patch_Img_A, Patch_Img_B;
	std::vector<CPatch> Patch_Img_Pre_A, Patch_Img_Pre_B; 
	// Disparity image
	cv::Mat _disp_A, _disp_B;
	cv::Mat _f_disp_A, _f_disp_B;
	// Image size
	cv::Size _imgSize;
	// flag for multi-scale PatchMatch
	bool _prevPM_exist;
	
	cv::RNG rng_refine;
	cv::Ptr<CCostFunction> _cost;
	// Initialize patch for each pixel
	void initPatch(int scale = 1);
	// Update disparity image based on current Patch_Img
	void updateDisp();
	void updateDispA();
	void updateDispB();
	void updateFloatDisp();

	// Plane refinement step
	void planeRefine(CPatch & curr_ptch);
};


#endif