#pragma once
#ifndef _COST_FUNCTION_H_
#define _COST_FUNCTION_H_

#include <opencv2/opencv.hpp>

class CPatch;

class CCostFunction
{
public:
	CCostFunction (const cv::Mat &aimg, const cv::Mat &bimg, cv::Size winSize, float color_weight, float grad_balance, float t_col, float t_grad);

	double calcCost_A(const cv::Point2i in_pixel, CPatch &in_patch);
	double calcCost_B(const cv::Point2i in_pixel, CPatch &in_patch);

private:
// 	cv::Mat _aimg, _bimg;
	cv::Size _img_size;
	cv::Mat _aimg_exp, _bimg_exp;
	// Abs image gradient in X direction, image has same size as original image (i.e. no copy border)
	cv::Mat _aimg_gradX, _bimg_gradX;
	cv::Size _winSize;
	int _offsetX, _offsetY;
	float _color_weight;
	float _grad_balance;
	float _t_col, _t_grad;
};


#endif