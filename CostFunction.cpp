#include "CostFunction.h"
#include "Patch.h"

CCostFunction::CCostFunction(const cv::Mat &aimg, const cv::Mat &bimg, cv::Size winSize, float color_weight, 
	float grad_balance, float t_col, float t_grad):
	_winSize(winSize), _color_weight(color_weight), _grad_balance(grad_balance), _t_col(t_col), _t_grad(t_grad)
{
	_img_size.height = aimg.rows;
	_img_size.width = aimg.cols;

	int top, bottom, left, right;
	top = _winSize.height / 2;
	bottom = top;
	left = _winSize.width / 2;
	right = left;
	copyMakeBorder(aimg, _aimg_exp, top, bottom, left, right, cv::BORDER_CONSTANT, 0);
	copyMakeBorder(bimg, _bimg_exp, top, bottom, left, right, cv::BORDER_CONSTANT, 0);

	_offsetX = _winSize.width / 2;
	_offsetY = _winSize.height / 2;
	
	// Create gradient image for a,b views
	Sobel(aimg, _aimg_gradX, CV_32F, 1, 0, 1, 1, 0, cv::BORDER_DEFAULT);
	_aimg_gradX = cv::abs(_aimg_gradX);

	Sobel(bimg, _bimg_gradX, CV_32F, 1, 0, 1, 1, 0, cv::BORDER_DEFAULT);
	_bimg_gradX = cv::abs(_bimg_gradX);
}

double CCostFunction::calcCost_A(const cv::Point2i in_pixel, CPatch &in_patch)
{
	// Each central pixel, calculate its neighbors
	double cost = 0;
	uchar p_i = _aimg_exp.at<uchar>(in_pixel);
	uchar q_i = 0;
	uchar num_valid_neighbor = 0;
	for (int qy = in_pixel.y - _offsetY; qy < in_pixel.y + _offsetY; qy++)
	{
		for (int qx = in_pixel.x - _offsetX; qx < in_pixel.x + _offsetX; qx++)
		{
			q_i = _aimg_exp.at<uchar>(qy + _offsetY, qx + _offsetX);
			if (q_i == 0)
				continue;
			// Eq(4)
			float w = exp( -abs(q_i - p_i) / _color_weight);
			// Eq(5)
			float p = 0;
			float disp = in_patch.disparity(qx, qy);
			int qbx = cvRound(qx - disp);
			if (qbx < 0 || disp < 0) // Invisible for view b
			{
				p = (1 - _grad_balance) *  _t_col + _grad_balance * _t_grad;
			}
			else
			{
				int qby = qy;
				uchar qb_i = _bimg_exp.at<uchar>(qby + _offsetY, qbx + _offsetX);
				float q_g = _aimg_gradX.at<float>(qy, qx);
				float qb_g = _bimg_gradX.at<float>(qby, qbx);

				p = (1 - _grad_balance) * MIN(abs(q_i - qb_i), _t_col) + 
					_grad_balance * MIN(abs(q_g - qb_g), _t_grad);
// 				std::cout << q_i-qb_i << ",";
			}
			// Eq(3)
			cost += w * p;
			num_valid_neighbor++;
		}
	}
	cost /= num_valid_neighbor;

	return cost;
}

double CCostFunction::calcCost_B(const cv::Point2i in_pixel, CPatch &in_patch)
{
	// Each central pixel, calculate its neighbors
	double cost = 0;
	uchar p_i = _bimg_exp.at<uchar>(in_pixel);
	uchar q_i = 0;
	uchar num_valid_neighbor = 0;
	for (int qy = in_pixel.y - _offsetY; qy < in_pixel.y + _offsetY; qy++)
	{
		for (int qx = in_pixel.x - _offsetX; qx < in_pixel.x + _offsetX; qx++)
		{
			q_i = _bimg_exp.at<uchar>(qy + _offsetY, qx + _offsetX);
			if (q_i == 0)
				continue;
			// Eq(4)
			float w = exp( -abs(q_i - p_i) / _color_weight);
			// Eq(5)
			float p = 0;
			float disp = in_patch.disparity(qx, qy);
			int qax = cvRound(qx + disp);//
			if (qax >= _img_size.width || disp < 0) // Invisible for view A
			{
				p = (1 - _grad_balance) *  _t_col + _grad_balance * _t_grad;
			}
			else
			{
				int qay = qy;
				uchar qa_i = _aimg_exp.at<uchar>(qay + _offsetY, qax + _offsetX);
				float q_g = _bimg_gradX.at<float>(qy, qx);
				float qa_g = _aimg_gradX.at<float>(qay, qax);

				p = (1 - _grad_balance) * MIN(abs(q_i - qa_i), _t_col) + 
					_grad_balance * MIN(abs(q_g - qa_g), _t_grad);
				// 				std::cout << q_i-qb_i << ",";
			}
			// Eq(3)
			cost += w * p;
			num_valid_neighbor++;
		}
	}
	cost /= num_valid_neighbor;

	return cost;
}