#ifndef _STEREO_H_
#define _STEREO_H_
#include <stdio.h>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "PatchMatch.h"


class Stereo
{
public:
	enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3, STEREO_PM=4 };

	Stereo(int stereo_type, const std::string &exfilename, const std::string &infilename, const std::string &configname);
	~Stereo(){delete bm; delete sgbm;};
	void init();
	void setInputImages (cv::Mat &leftImg, cv::Mat &rightImg);
	void setRectImages (cv::Mat &l_rect_img, cv::Mat &r_rect_img);
	cv::Mat& calcDisparityMap();
	cv::Mat& calcDisparityMap(cv::Mat &leftImg, cv::Mat &rightImg);
	cv::Mat& getDisparityMap();
	cv::Mat& getnormDisparityMap();
	cv::Mat& getsmoothDisparityMap();
	cv::Mat& getR1();
	cv::Mat& get3dImage();
	cv::Rect& getLeftROI();
	cv::Mat& getcameraMatrix(const int lr = 0);
	cv::Mat& getdistCoeffs(const int lr = 0);
	void getRectifiedImage(cv::Mat &iLeftImg, cv::Mat &iRightImg);
	void calc3DPoints();
	void projectDepthTo3D(const cv::Mat& depth_map, cv::Mat &_3DImage, const cv::Mat& camMat);

	void saveXYZ(const char* filename, const cv::Mat& mat);
	void saveXYZLR(const char* filename, const cv::Mat& _3dmat, const cv::Mat& disparity);
	void saveXYZRGB(const char* filename, const cv::Mat& _3dmat, const cv::Mat& color_map);
	void saveXYZRGBLR(const char* filename, const cv::Mat& _3dmat, const cv::Mat& color_map, const cv::Mat& disparity);

	void save3DToTXT(const std::string &filename);
	void save3D2DToTXT(const std::string &filename);
	void save3DRGBToTXT(const std::string &filename);
	void save3DRGB2DToTXT(const std::string &filename);

	void saveConfig(const std::string &filename);
protected:
private:
	// Image size
	cv::Size image_size;
	// Pattern width
	int pattern_width;
	// Pattern height
	int pattern_height;
	// Camera parameters
	cv::Mat cameraMatrix[2];
	cv::Mat distCoeffs[2];
	cv::Mat R, T, R1, R2, P1, P2, Q;
	// Albedo
	cv::Mat albedo[2];

	// Stereo Matching type
	cv::StereoBM *bm;
	cv::StereoSGBM *sgbm;
	CStereoPM *pm;
	// Parameters for stereo matching
	int match_type;
	int SADWindowSize;
	int numberOfDisparities;
	cv::Rect roi1, roi2;
	int preFilterCap;
	int minDisparity;
	int textureThreshold;
	int uniquenessRatio;
	// Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range
	int speckleWindowSize;
	// Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough
	int speckleRange;
	// Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check
	int disp12MaxDiff;
	// full-scale two-pass dynamic programming
	bool fullDP;

	// Raw left and right image
	cv::Mat leftImage, rightImage;
	// undistorted image
	cv::Mat recLeftImage, recRightImage;
	// rectification map
	cv::Mat rmap[2][2];
	// Raw Disparity Image
	cv::Mat rawDisparity;
	// Modified Disparity Image
	cv::Mat modDisparity;
	// Spreaded Disparity for visualisation;
	cv::Mat normDisparity;
	// Smoothed Disparity for denoising;
	cv::Mat smoothDisparity;
	// 3D image
	cv::Mat _3dImage;
	// Depth Map from stereo
	cv::Mat depth_stereo;

	void writeMatToFile(cv::Mat& m, const char* filename);
	void smooth3DWithMask(cv::Mat &src_disp, double thresh, cv::Size win_size = cv::Size(3,3));
	void removeOutlierDisparity(cv::Rect region, cv::Mat &disparity);
};
#endif 