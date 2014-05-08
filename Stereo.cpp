#include "Stereo.h"
#include <limits>
Stereo::Stereo(int stereo_type, const std::string &exfilename, const std::string &infilename, const std::string &configname):match_type(stereo_type)
{
	// Read camera parameters
	cv::FileStorage fs;
	if (!fs.open(exfilename, cv::FileStorage::READ))
	{
		std::cerr << "cannot open: " << exfilename << std::endl;
	}
	fs["R"] >> R; fs["T"] >> T;
	fs.release();
	if (!fs.open(infilename, cv::FileStorage::READ))
	{
		std::cerr << "cannot open: " << infilename << std::endl;
	}
	fs["M1"] >> cameraMatrix[0]; fs["D1"] >> distCoeffs[0];
	fs["M2"] >> cameraMatrix[1]; fs["D2"] >> distCoeffs[1];

	// init parameters
	if (!fs.open(configname, cv::FileStorage::READ))
	{
		std::cerr << "cannot open: " << configname << std::endl;
	}
	if (match_type == STEREO_BM)
	{
		bm = new cv::StereoBM;
		sgbm = NULL;
		pm = NULL;
		bm->state->textureThreshold = 16;
	}
	else if (match_type == STEREO_SGBM)
	{
		sgbm = new cv::StereoSGBM;
		bm = NULL;
		pm = NULL;
		fs["SADWindowSize"] >> sgbm->SADWindowSize;
		fs["numberOfDisparities"] >> sgbm->numberOfDisparities;
		fs["preFilterCap"] >> sgbm->preFilterCap;
		fs["minDisparity"] >> sgbm->minDisparity;
		fs["uniquenessRatio"] >> sgbm->uniquenessRatio;
		fs["speckleWindowSize"] >> sgbm->speckleWindowSize;
		sgbm->speckleRange = 32;
		sgbm->disp12MaxDiff = 1;
		fs["fullDP"] >> sgbm->fullDP;
	}
	else if (match_type == STEREO_PM)
	{
		pm = new CStereoPM;
		sgbm = NULL;
		bm = NULL;
		fs["Iter"] >> pm->Iter;
		fs["WindowSize"] >> pm->WindowSize;
		fs["maxDisparity"] >> pm->maxDisparity;
		fs["NumOfLayers"] >> NumOfLayers;
	}

	init();
	fs.release();
}

void Stereo::init()
{
	pattern_width = 8;
	pattern_height = 6;
	roi1 = cv::Rect(0,0,0,0);
	roi2 = cv::Rect(0,0,0,0);
}

void Stereo::setInputImages(cv::Mat &leftImg, cv::Mat &rightImg)
{
	leftImage = leftImg;
	rightImage = rightImg;
	image_size = leftImage.size();
	stereoRectify( cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], image_size, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1, image_size, &roi1, &roi2 );
	//Precompute maps for cv::remap()
	initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, image_size, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, image_size, CV_16SC2, rmap[1][0], rmap[1][1]);

	modDisparity.create(image_size, CV_32FC1);
	smoothDisparity.create(image_size, CV_32FC1);
}

cv::Mat& Stereo::calcDisparityMap()
{
	int color_mode = match_type == STEREO_SGBM ? 1:0;
	if (recLeftImage.empty())
	{
		// BM can only process 8bit grayscale image
		if (!color_mode && leftImage.channels() > 1)
		{
			cv::cvtColor(leftImage, leftImage, CV_BGR2GRAY);
			cv::cvtColor(rightImage, rightImage, CV_BGR2GRAY);
		}
		// RECTIFICATION
		cv::remap(leftImage, recLeftImage, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
		cv::remap(rightImage, recRightImage, rmap[1][0], rmap[1][1], cv::INTER_LINEAR);
	}

	cv::Mat recLeftImageBorder, recRightImageBorder;
	// Enlarge boarder (for SGBM)
	if (match_type == STEREO_SGBM)
	{
		cv::copyMakeBorder(recLeftImage, recLeftImageBorder, 0, 0, sgbm->numberOfDisparities, 0, IPL_BORDER_REPLICATE);
		cv::copyMakeBorder(recRightImage, recRightImageBorder, 0, 0, sgbm->numberOfDisparities, 0, IPL_BORDER_REPLICATE);
		// 	cv::imwrite("leftRecBorderImg.jpg",recLeftImageBorder);
		// 	cv::imwrite("rightRecBorderImg.jpg",recRightImageBorder);

	}
	
	if (match_type == STEREO_BM)
	{
		bm->state->roi1 = roi1;
		bm->state->roi2 = roi2;
		bm->state->preFilterCap = preFilterCap;
		bm->state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 9;
		bm->state->minDisparity = minDisparity;
		bm->state->numberOfDisparities = numberOfDisparities;
		bm->state->uniquenessRatio = uniquenessRatio;
		bm->state->speckleWindowSize = speckleWindowSize;
		bm->state->speckleRange = speckleRange;
		bm->state->disp12MaxDiff = disp12MaxDiff;

		(*bm)(recLeftImageBorder, recRightImageBorder, rawDisparity);
		rawDisparity = rawDisparity.colRange(numberOfDisparities, recLeftImageBorder.cols);
		modDisparity = rawDisparity / 16;
	}
	else if (match_type == STEREO_SGBM)
	{
		int cn = recLeftImage.channels();

		sgbm->P1 = 8*cn*sgbm->SADWindowSize*sgbm->SADWindowSize;
		sgbm->P2 = 32*cn*sgbm->SADWindowSize*sgbm->SADWindowSize;
// 		(*sgbm)(recLeftImage, recRightImage, rawDisparity);
		(*sgbm)(recLeftImageBorder, recRightImageBorder, rawDisparity);
		rawDisparity = rawDisparity.colRange(sgbm->numberOfDisparities, recLeftImageBorder.cols);
		rawDisparity.convertTo(rawDisparity, CV_32FC1);
		modDisparity = rawDisparity / 16.0f;
	}
	else if (match_type == STEREO_PM)
	{
		if (NumOfLayers > 1)
			(*pm)(recLeftImage, recRightImage, rawDisparity, NumOfLayers);
		else
			(*pm)(recLeftImage, recRightImage, rawDisparity);
		modDisparity = rawDisparity;
	}

	// Get a normalized disparity for display
	double max, min;
	cv::minMaxLoc(modDisparity, &min, &max);
	normDisparity = ((modDisparity - min) * (255-0)) / (max - min) + 0;
	cv::rectangle(normDisparity, roi1, cv::Scalar(255, 0, 0));
	return modDisparity;
}

cv::Mat& Stereo::calcDisparityMap(cv::Mat &leftImg, cv::Mat &rightImg)
{
	setInputImages(leftImg, rightImg);
	return (calcDisparityMap());
}

cv::Mat& Stereo::getDisparityMap()
{
	return modDisparity;
}

cv::Mat& Stereo::getnormDisparityMap()
{
	return normDisparity;
}

cv::Mat& Stereo::getsmoothDisparityMap()
{
	return smoothDisparity;
}

cv::Mat& Stereo::getR1()
{
	return R1;
}

cv::Mat& Stereo::get3dImage()
{
	return _3dImage;
}

cv::Rect& Stereo::getLeftROI()
{
	return roi1;
}

cv::Mat& Stereo::getcameraMatrix(const int lr /* = 0 */)
{
	return cameraMatrix[lr];
}

cv::Mat& Stereo::getdistCoeffs(const int lr /* = 0 */)
{
	return distCoeffs[lr];
}

void Stereo::setRectImages(cv::Mat &l_rect_img, cv::Mat &r_rect_img)
{
	recLeftImage = l_rect_img;
	recRightImage = r_rect_img;
}

void Stereo::getRectifiedImage(cv::Mat &iLeftImg, cv::Mat &iRightImg)
{
	iLeftImg = recLeftImage;
	iRightImg = recRightImage;
}

void Stereo::calc3DPoints()
{
	if (modDisparity.empty()) //modDisparity
	{
		std::cout << "ERROR: modDisparity is empty" << std::endl;
	}
// 	cv::reprojectImageTo3D(modDisparity(roi1), _3dImage, Q, true);
	cv::reprojectImageTo3D(modDisparity, _3dImage, Q, true);
	
}


void Stereo::projectDepthTo3D(const cv::Mat& depth_map, cv::Mat& points3d, const cv::Mat& camMat)
{
	float f = 0.5 * (camMat.at<double>(0,0) + camMat.at<double>(1,1));
	float cx = camMat.at<double>(0,2);
	float cy = camMat.at<double>(1,2);
	int ux_method = 2;
	if (ux_method == 1)
	{
		// First: if M = u(x)*m
		for ( int i = 0; i < depth_map.rows; i++)
		{
			for (int j = 0; j < depth_map.cols; j++)
			{
				float px = j - cx;
				float py = i - cy;
				// 			float m = sqrt(px*px + py*py + f*f);
				float ux = depth_map.at<float>(i, j);
				points3d.at<cv::Vec3f>(i, j)[0] = 4 * (j - cx) * ux;
				points3d.at<cv::Vec3f>(i, j)[1] = 4 * (i - cy) * ux;
				points3d.at<cv::Vec3f>(i, j)[2] = 4 * f * ux;
			}
		}
	}
	else if (ux_method == 2)
	{
		// Second: if u(x) is distance b/w optical centre and M
		for ( int i = 0; i < depth_map.rows; i++)
		{
			for (int j = 0; j < depth_map.cols; j++)
			{
				float px = j - cx;
				float py = i - cy;
	 			float m = sqrt(px*px + py*py + f*f);
				float ux = depth_map.at<float>(i, j);
				points3d.at<cv::Vec3f>(i, j)[0] = px * ux / m;
				points3d.at<cv::Vec3f>(i, j)[1] = py * ux / m;
				points3d.at<cv::Vec3f>(i, j)[2] = f * ux / m;
			}
		}
	}
	else if (ux_method == 3)
	{
		// Third: if u(x) = Z, M(X, X, Z)
		for ( int i = 0; i < depth_map.rows; i++)
		{
			for (int j = 0; j < depth_map.cols; j++)
			{
				float px = j - cx;
				float py = i - cy;
				float ux = depth_map.at<float>(i, j) + 50;
				points3d.at<cv::Vec3f>(i, j)[0] = px * ux / f;
				points3d.at<cv::Vec3f>(i, j)[1] = py * ux / f;
				points3d.at<cv::Vec3f>(i, j)[2] = ux;
			}
		}
	}
	else if (ux_method == 4)
	{
		// Forth: if u(x) is distance between m and M(X, X, Z)
		for ( int i = 0; i < depth_map.rows; i++)
		{
			for (int j = 0; j < depth_map.cols; j++)
			{
				float px = j - cx;
				float py = i - cy;
				float m = sqrt(px*px + py*py + f*f);
				
				float ux = depth_map.at<float>(i, j);
				points3d.at<cv::Vec3f>(i, j)[0] = (f / m) * px * ux;
				points3d.at<cv::Vec3f>(i, j)[1] = (f / m) * py * ux;
				points3d.at<cv::Vec3f>(i, j)[2] = (f / m) * f * ux;
			}
		}
	}
}


void Stereo::saveXYZ(const char* filename, const cv::Mat& mat)
{
	const double max_z = 1.0e4;
	FILE* fp = fopen(filename, "wt");
	for(int y = 0; y < mat.rows; y++)
	{
		for(int x = 0; x < mat.cols; x++)
		{
			cv::Vec3f point = mat.at<cv::Vec3f>(y, x);
			if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			fprintf(fp, "%f %f %f ", point[0], point[1], point[2]);
		}
	}
	fclose(fp);
}

void Stereo::saveXYZLR(const char* filename, const cv::Mat& _3dmat, const cv::Mat& disparity)
{
	const double max_z = 1.0e4;
	FILE* fp = fopen(filename, "wt");
	fprintf(fp, "X Y Z lx ly rx ry\n");
	for(int y = 0; y < _3dmat.rows; y++)
	{
		for(int x = 0; x < _3dmat.cols; x++)
		{
			cv::Vec3f point = _3dmat.at<cv::Vec3f>(y, x);
			if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			fprintf(fp, "%f %f %f ", point[0], point[1], point[2]);
			int d, lx, ly, rx, ry;
			d = disparity.at<short>(y, x);
			ly = ry = y;	// Rectified imgs are row (y) aligned
			lx = x;			// left img is reference
			rx = lx - d;
			fprintf(fp, "%d %d %d %d\n", lx, ly, rx, ry);
		}
	}
	fclose(fp);
}

void Stereo::saveXYZRGB(const char* filename, const cv::Mat& _3dmat, const cv::Mat& color_map)
{
	const double max_z = 500;
	FILE* fp = fopen(filename, "wt");
	fprintf(fp, "X Y Z R G B lx ly\n");
	for(int y = 0; y < _3dmat.rows; y++)
	{
		for(int x = 0; x < _3dmat.cols; x++)
		{
			cv::Vec3f point = _3dmat.at<cv::Vec3f>(y, x);
			cv::Vec3b c = color_map.at<cv::Vec3b>(y, x);
			if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			if(c[0] < 1 && c[1] < 1 && c[2] <1) continue;
			fprintf(fp, "%f %f %f ", point[0], point[1], point[2]);
			fprintf(fp, "%d %d %d ", c[2], c[1], c[0]);
			fprintf(fp, "%d %d\n", x, y);
		}
	}
	fclose(fp);
}

void Stereo::saveXYZRGBLR(const char* filename, const cv::Mat& _3dmat, const cv::Mat& color_map, const cv::Mat& disparity)
{
	const double max_z = 1.0e3;
	const double min_z = 1.0;
	cv::Mat hsv_map;
	cv::cvtColor(color_map,hsv_map,CV_BGR2HSV);
	FILE* fp = fopen(filename, "wt");
	fprintf(fp, "X Y Z R G B lx ly rx ry\n");
	for(int y = 0; y < _3dmat.rows; y++)
	{
		for(int x = 0; x < _3dmat.cols; x++)
		{
			cv::Vec3f point = _3dmat.at<cv::Vec3f>(y, x);
			if( point[2] < min_z || point[2] > max_z) continue;
			cv::Vec3b h = color_map.at<cv::Vec3b>(y, x);
			if(h[2] < 10) continue;
			cv::Vec3b c = color_map.at<cv::Vec3b>(y, x);
// 			if(c[0] < 1 && c[1] < 1 && c[2] <1) continue;

			fprintf(fp, "%f %f %f ", point[0], point[1], point[2]);
			fprintf(fp, "%d %d %d ", c[2], c[1], c[0]);

			int d, lx, ly, rx, ry;
			d = disparity.at<float>(y, x);
			ly = ry = y;	// Rectified imgs are row (y) aligned
			lx = x;			// left img is reference
			rx = lx - d;
			if (lx < 0) lx = 0;
			if (rx < 0) rx = 0;
			cv::Vec2s ori0 = rmap[0][0].at<cv::Vec2s>(ly, lx);
			cv::Vec2s ori1 = rmap[1][0].at<cv::Vec2s>(ry, (short)rx);

			fprintf(fp, "%d %d %d %d\n", ori0[0], ori0[1], ori1[0], ori1[1]);
		}
	}
	fclose(fp);
}

void Stereo::save3DToTXT(const std::string &filename)
{
	saveXYZ(filename.c_str(), _3dImage);
}

void Stereo::save3D2DToTXT(const std::string &filename)
{
	saveXYZLR(filename.c_str(), _3dImage, modDisparity);
}

void Stereo::save3DRGBToTXT(const std::string &filename)
{
	saveXYZRGB(filename.c_str(), _3dImage, recLeftImage(roi1));
}

void Stereo::save3DRGB2DToTXT(const std::string &filename)
{
	saveXYZRGBLR(filename.c_str(), _3dImage, recLeftImage, modDisparity);
}

void Stereo::saveConfig(const std::string &filename)
{
	cv::FileStorage f(filename, CV_STORAGE_WRITE);
	if( f.isOpened() )
	{
		f << "SADWindowSize" << SADWindowSize << "numberOfDisparities" << numberOfDisparities;
		f << "preFilterCap" << preFilterCap << "minDisparity" << minDisparity;
		f << "uniquenessRatio" << uniquenessRatio << "fullDP" << fullDP;
		f << "speckleWindowSize" << speckleWindowSize;
	}
	else
		std::cout << "Error: cannot open the configuration file\n";
	f.release();
}

void Stereo::writeMatToFile(cv::Mat& m, const char* filename)
{
	std::ofstream fout(filename);

	if(!fout)
	{
		std::cout << "File Not Opened" << std::endl;  return;
	}
	for (int cn = 0; cn < m.channels(); cn++)
	{
		for(int i=0; i<m.rows; i++)
		{
			for(int j=0; j<m.cols; j++)
			{
				if (m.channels() == 1)
				{
					fout<< (float)m.at<float>(i,j)<<"\t";
				}
				else if (m.channels() == 3)
				{
					fout<< (float)(m.at<cv::Vec3f>(i,j))[cn]<<"\t";
				}
			}
			fout<<std::endl;
		}
		fout << "End of channel\n";
		if (m.channels() > 1)
		{
			fout << "Start of channel:\n";
		}
	}
	

	fout.close();
}

void Stereo::smooth3DWithMask(cv::Mat &src_3d, double thresh, cv::Size win_size)
{
	// 1. Get mask based on depth (z)
	cv::Mat mask8;
	cv::Mat z (src_3d.size(), CV_32FC1);
// 	std::cout << src_3d.size() << std::endl;
	int from_to[] = { 2, 0 };
	cv::mixChannels(&src_3d, 1, &z, 1, from_to, 1);
	cv::threshold(z, mask8, thresh, 255, cv::THRESH_BINARY_INV);
	mask8.convertTo(mask8, CV_8UC1);
// 	writeMatToFile(mask32, "mask.txt");
// 	writeMatToFile(z, "z.txt");
	// 2. Filter the depth map based on the mask
	cv::Mat blur_xyz (src_3d);
	assert( mask8.rows == blur_xyz.rows && mask8.cols == blur_xyz.cols);
	for (int x = 0; x < blur_xyz.cols; x++)
	{
		for ( int y = 0; y < blur_xyz.rows; y++)
		{
			if (mask8.at<uchar>(y, x) == 0)
			{
				for ( int i = 0; i < blur_xyz.channels(); i++)
					blur_xyz.at<cv::Vec3f>(y, x)[i] = 0;
			}
		}
	}
	// 3. Apply Smoothing filter supplying with the mask
	std::vector<cv::Mat> xyz;
	cv::split(blur_xyz, xyz);
	cv::medianBlur(mask8, mask8, win_size.height);
// 	for (unsigned int i = 0; i < xyz.size(); i++)
// 		cv::medianBlur(xyz[i], xyz[i], win_size.height);
	cv::medianBlur(xyz[0], xyz[0], win_size.height);
	cv::merge(xyz, blur_xyz);
	
	for ( int x = 0; x < mask8.cols; x++)
	{
		for ( int y = 0; y < mask8.rows; y++)
		{
			if(mask8.at<uchar>(y, x) > 0)
			{
// 				for ( unsigned int i = 0; i < blur_xyz.channels(); i++)
// 				{
					src_3d.at<cv::Vec3f>(y,x)[0] = (blur_xyz.at<cv::Vec3f>(y,x)[0] / mask8.at<uchar>(y,x)) * 255;
					if (_isnan(src_3d.at<cv::Vec3f>(y,x)[0]))
					{
						std::cout << blur_xyz.at<cv::Vec3f>(y,x)[0] <<
							"," << blur_xyz.at<cv::Vec3f>(y,x)[1] <<
							"," << blur_xyz.at<cv::Vec3f>(y,x)[2] << std::endl;
						std::cout <<  mask8.at<uchar>(y,x) << std::endl;
					}
// 				}
			}
		}
	}
// 	writeMatToFile(src_3d, "src_3d.txt");
}

void Stereo::removeOutlierDisparity(cv::Rect region, cv::Mat &disparity)
{
	for (int i = region.y; i < region.y + region.height; i++)
	{
		for (int j = region.x; j < region.x + region.width; j++)
		{
			modDisparity.at<float>(i, j) = -1;
		}
	}
}