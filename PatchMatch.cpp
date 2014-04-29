#include "PatchMatch.h"

int disp_num = 0;
int disp_numA = 0;
int disp_numB = 0;

CStereoPM::CStereoPM() : Iter(3), WindowSize(35), maxDisparity(224), rng_refine(cv::getTickCount())
{
}


// static void computeDisparityPM( const cv::Mat& img1, const cv::Mat& img2,
// 	cv::Mat disp1)
// {
// 
// }

void CStereoPM::operator()(cv::Mat &left, cv::Mat &right, cv::Mat &disp)
{
	// Initialization
	Left_Img = left.clone();
	Right_Img = right.clone();
	assert(Left_Img.size == Right_Img.size && Left_Img.type() == Right_Img.type());
	_imgSize.width = Left_Img.cols;
	_imgSize.height = Left_Img.rows;
	if (Left_Img.channels() == 3)
		cvtColor(Left_Img, Left_Img, CV_BGR2GRAY);
	if (Right_Img.channels() == 3)
		cvtColor(Right_Img, Right_Img, CV_BGR2GRAY);
	_disp_A.create( Left_Img.size(), CV_16S);
	_disp_B.create( Right_Img.size(), CV_16S);

	initPatch();
// 	updateDispA();
// 	updateDispB();
// 	updateDisp();

	_cost = new CCostFunction(Left_Img, Right_Img, cv::Size(WindowSize, WindowSize), 10, 0.9f, 10, 2);

	for (int iter = 0; iter < Iter; iter++)
	{
		std::cout << "It is: " << iter << " time\n";
		if (iter % 2 == 0) // odd iteration: from top-left to bottom-right
		{
			double up_cost = 0.0, left_cost = 0.0, centre_cost = 0.0;
			// Image a (odd)
			std::cout << "Odd Image A\n";
			for (int y = 1; y < _imgSize.height; y++)
			{
				for (int x = 1; x < _imgSize.width; x++)
				{
					if (Left_Img.at<uchar>(y,x) < 1) continue;	// If invalid pixel, not process it

					// Current cost
					centre_cost = _cost->calcCost_A(cv::Point2i(x, y), Patch_Img_A[y*_imgSize.width + x]);
					// Up neighbor
					up_cost = _cost->calcCost_A(cv::Point2i(x, y), Patch_Img_A[(y-1)*_imgSize.width + x]);					
					// Left neighbor
					left_cost = _cost->calcCost_A(cv::Point2i(x, y), Patch_Img_A[y*_imgSize.width + x - 1]);
					
					if (centre_cost > up_cost && left_cost > up_cost) // upper neighbor is best
					{
						Patch_Img_A[y*_imgSize.width + x] = Patch_Img_A[(y-1)*_imgSize.width + x];
					}
					else if (centre_cost > left_cost && up_cost > left_cost) // left neighbor is best
					{
						Patch_Img_A[y*_imgSize.width + x] = Patch_Img_A[y*_imgSize.width + x-1];
					}
					// View propagation
					int x_b = cvRound( x - Patch_Img_A[y*_imgSize.width + x].disparity() );
					if (x_b >= 0)
					{
						if (_cost->calcCost_B(cv::Point2i(x_b, y), Patch_Img_B[y*_imgSize.width + x_b])
							> _cost->calcCost_B(cv::Point2i(x_b, y), Patch_Img_A[y*_imgSize.width + x]))
							Patch_Img_B[y*_imgSize.width + x_b] = Patch_Img_A[y*_imgSize.width + x];
					}
					planeRefine(Patch_Img_A[y*_imgSize.width + x]);
				}
// 				updateDispA();
			}
			// Image b (odd)
			std::cout << "Odd Image B\n";
			for (int y = 1; y < _imgSize.height; y++)
			{
				for (int x = 1; x < _imgSize.width; x++)
				{
					if (Right_Img.at<uchar>(y,x) < 1) continue;	// If invalid pixel, not process it

					// Current cost
					centre_cost = _cost->calcCost_B(cv::Point2i(x, y), Patch_Img_B[y*_imgSize.width + x]);
					// Up neighbor
					up_cost = _cost->calcCost_B(cv::Point2i(x, y), Patch_Img_B[(y-1)*_imgSize.width + x]);					
					// Left neighbor
					left_cost = _cost->calcCost_B(cv::Point2i(x, y), Patch_Img_B[y*_imgSize.width + x - 1]);

					if (centre_cost > up_cost && left_cost > up_cost) // upper neighbor is best
					{
						Patch_Img_B[y*_imgSize.width + x] = Patch_Img_B[(y-1)*_imgSize.width + x];
					}
					else if (centre_cost > left_cost && up_cost > left_cost) // left neighbor is best
					{
						Patch_Img_B[y*_imgSize.width + x] = Patch_Img_B[y*_imgSize.width + x-1];
					}
					// View propagation
					int x_a = cvRound( x + Patch_Img_B[y*_imgSize.width + x].disparity() );
					if (x_a < _imgSize.width)
					{
						if (_cost->calcCost_A(cv::Point2i(x_a, y), Patch_Img_A[y*_imgSize.width + x_a])
							> _cost->calcCost_A(cv::Point2i(x_a, y), Patch_Img_B[y*_imgSize.width + x]))
							Patch_Img_A[y*_imgSize.width + x_a] = Patch_Img_B[y*_imgSize.width + x];
					}
					planeRefine(Patch_Img_B[y*_imgSize.width + x]);
				}
// 				updateDispB();
			}
		}
		else	// even iteration: from bottom-right to top-left 
		{
			double bottom_cost = 0.0, right_cost = 0.0, centre_cost = 0.0;
			// Image a (even)
			std::cout << "Even Image A\n";
			for (int y = _imgSize.height - 2; y >= 0; y--)
			{
				for (int x = _imgSize.width - 2; x >= 0; x--)
				{
					// Current cost
					centre_cost = _cost->calcCost_A(cv::Point2i(x, y), Patch_Img_A[y*_imgSize.width + x]);
					// Bottom neighbor
					bottom_cost = _cost->calcCost_A(cv::Point2i(x, y), Patch_Img_A[(y+1)*_imgSize.width + x]);					
					// Right neighbor
					right_cost = _cost->calcCost_A(cv::Point2i(x, y), Patch_Img_A[y*_imgSize.width + x + 1]);

					if (centre_cost > bottom_cost && right_cost > bottom_cost)	// bottom neighbor is best
					{
						Patch_Img_A[y*_imgSize.width + x] = Patch_Img_A[(y+1)*_imgSize.width + x];
					}
					else if (centre_cost > right_cost && bottom_cost > right_cost)	// right neighbor is best
					{
						Patch_Img_A[y*_imgSize.width + x] = Patch_Img_A[y*_imgSize.width + x+1];
					}
					// View propagation
					int x_b = cvRound( x - Patch_Img_A[y*_imgSize.width + x].disparity() );
					if (x_b >= 0)
					{
						if (_cost->calcCost_B(cv::Point2i(x_b, y), Patch_Img_B[y*_imgSize.width + x_b])
							> _cost->calcCost_B(cv::Point2i(x_b, y), Patch_Img_A[y*_imgSize.width + x]))
							Patch_Img_B[y*_imgSize.width + x_b] = Patch_Img_A[y*_imgSize.width + x];
					}
					planeRefine(Patch_Img_A[y*_imgSize.width + x]);
				}
// 				updateDispA();
			}
			// Image b (even)
			std::cout << "Even Image B\n";
			for (int y = _imgSize.height - 2; y >= 0; y--)
			{
				for (int x = _imgSize.width - 2; x >= 0; x--)
				{
					// Current cost
					centre_cost = _cost->calcCost_B(cv::Point2i(x, y), Patch_Img_B[y*_imgSize.width + x]);
					// Bottom neighbor
					bottom_cost = _cost->calcCost_B(cv::Point2i(x, y), Patch_Img_B[(y+1)*_imgSize.width + x]);					
					// Right neighbor
					right_cost = _cost->calcCost_B(cv::Point2i(x, y), Patch_Img_B[y*_imgSize.width + x + 1]);

					if (centre_cost > bottom_cost && right_cost > bottom_cost)	// bottom neighbor is best
					{
						Patch_Img_B[y*_imgSize.width + x] = Patch_Img_B[(y+1)*_imgSize.width + x];
					}
					else if (centre_cost > right_cost && bottom_cost > right_cost)	// right neighbor is best
					{
						Patch_Img_B[y*_imgSize.width + x] = Patch_Img_B[y*_imgSize.width + x+1];
					}
					// View propagation
					int x_a = cvRound( x + Patch_Img_B[y*_imgSize.width + x].disparity() );
					if (x_a < _imgSize.width)
					{
						if (_cost->calcCost_A(cv::Point2i(x_a, y), Patch_Img_A[y*_imgSize.width + x_a])
					> _cost->calcCost_A(cv::Point2i(x_a, y), Patch_Img_B[y*_imgSize.width + x]))
					Patch_Img_A[y*_imgSize.width + x_a] = Patch_Img_B[y*_imgSize.width + x];
					}
					planeRefine(Patch_Img_B[y*_imgSize.width + x]);
				}
// 				updateDispB();
			}
		}		
	}
	updateDisp();
	disp = _disp_A;
}

void CStereoPM::planeRefine(CPatch & curr_ptch)
{
	float delta_z_max = maxDisparity / 2;
	float delta_n_max = 1;
	float dz = 0, dnx = 0, dny = 0, dnz = 0;
	while (delta_z_max > 0.1)
	{
		// Get change for this iteration
		dz = rng_refine.uniform(-delta_z_max, delta_z_max);
		dnx = rng_refine.uniform(-delta_n_max, delta_n_max);
		dny = rng_refine.uniform(-delta_n_max, delta_n_max);
		dnz = rng_refine.uniform(-delta_n_max, delta_n_max);
		// Generate new plane (patch)
		CPatch temp_ptch = curr_ptch;
		// Get new plane (patch)
		cv::Point3f n = curr_ptch.normalForm();
		n += cv::Point3f(dnx, dny, dnz);
		float length = sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
		n.x /= length;
		n.y /= length;
		n.z /= length;
		float z = curr_ptch.disparity() + dz;
		temp_ptch.setPatch(n.x, n.y, n.z, z);
		// Update plane if new plane is better
		if (curr_ptch.side == CPatch::A)
		{
			if(_cost->calcCost_A(temp_ptch.pixelCoord(), temp_ptch) 
				< _cost->calcCost_A(curr_ptch.pixelCoord(), curr_ptch))
				curr_ptch = temp_ptch;
		}
		else
		{
			if(_cost->calcCost_B(temp_ptch.pixelCoord(), temp_ptch) 
				< _cost->calcCost_B(curr_ptch.pixelCoord(), curr_ptch))
				curr_ptch = temp_ptch;
		}
		// Update maximum allowed change
		delta_z_max /= 2;
		delta_n_max /= 2;
	}
	
}

void CStereoPM::initPatch()
{
	Patch_Img_A.reserve(_imgSize.width * _imgSize.height);
	Patch_Img_B.reserve(_imgSize.width * _imgSize.height);

	for (int y = 0; y < _imgSize.height; y++)
	{
		for (int x = 0; x < _imgSize.width; x++)
		{
			if (Left_Img.at<uchar>(y,x) < 0)	// For invalid pixels init default plane
				Patch_Img_A.push_back(CPatch(0.0, 0.0, 0.0, maxDisparity));
			else
				Patch_Img_A.push_back(CPatch(x,y,maxDisparity,CPatch::A));

			if (Right_Img.at<uchar>(y,x) < 0)	// For invalid pixels init default plane
				Patch_Img_B.push_back(CPatch(0.0, 0.0, 0.0, maxDisparity));
			else
				Patch_Img_B.push_back(CPatch(x,y,maxDisparity,CPatch::B));
		}
	}
}

void CStereoPM::updateDisp()
{
	for (int y = 0; y < _imgSize.height; y++)
	{
		for (int x = 0; x < _imgSize.width; x++)
		{
			_disp_A.at<short>(y, x) = cvRound(Patch_Img_A[y * _imgSize.width + x].disparity());
			_disp_B.at<short>(y, x) = cvRound(Patch_Img_B[y * _imgSize.width + x].disparity());
		}
	}
// 	imwrite("dispA"+std::to_string(static_cast<long long>(disp_num))+".png", _disp_A);
// 	imwrite("dispB"+std::to_string(static_cast<long long>(disp_num))+".png", _disp_B);
// 	disp_num++;
}

void CStereoPM::updateDispA()
{
	for (int y = 0; y < _imgSize.height; y++)
	{
		for (int x = 0; x < _imgSize.width; x++)
		{
			_disp_A.at<short>(y, x) = cvRound(Patch_Img_A[y * _imgSize.width + x].disparity());

		}
	}
	imwrite("dispA"+std::to_string(static_cast<long long>(disp_numA))+".png", _disp_A);
	disp_numA++;
}

void CStereoPM::updateDispB()
{
	for (int y = 0; y < _imgSize.height; y++)
	{
		for (int x = 0; x < _imgSize.width; x++)
		{
			_disp_B.at<short>(y, x) = cvRound(Patch_Img_B[y * _imgSize.width + x].disparity());
		}
	}
	imwrite("dispB"+std::to_string(static_cast<long long>(disp_numB))+".png", _disp_B);
	disp_numB++;
}