#include "Stereo.h"
#include <fstream>

int main( int argc, char** argv )
{
	std::string dir = ".\\dataset";
	std::string dataset = "\\RPN";
// 	std::string l_name = "\\view1", r_name = "\\view5";
	std::string l_name = "\\leftRecImg", r_name = "\\rightRecImg";
	std::string img_type = ".jpg";

	const char* algorithm_opt = "--algorithm=";
	const char* img1_filename = 0;
	const char* img2_filename = 0;
	const char* intrinsic_filename = 0;
	const char* extrinsic_filename = 0;
	const char* config_filename = 0;
	const char* numK = 0;
	int alg = Stereo::STEREO_SGBM;

	for( int i = 1; i < argc; i++ )
	{
		if( argv[i][0] != '-' )
		{
			if( !img1_filename )
				img1_filename = argv[i];
			else
				img2_filename = argv[i];
		}
		else if( strncmp(argv[i], algorithm_opt, strlen(algorithm_opt)) == 0 )
		{
			char* _alg = argv[i] + strlen(algorithm_opt);
			alg = strcmp(_alg, "bm") == 0 ? Stereo::STEREO_BM :
				strcmp(_alg, "sgbm") == 0 ? Stereo::STEREO_SGBM :
				strcmp(_alg, "hh") == 0 ? Stereo::STEREO_HH :
				strcmp(_alg, "var") == 0 ? Stereo::STEREO_VAR :
				strcmp(_alg, "pm") == 0 ? Stereo::STEREO_PM : -1;
			if( alg < 0 )
			{
				printf("Command-line parameter error: Unknown stereo algorithm\n\n");
// 				print_help();
				return -1;
			}
		}
		else if( strcmp(argv[i], "-i" ) == 0 )
			intrinsic_filename = argv[++i];
		else if( strcmp(argv[i], "-e" ) == 0 )
			extrinsic_filename = argv[++i];
		else if( strcmp(argv[i], "-c" ) == 0 )
			config_filename = argv[++i];
		else
		{
			printf("Command-line parameter error: unknown option %s\n", argv[i]);
			return -1;
		}
	}
	std::string folderName = "results";
	std::string folderCreateCmd = "mkdir " + folderName;
	system(folderCreateCmd.c_str());

	Stereo stereo(alg, extrinsic_filename, intrinsic_filename, config_filename);
	// Set input stereo image pair	
	cv::Mat Img1 = cv::imread(img1_filename);
	cv::Mat Img2 = cv::imread(img2_filename);

// 	stereo.setInputImages(Img1, Img2);
	stereo.setRectImages(Img1, Img2);
	
	cv::Mat disparity, recLeftImage, recRightImage;
	disparity = stereo.calcDisparityMap();
// 	disparity = stereo.getnormDisparityMap();
	stereo.getRectifiedImage(recLeftImage, recRightImage);
	cv::imwrite(folderName +"\\leftRecImg.png", recLeftImage);
	cv::imwrite(folderName +"\\rightRecImg.png", recRightImage);
	cv::imwrite(folderName +"\\disparity.png", disparity);
// 	stereo.calc3DPoints();
	return 0;
}