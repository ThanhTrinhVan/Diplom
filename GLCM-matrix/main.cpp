#include <iostream>
#include <opencv2/opencv.hpp>
#include <Windows.h>
#include <fstream>
#include <vector>
#include "GLCM.h"

using namespace std;
using namespace cv;

GLCM* glcm = new GLCM(GRAY_8);
ChannelRGB lsChannel[6] = { CHANNEL_R, CHANNEL_G, CHANNEL_B, CHANNEL_RG, CHANNEL_RB, CHANNEL_GB };
char nameChannel[6][50] = { "Channel Red", "Channel Green", "Channel Blue", "Channel Red-Green", "Channel Red-Blue", "Channel Green-Blue" };

void imadjust(const Mat1b& src, Mat1b& dst, float low_in, float high_in, float low_out, float high_out)
{
	dst = src.clone();
	float scale = (high_out - low_out) / (high_in - low_in);
	for (int r = 0; r < dst.rows; ++r)
	{
		for (int c = 0; c < dst.cols; ++c)
		{
			int vs = max(src(r, c) - low_in*255, 0);
			int vd = min(int(vs * scale) + low_out * 255, high_out * 255);
			dst(r, c) = saturate_cast<uchar>(vd);
		}
	}
}

void work(Mat &img, std::ofstream &outFile)
{
	vector<Mat1b> planes;
	split(img, planes);
	for (int i = 0; i < 3; i++) {
		imadjust(planes[i], planes[i], 0.2, 0.75, 0.0, 1.0);
	}
	Mat res;
	merge(planes, res);

	for (int i = 0; i < 6; i++) {
		std::cout << nameChannel[i] << std::endl;
		glcm->reset();

		Mat imgComponent = glcm->getImgByChannel(res, lsChannel[i]);

		glcm->calculateMatrix(imgComponent, 2, 0);
		glcm->normalizeMatrix();

		glcm->calculateFeatures();
		glcm->printFeatures_Haralick(outFile);
	}
}

int main()
{	
	std::ofstream outFile;
	
	/*outFile.open("I:/GradEs/Testing/Brown_rust_BMP/temp_adjust.csv");
	if (!outFile.is_open())
		exit(-1);
	std::string path("I:/GradEs/Testing/Brown_rust_BMP/");
	path.append("\\*");
	WIN32_FIND_DATA data;
	HANDLE hFind;
	if ((hFind = FindFirstFile(path.c_str(), &data)) != INVALID_HANDLE_VALUE) {
		do {
			std::cout << "			Working on file: " << data.cFileName << std::endl;

			std::string filePath("I:/GradEs/Testing/Brown_rust_BMP/");
			filePath.append(data.cFileName);
			Mat img = imread(filePath);
			if (img.rows != 0 && img.cols != 0) {
				outFile << data.cFileName << ",";
				work(img, outFile);
				outFile << std::endl;
			}
			std::cout << "           *** Finished on this file*** \n";

		} while (FindNextFile(hFind, &data) != 0);
		FindClose(hFind);
	}*/
	
	for (int i = 1; i <= 10; i++) {
		std::cout << " Folder: " << i << std::endl;
		std::string folder = std::to_string(i) + "/";
		std::string path("I:/GradEs/Testing/NormGoodBMP/" + folder);
		outFile.open(path + "imadjust.csv");
		if (!outFile.is_open())
			exit(-1);
		path.append("\\*");
		WIN32_FIND_DATA data;
		HANDLE hFind;
		if ((hFind = FindFirstFile(path.c_str(), &data)) != INVALID_HANDLE_VALUE) {
			do {
				std::cout << "			Working on file: " << data.cFileName << std::endl;

				std::string filePath("I:/GradEs/Testing/NormGoodBMP/" + folder);
				filePath.append(data.cFileName);
				Mat img = imread(filePath);
				if (img.rows != 0 && img.cols != 0) {
					outFile << data.cFileName << ",";
					work(img, outFile);
					outFile << std::endl;
				}
				std::cout << "           *** Finished on this file*** \n";

			} while (FindNextFile(hFind, &data) != 0);
			FindClose(hFind);
		}
		outFile.close();
	}
	
	//Mat img = imread("I:/GradEs/Testing/Brown_rust_BMP/1-1.bmp");
	////imshow("Img", img);

	///*Mat res = Mat::zeros(img.size(), img.type());
	//double alpha = 1.2;
	//double beta = 51;
	//for (int y = 0; y < img.rows; y++)
	//	for (int x = 0; x < img.cols; x++)
	//		for (int c = 0; c < 3; c++)
	//			res.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(alpha*(img.at<Vec3b>(y, x)[c]) + beta);*/

	//outFile.open("./temp.csv");
	//work(img, outFile);
	//waitKey(0);
	//destroyAllWindows();
	//outFile.close();

	system("pause");
	delete glcm;
	return 0;
}
