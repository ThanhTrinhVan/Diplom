#include <iostream>
#include <opencv2/opencv.hpp>
#include <Windows.h>

#include "Preprocess.h"
#include "GLCM.h"

ChannelRGB lsChannel[6] = { CHANNEL_B, CHANNEL_G, CHANNEL_R, CHANNEL_RB, CHANNEL_GB, CHANNEL_RG };
char nameChannel[6][50] = { "Channel Blue", "Channel Green", "Channel Red", "Channel Red-Blue", "Channel Green-Blue", "Channel Red-Green" };

void pre_dialog(cv::Mat &img) {
	std::cout << "Input image: w= " << img.cols << "; h= " << img.rows << std::endl;
	//cv::imshow("Input image", img);

	std::cout << "Preprocessing... \n";
	Preprocess* preprocessor = new Preprocess(img);
	Mat* normalized = preprocessor->processing();
	//cv::imshow("Normalized image", *normalized);

	std::cout << "Calculate Haralick's features... \n";
	GLCM* glcm = new GLCM(GRAY_8, DIRECTION_0);
	Mat* tmp = new Mat();

	for (int i = 0; i < 6; i++) {
		std::cout << "		" << nameChannel[i] << std::endl;
		*tmp = glcm->getImgByChannel(*normalized, lsChannel[i]);
		//cv::imshow("Image by channel", *tmp);
		glcm->calculateMatrix(*tmp);
		//glcm->printMatrix();
		glcm->normalizeMatrix();
		//glcm->printMatrix();

		glcm->calculateFeatures();
		glcm->printFeatures_Haralick();
	}
	cv::waitKey();
	cv::destroyAllWindows();
	delete preprocessor, normalized;
	delete glcm, tmp;
}

int main()
{
	std::cout << "		*** System recognition *** \n";
	
	cv::Mat img;
	/*
	//std::string path("E:/LuanAn/Demo1/Data/");
	std::string path("E:/LuanAn/300_Photos/Brown_rust/");
	path.append("\\*");
	WIN32_FIND_DATA data;
	HANDLE hFind;
	if ((hFind = FindFirstFile(path.c_str(), &data)) != INVALID_HANDLE_VALUE) {
		do {
			std::string filePath("E:/LuanAn/300_Photos/Brown_rust/");
			filePath.append(data.cFileName);
			img = cv::imread(filePath);
			if (img.rows != 0 && img.cols != 0) {
				std::cout << "     Working on file: " << data.cFileName << std::endl;
				pre_dialog(img);
				std::cout << "           *** Finished on this file*** \n";
			}
		} while (FindNextFile(hFind, &data) != 0);
		FindClose(hFind);
	}
	*/
	
	img = cv::imread("E:/LuanAn/300_Photos/Brown_rust/3.jpg"); // 3.jpg img4 2
	pre_dialog(img);
	
	system("pause");
	return 0;
}