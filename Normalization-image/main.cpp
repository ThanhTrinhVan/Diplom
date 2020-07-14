#include <iostream>
#include <Windows.h>
#include "opencv2/opencv.hpp"
#include "Preprocess.h"

using namespace cv;

Preprocess* preprocessor = new Preprocess();
char nameFolder[10][50] = { "Brown_rust/", "Dark_brown_spotting/", "Powdery_mildew/",
					"Pyrenophorosis/", "Root_rot/", "Septoria/", "Smut/", "Snow_mold/", "Striped_mosaic/", "Yellow_rust/" };

void test_img(std::string path)
{
	Mat img = imread(path);
	//resize(img, img, Size(600, 800));
	imshow("Source", img);
	preprocessor->updateImg(img);
	Mat* norm = preprocessor->processing();
	imshow("Result", *norm);
}

void test_folder(std::string pathSrc, std::string pathDst)
{
	std::string findPath(pathSrc);
	findPath.append("\\*");
	WIN32_FIND_DATA data;
	HANDLE hFind;
	if ((hFind = FindFirstFile(findPath.c_str(), &data)) != INVALID_HANDLE_VALUE) {
		do {
			std::cout << "File: " << data.cFileName;
			std::string filePath(pathSrc);
			filePath.append(data.cFileName);

			cv::Mat img = cv::imread(filePath);
			if (img.rows != 0 && img.cols != 0) {
				try {
					preprocessor->updateImg(img);
					cv::Mat* normalized = preprocessor->processing();
					std::string pathSave(pathDst);
					pathSave.append(data.cFileName);
					cv::imwrite(pathSave, *normalized);
				}
				catch (...) {
					continue;
				}
			}
			std::cout << "  - done! \n";
		} while (FindNextFile(hFind, &data) != 0);
		FindClose(hFind);
	}
}

int main() 
{
	/*
		Smut/Urocystis colchici Henallt Common 20May2010 152 (c) RGWoods.jpg
		Brown_rust/Screen Shot 2018-10-07 at 19.28.02.png  2.jpg 1.jpg
		Yellow_rust/2.jpg 15855609923_b0b805a93c_b.jpg  Yellow_rust/Screen Shot 2018-10-07 at 19.45.41.png
		59f0b8c8edf76.jpeg
		Pyrenophorosis/Screen Shot 2018-10-07 at 18.50.42.png
	*/

	/*test_img("I:/GradEs/300_Photos/Pyrenophorosis/Screen Shot 2018-10-07 at 18.50.42.png");
	waitKey();*/

	std::string path1 = "I:/GradEs/300_Photos/";
	std::string path2 = "I:/GradEs/NormImage/Result_2/";
	for (int i = 0; i < 10; i++) {
		std::cout << "		Folder: " << nameFolder[i] << std::endl;
		test_folder(path1 + nameFolder[i], path2 + nameFolder[i]);
	}
	system("pause");

	destroyAllWindows();
	delete preprocessor;
	return 0;
}