#include <iostream>
#include <Windows.h>
#include <fstream>
#include <vector>
#include <random>
#include "opencv2/opencv.hpp"
#include "Preprocess.h"
#include "GLCM.h"
#include "Classifier.h"

using namespace cv;
using namespace std;

Preprocess* preprocessor = new Preprocess();
char nameFolder[10][50] = { "Brown_rust/", "Dark_brown_spotting/", "Powdery_mildew/",
					"Pyrenophorosis/", "Root_rot/", "Septoria/", "Smut/", "Snow_mold/", "Striped_mosaic/", "Yellow_rust/" };

GLCM* glcm = new GLCM(GRAY_8);
ChannelRGB lsChannel[6] = { CHANNEL_R, CHANNEL_G, CHANNEL_B, CHANNEL_RG, CHANNEL_RB, CHANNEL_GB };
char nameChannel[6][50] = { "Channel Red", "Channel Green", "Channel Blue", "Channel Red-Green", "Channel Red-Blue", "Channel Green-Blue" };

int m = 1;			// количество статистических испытаний. 
double S = 0.05;	// СКО случайных отклонений от эталонных значений method 1 SKO = 0.05, method 2 SKO = 0.04

int diagnostic(int k) 
{
	// k - номер заболевания на входе 9 - unknown
	int kb_in = pow(2, k);  // входной код заболевания (нужен для определения правильности диагностики)
	Classifier* dig = new Classifier(k, 8);
	dig->reset();
	vector<double> noise;
	random_device rd;
	mt19937 gen(rd());
	normal_distribution<double> d(0.0, 1.0);
	for (int i = 0; i < 24; i++)
		noise.push_back(d(gen));
	dig->loadData("data.csv", noise, S);
	dig->calculateS();
	dig->calculateSKO();
	dig->membershipFunction();
	dig->printData();
	int k_out = dig->predict();
	return k_out;
}

void work(Mat &img)
{
	vector<Mat1b> planes;
	split(img, planes);
	std::ofstream outFile;
	outFile.open("I:/GradEs/Testing/unittest.csv");
	if (!outFile.is_open())
		exit(-1);

	for (int i = 0; i < 6; i++) {
		std::cout << nameChannel[i] << std::endl;
		glcm->reset();

		Mat imgComponent = glcm->getImgByChannel(img, lsChannel[i]);

		glcm->calculateMatrix(imgComponent, 2, 0);
		glcm->printMatrix();
		glcm->normalizeMatrix();
		glcm->printMatrix();

		glcm->calculateFeatures();
		glcm->printFeatures_Haralick(outFile);
	}
}

void test_img(std::string path)
{
	Mat img = imread(path);
	//resize(img, img, Size(600, 800));
	imshow("Source", img);
	preprocessor->updateImg(img);
	Mat* norm = preprocessor->processing();
	imshow("Result", *norm);
	work(*norm);
	diagnostic(2);
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
		Smut/Urocystis colchici Henallt Common 20May2010 152 (c) RGWoods.jpg									1
		Brown_rust/Screen Shot 2018-10-07 at 19.28.02.png  2.jpg 1.jpg
		Yellow_rust/2.jpg 15855609923_b0b805a93c_b.jpg  Yellow_rust/Screen Shot 2018-10-07 at 19.45.41.png      1
		59f0b8c8edf76.jpeg
		Pyrenophorosis/Screen Shot 2018-10-07 at 18.50.42.png
	*/

	test_img("I:/GradEs/300_Photos/Septoria/Septoria-tritici (1).jpg");
	waitKey();

	/*
	std::string path1 = "I:/GradEs/300_Photos/";
	std::string path2 = "I:/GradEs/NormImage/Result_2/";
	for (int i = 0; i < 10; i++) {
		std::cout << "		Folder: " << nameFolder[i] << std::endl;
		test_folder(path1 + nameFolder[i], path2 + nameFolder[i]);
	}
	system("pause");
	*/
	destroyAllWindows();
	delete preprocessor;
	return 0;
}