#include <iostream>
#include <opencv2/opencv.hpp>

#include "Preprocess.h"
#include "GLCM.h"

int main()
{
	std::cout << "		*** System recognition *** \n";
	cv::Mat img = cv::imread("E:/LuanAn/Demo1/Data/anh1.jpg");
	std::cout << "Input image: w= " << img.cols << "; h= " << img.rows << std::endl;
	cv::imshow("Input image", img);
	if (img.cols < 200 || img.rows < 400)
		cv::resize(img, img, Size(600, 800));
	
	std::cout << "Preprocessing... \n";
	Preprocess* preprocessor = new Preprocess(img);
	Mat* normalized = preprocessor->processing();
	cv::imshow("Normalized image", *normalized);

	std::cout << "Calculate Haralick's features... \n";
	GLCM* glcm = new GLCM(GRAY_8, DIRECTION_0);
	Mat* tmp = new Mat();
	//*normalized = cv::imread("E:/LuanAn/Demo1/Data/test_glcm.png");
	*tmp = glcm->getImgByChannel(*normalized, CHANNEL_B);
	cv::imshow("Image by channel", *tmp);
	glcm->calculateMatrix(*tmp);
	glcm->printMatrix();
	glcm->normalizeMatrix();
	glcm->printMatrix();

	glcm->calculateFeatures();
	glcm->printFeatures_Haralick();
	
	cv::waitKey();
	cv::destroyAllWindows();
	delete preprocessor, normalized;
	delete glcm, tmp;

	system("pause");
	return 0;
}