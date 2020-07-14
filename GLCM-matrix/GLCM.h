#pragma once
#include <opencv2/opencv.hpp> // for using library opencv
#include <fstream> // for working with file (write output)

using namespace cv;
// Gray level - 4, 8, 16
enum GrayLevel
{
	GRAY_4,
	GRAY_8,
	GRAY_16
};

enum ChannelRGB
{
	CHANNEL_R,
	CHANNEL_G,
	CHANNEL_B,
	CHANNEL_RG,
	CHANNEL_RB,
	CHANNEL_GB
};


class GLCM
{
public:
	GLCM();
	GLCM(GrayLevel gl);
	~GLCM();

	Mat getImgByChannel(Mat &img, ChannelRGB cn);
	void update(GrayLevel gl);

	void calculateMatrix(Mat &img, int delta_row, int delta_col);
	void normalizeMatrix();
	void calculateFeatures();

	double** getMatrix();
	void printMatrix();
	double* getFeatures_Haralick();
	void printFeatures_Haralick(std::ofstream &outFile);

	void reset();

private:
	double** matrix;
	double* features; // [6] = contrast, correlation, energy, homogeneity, entropy, Inverse Difference Moment
	int gLevel; // size of matrix - gray level
	double sumValue;
	char nameFeatures[6][50] = { "     -Contrast: ", "     -Correlation: ", "     -Energy: ", "     -Homogeneity: ",
		"     -Entropy: ", "     -Inverse Difference Moment: " };
};