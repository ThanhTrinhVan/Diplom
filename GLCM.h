#pragma once
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace cv;

// Gray level - 4, 8, 16
enum GrayLevel
{
	GRAY_4,
	GRAY_8,
	GRAY_16
};

// Statistical Direction: 0, 45, 90, 135 - Offset: [0,1] [-1,1], [-1,0], [-1,-1]
enum Direction
{
	DIRECTION_0,
	DIRECTION_45,
	DIRECTION_90,
	DIRECTION_135
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
	GLCM(GrayLevel level_, Direction direct_);
	~GLCM();

	void calculateMatrix(Mat &img);
	void calculateFeatures();
	void normalizeMatrix();
	Mat getImgByChannel(Mat &img, ChannelRGB cn);
	void update(GrayLevel level_, Direction direct_);

	double** getMatrix();
	void printMatrix();
	double* getFeatures_Haralick();
	void printFeatures_Haralick();

private:
	double countValue(Mat &img, int id_u, int id_v);

	double** matrix;
	double* features; 
	Direction direct; // [6] = contrast, correlation, energy, homogeneity, entropy, Inverse Difference Moment
	int G_level; // Gray Level
	int i, j, iu, jv;
	double sumValue;
	char NameFeatures[6][50] = { "     -Contrast: ", "     -Correlation: ", "     -Energy: ", "     -Homogeneity: ",
		"     -Entropy: ", "     -Inverse Difference Moment: " };
};

