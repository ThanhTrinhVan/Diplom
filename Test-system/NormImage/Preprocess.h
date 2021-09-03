#pragma once
#ifndef PREPROCESS_IMAGE_H
#define PREPROCESS_IMAGE_H

#include <opencv2/opencv.hpp>
#include <math.h>
#include <vector>

// #define DEBUG
using namespace cv;

class Preprocess
{
public:
	Preprocess();
	~Preprocess();

	Mat* processing();				// preprocessing image
	void updateImg(Mat& new_img);	// update source image
	Mat* getSourceImg() { return src; };

private:
	Mat* src; // source image
	Mat* dst; // destination image

	Mat watershedSegmentation(Mat &img, int centreW, int centreH);
	Mat* rotate(Mat &img, double angle, Point &center);
	Mat* crop(Mat& img); // size 300x100

	const Scalar red = Scalar(0, 0, 255);
	const Scalar green = Scalar(0, 255, 0);
	const Scalar blue = Scalar(255, 0, 0);
};

class WatershedSegmenter {
private:
	cv::Mat markers;
public:
	bool setMarkers(cv::Mat& markerImage)
	{
		if (countNonZero(markerImage) < 1)
			return false;
		markerImage.convertTo(markers, CV_32S);
		return true;
	}

	cv::Mat process(cv::Mat &image)
	{
		if (image.rows == 0 || image.cols == 0)
			return image;
		cv::watershed(image, markers);
		markers.convertTo(markers, CV_8U);
		return markers;
	}
};

double getAvgPixel(Mat img);
#endif // !PREPROCESS_IMAGE_H