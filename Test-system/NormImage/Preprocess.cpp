#include "Preprocess.h"

double getAvgPixel(Mat img)
{
	double res = 0, num = 0;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			if (img.at<uchar>(i, j) != 0) {
				res += img.at<uchar>(i, j);
				num = num + 1.0;
			}
	return res / num;
}

Preprocess::Preprocess()
{
	// create black image
	src = new Mat(600, 800, CV_8UC1, Scalar(0, 0, 0));
	dst = new Mat();
}

Preprocess::~Preprocess()
{
	if (src != nullptr)
		delete src;
	if (dst != nullptr)
		delete dst;
}

Mat * Preprocess::processing()
{
	// preprocess
	Mat img = src->clone(), drawing = src->clone();
	double angle = 0;
	Point center(0, 0);
	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;

	pyrMeanShiftFiltering(img, img, 3, 8); // 8,3
	#ifdef DEBUG
		imshow("MeanShiftFiltering", img);
	#endif // DEBUG

	// Create a kernel that we will use to sharpen our image
	Mat kernel = (Mat_<float>(3, 3) <<
		0, -1, 0,
		-1, 5, -1,
		0, -1, 0); // an approximation of second derivative, a quite strong kernel
	// do the laplacian filtering as it is
	// well, we need to convert everything in something more deeper then CV_8U
	// because the kernel has some negative values,
	// and we can expect in general to have a Laplacian image with negative values
	// BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
	// so the possible negative number will be truncated
	Mat imgLaplacian;
	filter2D(img, imgLaplacian, CV_32F, kernel);
	Mat sharp;
	img.convertTo(sharp, CV_32F);
	img = sharp - imgLaplacian;
	// convert back to 8bits gray scale
	img.convertTo(img, CV_8UC3);
	imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
	#ifdef DEBUG
		//imshow("Laplace Filtered Image", imgLaplacian);
		imshow("Sharped Image", img);
	#endif // DEBUG

	// Watershed segmentation
	Mat dest = watershedSegmentation(drawing, img.cols / 4, img.rows / 4);
	#ifdef DEBUG
		imshow("SegmentationColor", dest);
	#endif // DEBUG

	cvtColor(dest, dest, CV_RGB2HSV);
	#ifdef DEBUG
		imshow("HSV", dest);
	#endif // DEBUG
	Mat hsv[3];
	split(dest, hsv);
	double vh = getAvgPixel(hsv[0]), vs = getAvgPixel(hsv[1]), vv = getAvgPixel(hsv[2]);
	inRange(dest, Scalar(vh, vs, vv), Scalar(255, 255, 255), dest);
	Mat structuringElement = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(dest, dest, MORPH_CLOSE, structuringElement);
	#ifdef DEBUG
		imshow("Range", dest);
	#endif // DEBUG

	findContours(dest, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	double maxS = 0;
	int idM = 0;
	for (int i = 0; i < contours.size(); i++) {
		if (contourArea(contours[i]) > maxS) {
			idM = i;
			maxS = contourArea(contours[i]);
		}
	}
	RotatedRect minBound(Point2f(0, 0), Size2f(0, 0), 0);
	if (contours.size() > 0) {
		drawContours(drawing, contours, idM, red, 2);
		minBound = fitEllipse(contours[idM]);
	}
	if (maxS < (double)(dest.rows*dest.cols) / 100.0) {
		dest = watershedSegmentation(img, img.cols / 8, img.rows / 8);
		cvtColor(dest, img, COLOR_BGR2GRAY);
		#ifdef DEBUG
			imshow("Gray", img);
		#endif // DEBUG

		Mat thresh;
		double vthresh = getAvgPixel(img);
		threshold(img, thresh, vthresh, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		#ifdef DEBUG
			imshow("Theshold", thresh);
		#endif // DEBUG

		Mat structuringElement = getStructuringElement(MORPH_ELLIPSE, Size(img.cols / 8, img.rows / 8));
		morphologyEx(thresh, thresh, MORPH_CLOSE, structuringElement);
		#ifdef DEBUG
			imshow("Grouping", thresh);
		#endif // DEBUG
		
		if (contours.size() > 0){
			contours.clear();
			hierarchy.clear();
		}
		findContours(thresh, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		maxS = 0;
		idM = 0;
		for (int i = 0; i < contours.size(); i++) {
			if (contourArea(contours[i]) > maxS) {
				idM = i;
				maxS = contourArea(contours[i]);
			}
		}
		drawContours(drawing, contours, idM, green, 2);
		minBound = fitEllipse(contours[idM]);
	}
	ellipse(drawing, minBound, green, 2);
	Point2f rect_points[4];
	minBound.points(rect_points);
	for (int j = 0; j < 4; j++)
		line(drawing, rect_points[j], rect_points[(j + 1) % 4], blue, 2);
	#ifdef DEBUG
		imshow("Drawing", drawing);
	#endif // DEBUG

	center = minBound.center;
	if (minBound.angle > 90)
		angle = minBound.angle - 180;
	else
		angle = minBound.angle;
	
	///////////////////////////////////////////////
	dst = rotate(*src, angle, center);
	#ifdef DEBUG
		imshow("Rotate", *dst);
	#endif // DEBUG
	Range drows(max((int)(center.y - minBound.size.height / 2), 0), min((int)(center.y + minBound.size.height / 2), img.rows));
	Range dcols(max((int)(center.x - minBound.size.width / 2), 0), min((int)(center.x + minBound.size.width / 2), img.cols));
	*dst = (*dst)(drows, dcols);
	resize(*dst, *dst, Size(100, 300));
	return dst;
}


void Preprocess::updateImg(Mat & new_img)
{
	*src = new_img;
}

Mat Preprocess::watershedSegmentation(Mat & img, int centreW, int centreH)
{
	// Watershed segmentation
	Mat markers(img.size(), CV_8U, cv::Scalar(-1));
	//top rectangle
	markers(Rect(0, 0, img.cols, 5)) = Scalar::all(1);
	//bottom rectangle
	markers(Rect(0, img.rows - 5, img.cols, 5)) = Scalar::all(1);
	//left rectangle
	markers(Rect(0, 0, 5, img.rows)) = Scalar::all(1);
	//right rectangle
	markers(Rect(img.cols - 5, 0, 5, img.rows)) = Scalar::all(1);
	//centre rectangle
	markers(Rect((img.cols / 2) - (centreW / 2), (img.rows / 2) - (centreH / 2), centreW, centreH)) = Scalar::all(2);
	markers.convertTo(markers, CV_BGR2GRAY);
	#ifdef DEBUG
		imshow("markers", markers);
	#endif // DEBUG
	WatershedSegmenter segmenter;
	segmenter.setMarkers(markers);
	// For grayscale sharped image
	Mat dest;
	cv::Mat wshedMask = segmenter.process(img);
	cv::Mat mask;
	convertScaleAbs(wshedMask, mask, 1, 0);
	threshold(mask, mask, 1, 255, THRESH_BINARY);
	bitwise_and(img, img, dest, mask);
	dest.convertTo(dest, CV_8U);
	#ifdef DEBUG
		imshow("Watershed Segmentation", dest);
	#endif // DEBUG
	return dest;
}

Mat * Preprocess::rotate(Mat & img, double angle, Point & center)
{
	Mat* res = new Mat();
	Mat* rot = new Mat();
	*rot = getRotationMatrix2D(center, angle, 1.0);
	// determine bounding rectangle, center not relevant
	/*Rect2f bbox = RotatedRect(center, img->size(), angle).boundingRect2f();
	// adjust transformation matrix
	rot->at<double>(0, 2) += bbox.width / 2.0 - center.x;//img.cols / 2.0;
	rot->at<double>(1, 2) += bbox.height / 2.0 - center.y;//img.rows / 2.0;
	warpAffine(img, *res, *rot, bbox.size());*/
	warpAffine(img, *res, *rot, img.size());
	delete rot;
	return res;
}

Mat * Preprocess::crop(Mat & img)
{
	return &img;
}
