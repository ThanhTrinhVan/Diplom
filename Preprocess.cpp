#include "Preprocess.h"

static bool abs_compare(double a, double b) // для сортировки vector по модулю
{
	return (abs(a) < abs(b));
}

Preprocess::Preprocess()
{
	// создать черное изображение
	src = new Mat(600, 800, CV_8UC1, Scalar(0, 0, 0));
	dst = new Mat();
}

Preprocess::Preprocess(Mat & src_img)
{
	src = new Mat(src_img);
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
	Mat* img_copy, *gray, *thresh, *info_area, *canny;
	img_copy = new Mat(src->clone());

	gray = new Mat();
	cvtColor(*img_copy, *gray, CV_BGR2GRAY);
	blur(*gray, *gray, Size(5, 5));
	//imshow("Gray Image", *gray);

	double thresh_value = getThreshValue(gray);
	std::cout << "  -Threshold's value: " << thresh_value << std::endl;
	thresh = new Mat();
	threshold(*gray, *thresh, thresh_value, 255, CV_THRESH_BINARY);
	//imshow("Threshold-binary image", *thresh);

	std::vector<std::vector<Point>> *contours = new std::vector<std::vector<Point>>();
	std::vector<Vec4i> *hierarchy = new std::vector<Vec4i>();

	// Находить центр информационной части
	findContours(*thresh, *contours, *hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	PointCenter center((img_copy->cols - 1.0) / 2.0, (img_copy->rows - 1.0) / 2.0);
	//info_area = new Mat(img_copy->rows, img_copy->cols, CV_8UC1, Scalar(0, 0, 0));
	info_area = new Mat(Size(img_copy->cols, img_copy->rows), CV_8UC1, Scalar(0));
	int index_info_area = -1;
	RotatedRect minRect;
	for (int i = 0; i < contours->size(); i++) {
		drawContours(*img_copy, *contours, i, color_red, 2, 8, *hierarchy, 0, Point());
		if (contourArea((*contours)[i]) > 0.3*img_copy->cols*img_copy->rows) {
			Moments M = moments((*contours)[i]);
			center.x = M.m10 / M.m00;
			center.y = M.m01 / M.m00;
			index_info_area = i;
			
			minRect = minAreaRect((*contours)[i]);
			center = minRect.center;
		}
	}
	std::cout << "  -Rotate center: (" << center.x << ", " << center.y << ") \n";
	circle(*img_copy, center, 5, color_blue, 2);
	drawContours(*info_area, *contours, index_info_area, Scalar(255, 255, 255), CV_FILLED, 8);
	//imshow("Information area", *info_area);

	contours->clear();
	hierarchy->clear();

	// Найти угол вращения
	canny = new Mat();
	Canny(*gray, *canny, thresh_value, 255);
	//imshow("Canny image", *canny);
	
	findContours(*canny, *contours, *hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	std::vector<double> *angles = new std::vector<double>();
	double angle = 0, length = 0;
	for (int i = 0; i < contours->size(); i++) {
		length = arcLength((*contours)[i], true);
		drawContours(*img_copy, *contours, i, color_blue, 2, 8, *hierarchy, 0, Point());
		if (length > 0.5*min(img_copy->rows, img_copy->cols)) {
			drawContours(*img_copy, *contours, i, color_blue, 2);
			try {
				RotatedRect ellipse = fitEllipse((*contours)[i]);
				if (ellipse.angle > 90)
					ellipse.angle -= 180;
				angles->push_back(ellipse.angle);
			}
			catch (...) {
				angles->push_back(0);
			}
		}
	}
	if (angles->size() != 0) {
		angle = *std::max_element(angles->begin(), angles->end(), abs_compare);
		if (abs(angle - *std::min_element(angles->begin(), angles->end(), abs_compare)) > 45)
			angle = accumulate(angles->begin(), angles->end(), 0.0) / angles->size();

		if (abs(angle) < 5)
			angle = 0;
	}
	std::cout << "  -Rotate angle: " << angle << std::endl;

	dst = rotateSrc(src, angle, center);
	info_area = rotateSrc(info_area, angle, center);
	//imshow("Rotated image", *dst);
	dst = cropping(dst, info_area, center);

	//imshow("Image copy", *img_copy);
	//imshow("Information area", *info_area);

	delete img_copy, gray, thresh, info_area, canny;
	delete contours, hierarchy;
	delete angles;
	return dst;
}

void Preprocess::updateImage(Mat & new_img)
{
	*src = new_img;
}

double Preprocess::getThreshValue(Mat * img)
{
	double val = 0.0;
	for (int i = 0; i < img->rows; i++) {
		for (int j = 0; j < img->cols; j++) {
			val += (double)img->at<uchar>(i, j);
		}
	}
	val /= (double)(img->rows*img->cols);
	return (val + 1.0);
}

Mat * Preprocess::rotateSrc(Mat * img, double & angle, PointCenter & center)
{
	Mat* res = new Mat();
	// get rotation matrix for rotating the image around its center in pixel coordinates
	// Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
	Mat* rot = new Mat();
	*rot = getRotationMatrix2D(center, angle, 1.0);
	// determine bounding rectangle, center not relevant
	//Rect2f bbox = RotatedRect(center, img->size(), angle).boundingRect2f();
	// adjust transformation matrix
	//rot->at<double>(0, 2) += bbox.width / 2.0 - (*img).cols / 2.0;
	//rot->at<double>(1, 2) += bbox.height / 2.0 - (*img).rows / 2.0;

	//warpAffine(*img, *dst, *rot, bbox.size());
	warpAffine(*img, *res, *rot, img->size());
	delete rot;
	return res;
}

Mat * Preprocess::cropping(Mat * img, Mat* infoArea, PointCenter &center)
{
	// src.at(i,j) is using (i,j) as (row,column) but Point(x,y) is using (x,y) as (column,row)
	Mat* res = new Mat();
	*res = img->clone();
	circle(*img, center, 2, color_red, 2);
	int height = infoArea->rows;
	int width = infoArea->cols;
	// pixel 0->255, 0 - black pixel
	Point *topLeft, *topRight, *bottomLeft, *bottomRight;
	bool find = false;
	topLeft = new Point(center);
	for (int i = 0; i < center.y; i++) {
		if (find == true)
			break;
		for (int j = center.x - 1; j >= 0; j--) {
			if ((int)infoArea->at<uchar>(i, j) == 0) {
				topLeft->x = j;
				topLeft->y = i;
				find = true;
				break;
			}
		}
	}
	circle(*img, *topLeft, 2, color_blue, 2);

	find = false;
	topRight = new Point(center);
	for (int i = 0; i < center.y; i++) {
		if (find == true)
			break;
		for (int j = center.x + 1; j < width; j++) {
			if ((int)infoArea->at<uchar>(i, j) == 0) {
				topRight->x = j;
				topRight->y = i;
				find = true;
				break;
			}
		}
	}
	circle(*img, *topRight, 2, color_blue, 2);

	find = false;
	bottomLeft = new Point(center);
	for (int i = height - 1; i > center.y; i--) {
		if (find == true)
			break;
		for (int j = center.x - 1; j >= 0; j--) {
			if ((int)infoArea->at<uchar>(i, j) == 0) {
				bottomLeft->x = j;
				bottomLeft->y = i;
				find = true;
				break;
			}
		}
	}
	circle(*img, *bottomLeft, 2, color_blue, 2);

	find = false;
	bottomRight = new Point(center);
	for (int i = height - 1; i > center.y; i--) {
		if (find == true)
			break;
		for (int j = center.x + 1; j < width; j++) {
			if ((int)infoArea->at<uchar>(i, j) == 0) {
				bottomRight->x = j;
				bottomRight->y = i;
				find = true;
				break;
			}
		}
	}
	circle(*img, *bottomRight, 2, color_blue, 2);
	//imshow("Image cropping", *img);
	int rangeRow[2] = { max(topLeft->y, topRight->y), min(bottomLeft->y, bottomRight->y) };
	int rangeCol[2] = { max(topLeft->x, bottomLeft->x), min(topRight->x, bottomRight->x) };
	bool resizeRow = true;
	bool resizeCol = true;
	if ((rangeRow[1] - rangeRow[0]) >= 300) {
		rangeRow[0] = (rangeRow[0] + rangeRow[1]) / 2 - 150;
		rangeRow[1] = (rangeRow[0] + rangeRow[1]) / 2 + 150;
		resizeRow = false;
	}
	if ((rangeCol[1] - rangeCol[0]) >= 100) {
		rangeCol[0] = (rangeCol[0] + rangeCol[1]) / 2 - 50;
		rangeCol[1] = (rangeCol[0] + rangeCol[1]) / 2 + 50;
		resizeCol = false;
	}
	*res = (*res)(Range(rangeRow[0], rangeRow[1]), Range(rangeCol[0], rangeCol[1]));
	try {
		if (resizeRow || resizeCol)
			resize(*res, *res, Size(100, 300));
	}
	catch (...) {
		return img;
	}
	delete topLeft, topRight, bottomLeft, bottomRight;
	return res;
}
