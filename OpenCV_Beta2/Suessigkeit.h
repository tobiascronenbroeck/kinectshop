#include "stdio.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
using namespace cv;

class Suessigkeit
{
public:
	Suessigkeit();
	Suessigkeit(string Pfad, string Name, int longerborder = 300);
	~Suessigkeit();
	static void customresize(Mat &reference, int longerborder);
	
	// Attribute:
	string sName;
	Mat RGBimage;
	Mat GrayScaleImage;
	vector<KeyPoint> referencekeypoints;
	Mat referencedescriptors;
	MatND hist;
};