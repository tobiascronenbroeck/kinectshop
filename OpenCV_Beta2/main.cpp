#define _CRT_SECURE_NO_DEPRECATE
#include "stdio.h"
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/flann/flann.hpp"
#include "Suessigkeit.h"
#include <ctime>


#define tresholdcamerafailure 250 // Wird zur beschleunigung und Stabilitaet benötigt! Standart 250
#define tresholdgoodcamerakeypoints 10 // Standart 10
#define minHessian 350           // Schwellwert fuer Surf Detector!

int minHessian_swap = minHessian; // Fuer andere Klassen!

using namespace std;
using namespace cv;

vector<Suessigkeit*> sortiment;

//////////////////////////////////////////////////////////////////////////////////
///  Int Database Vector!!!
//////////////////////////////////////////////////////////////////////////////////
void intdatabase()
{
	
	/*sortiment.push_back(new Suessigkeit("referenceimage-1-.jpg", "Weisse_Schokolade"));
	sortiment.push_back(new Suessigkeit("referenceimage-2-.jpg", "Weisse_Schokolade"));
	sortiment.push_back(new Suessigkeit("referenceimage-4-.jpg", "Weisse_Schokolade"));
	sortiment.push_back(new Suessigkeit("referenceimage-5-.jpg", "Weisse_Schokolade"));
	sortiment.push_back(new Suessigkeit("referenceimage-6-.jpg", "Weisse_Schokolade"));
	sortiment.push_back(new Suessigkeit("referenceimage-7-.jpg", "Weisse_Schokolade"));*/
	sortiment.push_back(new Suessigkeit("referenceimage-3-.jpg", "Weisse_Schokolade"));
}
//////////////////////////////////////////////////////////////////////////////////
/// customsurfdetector returns an pointer to the detected object
//////////////////////////////////////////////////////////////////////////////////
Suessigkeit* customsurfdetector(vector<Suessigkeit*> &sortiment, Mat &img_scene, double minFlaeche = 300)
{
	
	if (!(sortiment.empty()))
	{
			//SURF Keypoints fuer das Kamerabild wird berechnet!
			SurfFeatureDetector detector(minHessian);
			vector<KeyPoint> keypoints_scene;
            detector.detect(img_scene, keypoints_scene);
           
			/* Prueft ob Vergleich Sinn macht! ->
			falls zu wenige Keypoints gefunden werden wird hier schon NULL zurueckgegeben.*/
			if (keypoints_scene.size() < tresholdcamerafailure)  { return new Suessigkeit(); }

			/*Wenn Vergleich sinn macht werden die beschreibenden Merkmale des Kamerabildes erzeugt!
			Diese werden in der Matrix descriptors_scene gespeichert!*/
			SurfDescriptorExtractor extractor;
			Mat descriptors_scene;
			extractor.compute(img_scene, keypoints_scene, descriptors_scene); // Anwendung des SurfExtractors

            vector<Suessigkeit*>::iterator iter;
		    for (iter = sortiment.begin(); iter != sortiment.end(); iter++){
            //FLANNBasedMatcher
			FlannBasedMatcher matcher;
			vector< DMatch > matches;
			matcher.match((*iter)->referencedescriptors, descriptors_scene, matches);

			double max_dist=0;  double min_dist=150;
			//-- Quick calculation of max and min distances between keypoints
			for (int i = 0; i < ((*iter)->referencedescriptors).rows; i++)
			{
				double dist = matches[i].distance;
				if (dist < min_dist) min_dist = dist;
				if (dist > max_dist) max_dist = dist;
			}
            vector< DMatch > good_matches;

			//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
			for (int i = 0; i < ((*iter)->referencedescriptors).rows; i++)
			{
				if (matches[i].distance < 3 * min_dist)
				{
					good_matches.push_back(matches[i]);
				}
			}
            Mat img_matches;
			drawMatches((*iter)->GrayScaleImage, (*iter)->referencekeypoints, img_scene, keypoints_scene,
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            
			vector<Point2f> obj;  vector<Point2f> scene;
			for (int i = 0; i < good_matches.size(); i++)
			{
				//-- Get the keypoints from the good matches
				obj.push_back(((*iter)->referencekeypoints)[good_matches[i].queryIdx].pt);
				scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
			}
			
			

			// Es wird geprueft ob genuegend gute Keypoints vorhanden sind. (Performance und Stabilitaet)
			
			if (scene.size() < tresholdgoodcamerakeypoints){ break; }

			Mat H = findHomography(obj, scene, CV_RANSAC);
			
			std::vector<Point2f> obj_corners(4); 
			obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(((*iter)->GrayScaleImage).cols, 0);
			obj_corners[2] = cvPoint(((*iter)->GrayScaleImage).cols, ((*iter)->GrayScaleImage).rows); obj_corners[3] = cvPoint(0, ((*iter)->GrayScaleImage).rows);
			std::vector<Point2f> scene_corners(4);

			perspectiveTransform(obj_corners, scene_corners, H);

			Point2f aufpunkt = scene_corners[0];
			Point2f xEndpunkt = scene_corners[1];
			Point2f yEndpunkt = scene_corners[3];
			Point2f opposite = scene_corners[2];


			Point2f dist1 = xEndpunkt - aufpunkt;
			Point2f dist2 = yEndpunkt - aufpunkt;
			Point2f checkpoint1 = aufpunkt + dist1 + dist2;
			
			double distx = sqrt((dist1.x)*(dist1.x) + (dist1.y*dist1.y));
			double disty = sqrt((dist2.x)*(dist2.x) + (dist2.y*dist2.y));
			double area = fabs((dist1.x * dist2.x) + (dist1.y * dist2.y));
  
			if ((distx > 100) && (disty > 100))
			{
				if ((area >= minFlaeche) && (fabs((checkpoint1.x) - (opposite.x)) < 60 && (fabs((checkpoint1.y) - (opposite.y)) < 60)) && area <= 10000)
				{   
			        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
					line(img_matches, scene_corners[0] + Point2f(((*iter)->GrayScaleImage).cols, 0), scene_corners[1] + Point2f(((*iter)->GrayScaleImage).cols, 0), Scalar(0, 255, 0), 4);
					line(img_matches, scene_corners[1] + Point2f(((*iter)->GrayScaleImage).cols, 0), scene_corners[2] + Point2f(((*iter)->GrayScaleImage).cols, 0), Scalar(0, 255, 0), 4);
					line(img_matches, scene_corners[2] + Point2f(((*iter)->GrayScaleImage).cols, 0), scene_corners[3] + Point2f(((*iter)->GrayScaleImage).cols, 0), Scalar(0, 255, 0), 4);
					line(img_matches, scene_corners[3] + Point2f(((*iter)->GrayScaleImage).cols, 0), scene_corners[0] + Point2f(((*iter)->GrayScaleImage).cols, 0), Scalar(0, 255, 0), 4);
			
					putText(img_matches, (*iter)->sName, opposite + Point2f(200, 30), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 0));
					//Show detected matches
					imshow("Good Matches & Object detection", img_matches);
					return (*iter);
				}
			}
		}
	}
	return new Suessigkeit();
}

int main()
{

	VideoCapture cap(0);
	Mat image_1;
	intdatabase();
	vector<Suessigkeit*> auswahl;
	vector<Suessigkeit*>::iterator iter;

	while (waitKey(50) != 'q'){

		unsigned int start = clock();
		auswahl.clear();
		cap >> image_1;
		if (!image_1.data) { break; }
	
		Mat image_2(image_1);
		
		Suessigkeit::customresize(image_1, 400);
		cvtColor(image_1, image_1, CV_BGR2GRAY);
		Suessigkeit* result = customsurfdetector(sortiment, image_1);
		
		if ((result)->sName != "NULL")
		{
			cout << "Es handelt sich um das Objekt " << result->sName << endl;
		}

		double deltatime = clock() - start;
		int fps = (int)(1000 / (deltatime)) + 1;
		putText(image_2, to_string(fps), Point2f(image_2.cols,image_2.rows) - Point2f(80,20), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0),3);
		imshow("Kamerabild", image_2);		
		
	}
	return 0;
}
