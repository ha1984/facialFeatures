#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv\cv.h>
//#include <opencv2/improc/improc.hpp>

using namespace cv;
using namespace std;

CascadeClassifier face_cascade, eyes_cascade;
String window_name = "Face Detection";
Mat prev_gray;
vector<Point2f> points[2];
int MAX_COUNT = 30;
vector<Point2f> startPoints(MAX_COUNT);
Rect startFace;

float calc_alpha(vector<Point2f> point0, vector<Point2f> point1, Rect face0, Rect face){
	float alpha = 0.0f;
	float r = (float)face.width / 2.0f; //radius of cylinder
	float r0 = (float)face0.width / 2.0f; //radius of cylinder

	
	//calculate av. rotation angle:
	for (int i = 0; i < point0.size(); i++){
		if (abs(point0[i].x - r0) < r0 && abs(point1[i].x - r0) < r0){
			alpha += acos((point0[i].x - r0) / r0) - acos((point1[i].x - r0) / r0);
		}
	}
	alpha = alpha / (float)point0.size();

	return alpha;
}

void detectFaces(Mat frame){
	std::vector<Rect> faces;
	Mat frame_gray;

	Point priorCenter;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY); //convert to greyscale

	equalizeHist(frame_gray, frame_gray);
	
	//detect faces:
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	//iterate over all faces:
	for (int i = 0; i < faces.size(); i++){
		//find center of face:
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);


		if (abs(center.x - priorCenter.x) < frame.size().width / 2 &&
			abs(center.y - priorCenter.y) < frame.size().height / 2) {

			// Check to see if the user moved enough to update position                           
			if (abs(center.x - priorCenter.x) >2 &&
				abs(center.y - priorCenter.y) >2){
				center = priorCenter;
			}
		}

		//ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		rectangle(frame,Point(faces[i].x, faces[i].y),Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height),Scalar(0, 255, 255),1,8);
		priorCenter = center;
	}

	
	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, MAX_COUNT, 0.3);
	// We use two sets of points in order to swap
	// pointers
	
	//Size subPixWinSize(10, 10), winSize(25, 25);
	//Convert image to gray scale.
	//cvtColor(image, gray, CV_RGB2GRAY);
	//Feature detection is performed here...
	Point p;
	//Lucas Klan optical flow:
	Mat status;
	Mat error;
	//points[0].resize(MAX_COUNT);
	//cornerSubPix(frame_gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
	if (faces.size() > 0){
		Mat face = frame_gray(faces[0]);
		p.x = faces[0].x;
		p.y = faces[0].y;
		calcOpticalFlowPyrLK(prev_gray(faces[0]), frame_gray(faces[0]), points[0], points[1], status, error, Size(30, 30), 1, termcrit, 0, 0.0001);
	}
	else{
		p = priorCenter;
	}
	//display points:
	for (int i = 0; i < points[1].size(); i++){
	circle(frame, (Point)points[1][i] + p, 3.0, Scalar(0, 0, 255), -1, 8);
	}
	//calculate face rotation:
	float alpha;
	if (faces.size() > 0){
		alpha = calc_alpha(startPoints, points[1], startFace, faces[0]);
		cout << alpha << endl;
	}
	//display frame:
	imshow(window_name, frame);
	prev_gray = frame_gray;
	points[0] = points[1];
}


int main(){
	VideoCapture cap(0);
	Mat frame1, frame2;
	std::vector<Rect> faces;
	

	points[0].resize(MAX_COUNT);
	points[1].resize(MAX_COUNT);

	//load haarcascade qualifier:
	face_cascade.load("lbpcascade_frontalface_alt.xml");

	if (face_cascade.empty()){
		cout << "Error loading xml" << endl;
		return 1;
	}
	
	cap.open(0); // 0: use webcam

	cap.set(CV_CAP_PROP_FRAME_WIDTH, 250);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 250);

	if (!cap.isOpened()){
		cout << "ERROR ACQUIRING VIDEO FEED\n";
		getchar();
		return -1;
	}

	//initial face detection and start feature points:
	cap.read(frame1);
	cvtColor(frame1, prev_gray, COLOR_BGR2GRAY);

	equalizeHist(prev_gray, prev_gray);

	//detect faces:
	face_cascade.detectMultiScale(prev_gray, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	if (faces.size() > 0){
		goodFeaturesToTrack(prev_gray(faces[0]), points[0], MAX_COUNT, 0.001, 5.0, Mat(), 3, 0, 0.04);
		startFace = faces[0];
	}
	else {
		goodFeaturesToTrack(prev_gray, points[0], MAX_COUNT, 0.001, 5.0, Mat(), 3, 0, 0.04);
	}

	startPoints = points[0];

	while(1) {
		 //convert to greyscale
		cap.read(frame2);
		detectFaces(frame2);
		if (waitKey(30) >= 0)
			break;
	}
	return 0;
}