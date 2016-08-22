#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <vector>
using namespace std;
using namespace cv;
const string DATAPATH = "/media/tau/WIN7/下载/INRIAPerson/";

int main()
{

    HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9,1,
                      -1,HOGDescriptor::L2Hys,0.2,false,HOGDescriptor::DEFAULT_NLEVELS);


    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    ifstream List(DATAPATH+"Train/pos.lst");
    if(!List.is_open()){
        cout << "please check the path of .lst"  << endl;
        return -1;
    }

    string Line;
    while(getline(List,Line)){
        string PATH = DATAPATH +Line;
        Mat img = imread(PATH);


        if(!img.empty()){
            std::vector< Rect > foundLocations;
            hog.detectMultiScale(img,foundLocations,0.07,Size(4,4),Size(0,0),1.05,2,false);
            for(auto rect : foundLocations){
                rectangle(img,rect,Scalar(255,0,0),3);
            }
        }
        cv::imshow("img",img);
        char button = cv::waitKey(0);
        if(button == 27)
            break;
    }




}

