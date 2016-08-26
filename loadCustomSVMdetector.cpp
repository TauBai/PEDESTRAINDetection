#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
using namespace std;
using namespace cv;
const string DATAPATH = "/media/tau/WIN7/下载/INRIAPerson/";
const string XMLPATH = "/media/tau/WIN7/下载/INRIAPerson/hog_svm.xml";

int main()
{
    cv::Ptr<cv::ml::SVM> svm_ = cv::ml::SVM::load<ml::SVM>(XMLPATH);

    HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9,1,
                      -1,HOGDescriptor::L2Hys,0.2,true,100);


    ////////   opencv3.1.0 sample code , construct custom SVMDetector   /////////
    Mat sv = svm_->getSupportVectors();
    //cout << sv << endl;
    const int sv_total = sv.rows;
    Mat alpha,svidx;
    double rho = svm_->getDecisionFunction(0,alpha,svidx); //0 for one-class classification(regression, one/two class classification)
    alpha.convertTo(alpha, CV_32FC1);
    CV_Assert( alpha.total() == 1 && svidx.total() == 1 && sv_total == 1 );
    CV_Assert( (alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
               (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f) );
    CV_Assert( sv.type() == CV_32F );


    std::vector<float> hog_detector;
    hog_detector.clear();
    hog_detector.resize(sv.cols + 1);
    memcpy(&hog_detector[0], sv.ptr(), sv.cols*sizeof(hog_detector[0]));
    hog_detector[sv.cols] = (float)-rho;
    ////////////////////////////////////////////////
    hog.setSVMDetector(hog_detector);
    ///////////////////////////////////////////////


    //hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    ifstream testList(DATAPATH+"Test/pos.lst");
        if(!testList.is_open()){
            cout << "please check the path of pos.lst"  << endl;
            return -1;
        }

        cout << "finding pedestrain" << endl;
        string Line;
        Mat img;
        while(getline(testList,Line)){
            string PATH = DATAPATH + Line;
            img = imread(PATH);
            if(!img.empty()){

                std::vector< Rect > foundLocations;
                hog.detectMultiScale(img,foundLocations,0.,Size(2,2),Size(0,0),1.20,2,false);
                cout << foundLocations.size() << endl;
                for(auto rect : foundLocations){
                    rectangle(img,rect,Scalar(0,255,0),3);
                }
                cv::imshow("img",img);
                cv::waitKey(0);
            }else
                cout << "please check the test files" << endl;
        }
        testList.close();
}

