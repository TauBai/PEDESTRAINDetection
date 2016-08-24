#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
using namespace cv;

float accuracy(Ptr<ml::SVM> svm_ ,Mat& pos, Mat& neg, float& precision, float& recall);
int main()
{
    string path = "/media/tau/WIN7/下载/INRIAPerson/hog_svm.xml";   // path for trained svm xml
    string pathTEST = "/media/tau/WIN7/下载/INRIAPerson/test_64x128_H96/";
    string DATAPATH = "/media/tau/WIN7/下载/INRIAPerson/70X134H96/";


    HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9,1,
                      -1,HOGDescriptor::L2Hys,0.2,true,HOGDescriptor::DEFAULT_NLEVELS);


    Mat pos;   //test data
    Mat neg;   //test data

    //Mat test=imread("/media/tau/WIN7/下载/INRIAPerson/test_64x128_H96/pos/crop001501a.png");
    //imshow("t",t);
    //cout << t.size() << endl;
    cv::waitKey(0);
    ifstream rdPos(pathTEST+"pos.lst",ifstream::in);
    if(!rdPos.is_open()){
        cout << "Please check the pos.lst" << endl;
        return -1;
    }
    string line;
    while(getline(rdPos,line)){
        string posPATH = DATAPATH + line;
        Mat posSample = imread(posPATH);
        //imshow("posSample",posSample);
        //cv::waitKey();
        if(!posSample.empty() && posSample.size() == Size(70,134)){
            Mat clip(posSample,cv::Rect(3,3,64,128));
            vector<float> descriptor;
            hog.compute(clip,descriptor,Size(8,8),Size(0,0));
            Mat rowSample(1,(int)descriptor.size(), CV_32FC1,&descriptor[0]);
            pos.push_back(rowSample);
        }else
            cout << "please check the pos files" << endl;
    }
    rdPos.close();



    ifstream rdNeg(pathTEST+"neg.lst",ifstream::in);    //Note that size of negtive img is 64x128
    if(!rdNeg.is_open()){
        cout << "Please check the pos.lst" << endl;
        return -1;
    }
    while(getline(rdPos,line)){
        string negPATH = DATAPATH+ line;
        Mat negSample = imread(negPATH);
        if(!negSample.empty() && negSample.size()==Size(64,128) ){
            vector<float> descriptor;
            hog.compute(negSample,descriptor,Size(8,8),Size(0,0));
            Mat rowSample(1,(int)descriptor.size(), CV_32FC1,&descriptor[0]);
            neg.push_back(rowSample);
        }else
            cout << "please check the pos files" << endl;
    }
    rdNeg.close();



    float precision,recall;
    cv::Ptr<ml::SVM> svm_ = ml::SVM::load<cv::ml::SVM>(path);
    float accurate = accuracy(svm_,pos,neg,precision,recall);
    cout << " model accuracy in test: " << accurate << endl;
    cout << " precision : " << precision << endl;
    cout << " recall    : " << recall << endl;
    
}
float accuracy(Ptr<ml::SVM> svm_ ,Mat& pos, Mat& neg, float& precision, float& recall){

    float accurate= 0;
    precision = 0;
    recall = 0;

    Mat posLabel , negLabel;
    svm_->predict(pos,posLabel,0);
    svm_->predict(neg,negLabel,0);

    int tp = 0;
    int tn = 0;
    int n1 = posLabel.rows;
    for(int y = 0; y < n1; ++y){
        float* data = posLabel.ptr<float>(y);
        if( abs(data[0]) > 0.001)
            ++tp;
    }
    int n2 = negLabel.rows;
    for(int y = 0; y < n2; ++y){
        float* data = negLabel.ptr<float>(y);
        if( abs(data[0]) <= 0.001)
            ++tn;
    }

    float fp = n2 - tn;
    precision = float(tp)/(tp + fp)* 100;
    recall = float(tp)/ n1* 100;
    return accurate;
    
}
