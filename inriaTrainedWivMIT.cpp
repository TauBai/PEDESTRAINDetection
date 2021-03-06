#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
using namespace std;
using namespace cv;
const string DATAPATH = "/home/tau/INRIAPerson/";

int main()
{
    cv::Ptr<cv::ml::SVM> svm_ = cv::ml::SVM::create();

    svm_->setType(cv::ml::SVM::C_SVC);
    svm_->setKernel(cv::ml::SVM::LINEAR);
    svm_->setC(0.1);  //one-class svm 不需要
    svm_->setNu(0.5);  //C_SVC 不需要
    svm_->setTermCriteria(TermCriteria( TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, FLT_EPSILON ));
    svm_->setCoef0(0.0);  //实际上对于one_class和c_SVC, 这个和下面两个参数都用不上
    svm_->setDegree(3);
    svm_->setGamma(0);


    HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9,1,
                      -1,HOGDescriptor::L2Hys,0.2,true,HOGDescriptor::DEFAULT_NLEVELS);
    cv::Mat sample;
    vector<int> labels;
    ifstream rd(string(DATAPATH + "train_64x128_H96/pos.lst"), ifstream::in);
    if(!rd.is_open()){
        cout << "please check the path of pos.lst"  << endl;
        return -1;
    }

    string line;
    while(getline(rd,line)){
        string posPATH = "/home/tau/INRIAPerson/96X160H96/" + line;
        //cout << posPATH << endl;   //for debugging
        Mat pos = imread(posPATH);

        //imshow("pos",pos);
        if(!pos.empty()){
            Mat posCLIP(pos,cv::Rect(16,16,64,128));
            //cv::imshow("64x128",posCLIP);
            //std::cout << posCLIP.size() <<std::endl;
            //cv::waitKey(0);

            vector<float> descriptor;
            hog.compute(posCLIP,descriptor,Size(8,8),Size(0,0));

            Mat rowSample(1,(int)descriptor.size(), CV_32FC1,&descriptor[0]);
            sample.push_back(rowSample);
            labels.push_back(1);
        }else
            cout << "please check the pos files" << endl;
    }
    rd.close();
    rd.open(string(DATAPATH + "MITpedestrians128x64/pos.lst"), ifstream::in);
    if(!rd.is_open()){
        cout << "please check the path of mit pos.lst"  << endl;
        return -1;
    }
    while(getline(rd,line)){
        string posPATH = "/home/tau/INRIAPerson/MITpedestrians128x64/pos/" + line;
        //cout << posPATH << endl;   //for debugging
        Mat pos = imread(posPATH);

        //imshow("pos",pos);
        if(!pos.empty()){
            vector<float> descriptor;
            hog.compute(pos,descriptor,Size(8,8),Size(0,0));

            Mat rowSample(1,(int)descriptor.size(), CV_32FC1,&descriptor[0]);
            sample.push_back(rowSample);
            labels.push_back(1);
        }else
            cout << "please check the pos files" << endl;
    }
    rd.close();
    rd.open(string(DATAPATH + "neg.lst"), ifstream::in);
    if(!rd.is_open()){
        cout << "please check the path of mit pos.lst"  << endl;
        return -1;
    }
    int count = 0;
    while(getline(rd,line)){
        string negPATH = DATAPATH+line;
        //cout << negPATH << endl;   //for debugging
        Mat neg = imread(negPATH);

        //imshow("neg",neg);
        if(!neg.empty()){
            vector<float> descriptor;
            hog.compute(neg,descriptor,Size(8,8),Size(0,0));
            Mat rowSample(1,(int)descriptor.size(), CV_32FC1,&descriptor[0]);
            sample.push_back(rowSample);
            labels.push_back(-1);
            ++count;
        }else
            cout << "please check the neg files" << endl;
    }
    rd.close();


    if(!sample.empty()){
        cv::Ptr<cv::ml::TrainData> trainData =
              cv::ml::TrainData::create(sample, cv::ml::SampleTypes::ROW_SAMPLE,
                                        labels);

        cv::ml::ParamGrid c_grid(0.0001,1000,5);
        svm_->trainAuto(trainData,10,c_grid);
        std::cout << "training process finished " << std::endl;

        svm_->save(DATAPATH + "hog_svm.xml");
        std::cout << "SVM data saved under the directory of" << DATAPATH << std::endl;

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
        ifstream negList(DATAPATH+"Train/neg.lst");
        if(!negList.is_open()){
            cout << "please check the path of neg.lst"  << endl;
            return -1;
        }
        ofstream outList(string(DATAPATH + "neg.lst"),ofstream::app);
        if(!outList.is_open()){
            cout << "error writing img path to the neg.lst" << endl;
            return -1;
        }

        cout << "finding hard example" << endl;
        string negLine;
        while(getline(negList,negLine)){
            string negPATH = DATAPATH + negLine;
            Mat neg = imread(negPATH);
            //cv::imshow("neg",neg);
            //cv::waitKey(0);
            if(!neg.empty()){
                std::vector< Rect > foundLocations;
                hog.detectMultiScale(neg,foundLocations,0.07,Size(2,2),Size(0,0),1.05,2,false);

                for(auto rect : foundLocations){
                    string filename = "b"+to_string(count);
                    string negImgPath =DATAPATH +"neg/" + filename +".png";
                    Mat foundNEG(neg,rect);
                    resize(foundNEG,foundNEG,Size(64,128));
                    // negImgPath路径一定要正确, 不然不会有写操作, 比如缺少neg文件夹
                    vector<int> compression_params;
                    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
                    compression_params.push_back(0);
                    imwrite(negImgPath,foundNEG,compression_params);
                    outList << string("neg/" + filename +".png")<<'\n' ;
                    ++count;
                }
            }
        }
        cout << "all the neg img have been saved to the " << DATAPATH << "neg/ " << endl
                   << "you can refer to neg.lst in the " <<DATAPATH << endl;
        printf( "train error: %f\n", svm_->calcError(trainData, false, noArray()) );
        printf( "test error: %f\n\n", svm_->calcError(trainData, true, noArray()) );
        negList.close();
        outList.close();
    }else
        std::cout <<" training failed, please check the sample file" << std::endl;

}
