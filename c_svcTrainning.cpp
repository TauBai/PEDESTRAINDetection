#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
using namespace std;
using namespace cv;
const string DATAPATH = "/media/tau/WIN7/下载/INRIAPerson/";

int main()
{
    cv::Ptr<cv::ml::SVM> svm_ = cv::ml::SVM::create();

    svm_->setType(cv::ml::SVM::C_SVC);
    svm_->setKernel(cv::ml::SVM::LINEAR);
    //svm_->setC(0.1);  //one-class svm 不需要
    //svm_->setNu(0.5);  //C_SVC 不需要
    svm_->setTermCriteria(TermCriteria( TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, FLT_EPSILON ));
    //svm_->setCoef0(0.0);  //实际上对于one_class和c_SVC, 这个和下面两个参数都用不上
    //svm_->setDegree(3);
    //svm_->setGamma(0);


    HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9,1,
                      -1,HOGDescriptor::L2Hys,0.2,false,HOGDescriptor::DEFAULT_NLEVELS);
    cv::Mat sample;
    vector<int> labels;
    ifstream rd(string(DATAPATH + "train_64x128_H96/pos.lst"), ifstream::in);
    if(!rd.is_open()){
        cout << "please check the path of pos.lst"  << endl;
        return -1;
    }

    string line;
    while(getline(rd,line)){
        string posPATH = "/media/tau/WIN7/下载/INRIAPerson/96X160H96/" + line;
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
            //3.1.0的文档里我只看见有compute, 而2.4.11是getDestcriptor

            //Mat rowSample = Mat(descriptor,false);
            //Mat row_;
            // 显然rowSample是不连续的, 它为3780x1矩阵, 故不能用reshape, 只能resize
            //rowSample.resize(1);
            //此处不知道什么原因, resize 只保存了第一个元素, 也就是说变成了 1x1矩阵, 显然不符合我们的期望
            //即变成 1x3780 矩阵
            //rowSample.convertTo(row_,CV_32F);
            //仿照   Mat::Mat(vector<>,bool)函数 将vector拷贝到Mat中, 如下段代码
            //实际上就是Mat初始化的一种
            //Mat::Mat( int rows, int cols, int type, void* data, size_t step=AUTO_STEP )
            Mat rowSample(1,(int)descriptor.size(), CV_32FC1,&descriptor[0]);
            sample.push_back(rowSample);
            labels.push_back(1);
        }else
            cout << "please check the pos files" << endl;
    }
    rd.close();
    rd.open(DATAPATH + "neg.lst");
    int count = 0;
    while(getline(rd,line)){
        string negPATH = line;
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
        svm_->trainAuto(trainData);
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
                hog.detectMultiScale(neg,foundLocations,0.07,Size(4,4),Size(0,0),0.95,2,false);

                for(auto rect : foundLocations){
                    string filename = "enlarged1"+to_string(count);
                    string negImgPath =DATAPATH +"neg/" + filename +".png";
                    Mat foundNEG(neg,rect);
                    resize(foundNEG,foundNEG,Size(64,128));
                    // negImgPath路径一定要正确, 不然不会有写操作, 比如缺少neg文件夹
                    vector<int> compression_params;
                    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
                    compression_params.push_back(0);
                    imwrite(negImgPath,foundNEG,compression_params);
                    outList << negImgPath <<'\n' ;
                    ++count;
                }
            }
        }
        cout << "all the neg img have been saved to the " << DATAPATH << "/neg/ " << endl
                   << "you can refer to neg.lst in the " <<DATAPATH << endl;
        negList.close();
        outList.close();
    }else
        std::cout <<" training failed, please check the sample file" << std::endl;





}

