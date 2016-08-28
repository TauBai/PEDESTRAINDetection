#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <random>
using namespace std;
using namespace cv;
const string DATAPATH = "/home/tau/INRIAPerson/";
const string negPATH = "/home/tau/INRIAPerson/train_64x128_H96/";
int numOfImg = 10;

int main()
{

    std::random_device rd;     // only used once to initialise (seed) engine
    std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)


    ifstream negList(negPATH+ "/neg.lst");
        if(!negList.is_open()){
            cout << "please check the path of neg.lst"  << endl;
            return -1;
        }
        ofstream outList(string(DATAPATH + "neg.lst"),ofstream::out); // clear all the things in neg.lst
        outList.close();
        outList.open(DATAPATH + "neg.lst",ofstream::app);

        string Line;
        Mat img;

        if(!outList.is_open()){
            cout << "error writing img path to the neg.lst" << endl;
            return -1;
        }
        int count = 0;
        while(getline(negList,Line)){
            string PATH = DATAPATH + Line;
            img = imread(PATH);


            if(!img.empty() &&
                    img.rows > 64 &&
                    img.cols > 128){
                std::uniform_int_distribution<int> row(0,img.rows-64); // guaranteed unbiased // 概率相同
                std::uniform_int_distribution<int> col(0,img.cols-128); // guaranteed unbiased // 概率相同

                int t = numOfImg;
                while(t--){
                    int x = col(rng);
                    int y = row(rng);
                    string filename = to_string(count) + ".png";
                    string path = DATAPATH + "neg/" + filename;
                    Mat grub(img,cv::Rect(x,y,128,64));
                    imwrite(path,grub);
                    outList << path << '\n' ;
                    ++count;
                }


            }else
                cout << "please check the neg files" << endl;
        }



    negList.close();
}
