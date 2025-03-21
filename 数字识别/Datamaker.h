#pragma once
#include"HEAD.h"
using namespace std;
using namespace cv;
vector<vector<lf>> Input__data;
vector<vector<lf>> Input__label;
void Make_Data(){
    string path = "C:/Users/ASUS/Desktop/data";
    L(i, 0, 9) {
        vector<lf> Label(10,0),T;
        Label[i] = 1;
        string Data = path + cv::format("/%01d", i);
        vector<String> src_name;
        glob(Data, src_name, false);
        if (sz(src_name) == 0) {
            cout << "That's no file in " << Data << endl;
            exit(1);
        }
        for (auto& sc : src_name) {
            cout << sc << '\n';
            Mat image = imread(sc);
            Input__label.push_back(Label);
            T.clear();
            L(j, 0, image.rows - 1)L(k, 0, image.cols - 1){
                int sum = 0;
                L(p, 0, 2)sum += (int)image.at<Vec3b>(j, k)[p];
                T.push_back(sum / 3);
            }
            Input__data.push_back(T);
        }
    }
}
void Save_Data(const string& filename) {//训练数据
    ofstream outFile(filename, ios::out);
    if (outFile.is_open()) {
        outFile << sz(Input__data) << '\n';
        for (auto& a : Input__data) {
            outFile << sz(a) << ' ';
            for (auto& v : a)outFile << v << ' ';
            outFile << '\n';
        }
        for (auto& a : Input__label) {
            outFile << sz(a) << ' ';
            for (auto& v : a)outFile << v << ' ';
            outFile << '\n';
        }
        outFile.close();
    }
    else cout << "open failed\n";
}
void Load_Data(const string& filename, vector<vector<lf>>& input_data, vector<vector<lf>>& input_label) {//载入训练数据
    ifstream inFile(filename, ios::in);
    inFile.tie(0);
    vector<lf>t;
    if (inFile.is_open()) {
        int T, num;
        lf x;
        inFile >> T;
        L(i, 1, T) {
            inFile >> num;
            t.clear();
            L(j, 1, num)inFile >> x, t.push_back(x);
            input_data.push_back(t);
        }
        L(i, 1, T) {
            inFile >> num;
            t.clear();
            L(j, 1, num)inFile >> x, t.push_back(x);
            input_label.push_back(t);
        }
        inFile.close();
    }
    else cout << "open failed\n";
}
void Build_Data() {
    Save_Data("C:/Users/ASUS/Desktop/data/train_data.txt");
    Save_Data("C:/Users/ASUS/Desktop/data/test_data.txt");
}