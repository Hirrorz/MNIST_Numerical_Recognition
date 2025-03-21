#pragma once
#include<bits/stdc++.h>
#include<random>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
#define lf float
#define L(i, j, k) for(int i = (j); i <= (k); ++i)
#define R(i, j, k) for(int i = (j); i >= (k); --i)
#define sz(a) ((int) (a).size())
const int N = 2e6 + 5;
lf Learning_rate = 0.0007;
int dfn[N], dfn_tot = 0;//拓扑结构
random_device rd; mt19937 g(rd());
static default_random_engine Rand_;
static uniform_real_distribution<lf> u(-1, 1);

/*****神经元的参数******/
struct edge {
    int u, v; lf w;
}Tentacle[N*10];
vector<int>Neuron[N];//神经末梢编号
lf(*Type[N])(lf, bool);
lf B[N];//偏置
int edge_tot = 0, IN[N], in[N];
/********************/
int input_num = 1, output_num = 1;

/*********激活函数*************/
namespace tentacle {
    lf Sigmoid(lf x, bool k = 0) {
        lf y = 1 / (1 + exp(-x));
        if (k)return y * (1 - y);
        else return y;
    };
    lf Tanh(lf x, bool k = 0) {
        return 2.0 * Sigmoid(x * 2, k) - 1;
    };
    lf ReLU(lf x, bool k = 0) {
        if (k)return x >= 0.0;
        else return max<lf>(0, x);
    };
    lf Leaky_ReLU(lf x, bool k = 0) {
        if (k)return (x >= 0? 1:0.1);
        else return max<lf>(0.1*x, x);
    };
    lf Line(lf x, bool k = 0) {
        if (k)return 1;
        else return x;
    };
    lf getMSEloss(lf x1, lf x2) {
        return (x1 - x2) * (x1 - x2);
    };
    lf X_2(lf x, bool k = 0) {
        if (k)return x * 2;
        else return x * x;
    };
};
/*********激活函数*************/


void Save_NetWork_Configuration(const string& filename) {//保存神经网络参数
    ofstream outFile(filename, ios::out);
    if (outFile.is_open()) {
        outFile << dfn_tot << '\n';
        L(i, 1, dfn_tot)outFile << B[i] << ' ';
        outFile << '\n';
        outFile << edge_tot << '\n';
        L(i, 1, edge_tot) {
            auto [u, v, w] = Tentacle[i];
            outFile << u << ' ' << v << ' ' << w << '\n';
        }
        outFile.close();
    }
    else cout << "open failed\n";
}
void Load_NetWork_Configuration(const string& filename) {//载入神经参数
    ifstream inFile(filename, std::ios::in);
    int x;
    inFile >> x;
    L(i, 1, x)inFile >> B[i];
    inFile >> x;
    L(i, 1, x) {
        auto& [u, v, w] = Tentacle[i];
        inFile >> u >> v >> w;
    }
    inFile.close();
}


