#include<bits/stdc++.h>
#include"Datamaker.h"
using namespace std;
vector<vector<lf>> input_data, input_label;
vector<vector<int>>Layer;
vector<int>layer;
void connect(int x, int y) {
    Tentacle[++edge_tot] = { x,y,u(Rand_) };
    Neuron[x].push_back(edge_tot);
    IN[y]++; in[y] = IN[y];
}
void init() {
    queue<int> q;
    L(i, 0, sz(Layer.front())-1)q.push(Layer.front()[i]);
    while (sz(q)) {
        int u = q.front();
        q.pop();
        dfn[++dfn_tot] = u;
        for (auto& i : Neuron[u]) {
            auto& e = Tentacle[i];
            int v = e.v;
            in[v]--;
            if (!in[v])q.push(v);
        }
    }
}
void softmax(vector<lf>& a) {
    lf mx=-1e15,sum=0;
    L(i, 0, sz(a)-1)mx = max(mx, a[i]);
    L(i, 0, sz(a) - 1)a[i] -= mx;
    L(i, 0, sz(a) - 1)a[i] = exp(a[i]),sum+= a[i];
    L(i, 0, sz(a) - 1)a[i] /= sum;
}
vector<lf> forward(vector<lf>& data) {
    vector<lf> Y(dfn_tot + 1);
    L(i, input_num + 1, dfn_tot)Y[i] = B[i];
    L(i, 0, sz(Layer.front())-1)Y[Layer.front()[i]] = data[i];
    L(i, 1, dfn_tot) {
        int u = dfn[i];
        Y[u] = Type[u](Y[u], 0);
        for (auto& to : Neuron[u]) {
            auto& e = Tentacle[to];
            Y[e.v] += e.w * Y[u];
        }
    }
    vector<lf>Ans(sz(Layer.back()));
    L(i, 0, sz(Layer.back())-1)Ans[i] = Y[Layer.back()[i]];
    softmax(Ans);
    return Ans;
}
lf Beta = 0.9; 
lf v_weight[N] = { 0.0 }, v_bias[N] = { 0.0 },s_bias[N], s_weight[N];
void train(vector<vector<lf>>& Data, vector<vector<lf>>& Label,int S) {// 训练函数，使用梯度下降法优化网络参数
    L(i, 1, dfn_tot)s_bias[i] = 0;
    L(i, 1, edge_tot)s_weight[i] = 0;
    for (int T = 0; T < S;++T) {// 样本
        int t =1ll*g()%sz(Data)*g()%sz(Data);
        auto& data = Data[t];
        auto& label = Label[t];
        vector<lf> Y(dfn_tot + 1), deta(dfn_tot + 1), OUT(dfn_tot + 1),Ans(sz(Layer.back()));
        L(i, input_num + 1, dfn_tot)Y[i] = B[i];
        L(i, 0, sz(Layer.front())-1)Y[Layer.front()[i]] = data[i] / 255;//输入层数据导入
        L(i, 1, dfn_tot) {
            int u = dfn[i];
            deta[u] = Type[u](Y[u], -1);//非线性变换后，对其的导数
            OUT[u] = Type[u](Y[u], 0);//非线性变换结果
            for (auto& to : Neuron[u]) {
                auto& e = Tentacle[to];
                int v = e.v;
                Y[v] += e.w * OUT[u];
            }
        }
        L(i, 0, sz(Layer.back()) - 1)Ans[i] = Y[Layer.back()[i]];
        softmax(Ans);
        L(i, 0, sz(Layer.back()) - 1){
            int u = Layer.back()[i];
            Y[u] = deta[u] * 2.0 * (Ans[i] - label[i]);
            s_bias[u] += Y[u];
        }
        R(i, dfn_tot, 1) {
            int u = dfn[i];
            if (!sz(Neuron[u]))continue;
            Y[u] = 0;
            for (auto& to : Neuron[u]) {
                auto& e = Tentacle[to];
                int v = e.v;
                Y[u] += Y[v] * e.w * deta[u];
                s_weight[to] += Y[v] * OUT[u];
            }
            s_bias[u] += Y[u];
        }
    }
    L(i, input_num + 1, dfn_tot) {
        s_bias[i] /= S;
        v_bias[i] = Beta * v_bias[i] + (1 - Beta) * s_bias[i];// 使用动量更新偏置
        B[i] -= v_bias[i] * Learning_rate;
    }
    L(i, 1, edge_tot) {
        s_weight[i] /= S;
        v_weight[i] = Beta * v_weight[i] + (1 - Beta) * s_weight[i];// 使用动量更新权重
        auto& e = Tentacle[i];
        e.w -= v_weight[i] * Learning_rate;
    }
}
void predict(vector<vector<lf>>& test_data, vector<vector<lf>>& test_label,int t=0) {
    int cnt = 0,i,ans,r=t? t: sz(test_data);
    lf loss = 0,mx;
    L(s,0,r-1) {
        if (t)i = 1ll * g() % sz(test_data) * g() % sz(test_data);
        else i = s;
        auto& Data = test_data[i];
        auto& Label = test_label[i];
        auto pred = forward(Data);       // 前向传播预测值
        mx = -1e5,ans = 0;
        L(j, 0, sz(pred) - 1)if (pred[j] > mx)mx = pred[j], ans = j;
        L(j, 0, sz(pred) - 1)loss += tentacle::getMSEloss(pred[j], Label[j]);
        cnt += (test_label[i][ans] == 1);
    }
    cout << " loss: " << loss << "\ncorrect rate:" << 1.0 * cnt /r << '\n';
}
int main() {
    Learning_rate = 0.01;
    srand(time(0));
    Rand_.seed(time(0));
    //Make_Data();
    //Build_Data();
    Load_Data("C:/Users/ASUS/Desktop/data/train_data.txt", input_data, input_label);
    cout << "Load Success\n";

    int now = 1;

    input_num = sz(input_data[0]);
    Layer.push_back(layer);
    L(i, 0, input_num-1)Type[now] = tentacle::Line, Layer.back().push_back(now), now++;
    Layer.push_back(layer);

    int MidLayer_num = 15;
    L(i, 1, MidLayer_num) {
        Layer.back().push_back(now);
        Type[now] = tentacle::Leaky_ReLU;
        now++;
    }
    Layer.push_back(layer);

    output_num = sz(input_label[0]);
    L(i, 0, output_num-1) {
        Layer.back().push_back(now);
        Type[now] = tentacle::Line;
        now++;
    }
    L(i, 0, 1)for (auto& u : Layer[i]) for (auto& v : Layer[i + 1])connect(u, v);
    init();
    L(k, 1, 200000) { //迭代次数
        train(input_data, input_label, 1);
        if (k % 10000)continue;
        cout << "Time: " << k << '\n';
        predict(input_data, input_label, 1000);
    }
    predict(input_data, input_label);
    system("pause");
    return 0;
}
