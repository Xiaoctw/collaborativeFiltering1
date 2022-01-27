/*
<%
cfg['compiler_args'] = ['-std=c++11', '-undefined dynamic_lookup']
%>
<%
setup_pybind11(cfg)
%>
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <time.h>
#include<map>
//##include <map>

typedef unsigned int ui;

using namespace std;
namespace py = pybind11;

int randint_(int end)
{
/**
生成随机数，最大值为end
*/
    return rand() % end;
}

float randfloat_(int end){
       float val=(rand()%1007)/1007.0;
       return val*end;
}

int search(std::vector<float> & vector1, float value){
    /**
    对概率进行search，根据生成的随机数选择采样物品的下表
    */
    long beg=0;
    long end=vector1.size()-1;
    long res=end;
    while (beg<=end){
        long mid=(end-beg)/2+beg;
//        if (vector1[mid]==value){
//            return mid;
//        }
        if (vector1[mid]>value){
            end=mid-1;
//            res=end;
        } else{ //vector1[mid]<val
            res=mid;
            beg=mid+1;
        }
    }
    return res;
}


py::array_t<int> sample_negative(int user_num, int item_num, int train_num, std::vector<std::vector<int>> allPos, int neg_num)
{
    int perUserNum = train_num / (user_num+1);
    int row = neg_num + 2;
   // py::array_t<int> S_array = py::array_t<int>({user_num * perUserNum, row});
    py::array_t<int> S_array=py::array_t<int>({train_num,row});
    py::buffer_info buf_S = S_array.request();
    int *ptr = (int *)buf_S.ptr;
    int i=0;
    while(i<train_num){
        int tem_user=randint_(user_num);
        std::vector<int> pos_items=allPos[tem_user];
        if(pos_items.size()==0){
            continue;
        }
        int pos_index=randint_(pos_items.size());
        int pos_item=pos_items[pos_index];
        while(true){
            int neg_item=randint_(item_num);
//            if(pos_items.find(neg_item)!=pos_items.end()){
              if(find(pos_items.begin(),pos_items.end(),neg_item)!=pos_items.end()){
                continue;
            }else{
                  ptr[i*row]=tem_user;
                  ptr[i*row+1]=pos_item;
                  ptr[i*row+2]=neg_item;
                  break;
            }
        }
        i+=1;
    }

    return S_array;
}


py::array_t<int> sample_negative_score(int user_num, int item_num, int train_num, std::vector<std::vector<int>> allPos, std::vector<std::vector<int>> allPosScore,int neg_num)
{
    //n_user
    int perUserNum = train_num / (user_num+1);
    int row = neg_num + 3;
    py::array_t<int> S_array = py::array_t<int>({train_num, row});
    py::buffer_info buf_S = S_array.request();
    int *ptr = (int *)buf_S.ptr;
   // for (int user = 0; user < user_num; user++)
   // {
     //   std::vector<int> pos_item = allPos[user];
      //  std::vector<int> pos_score=allPosScore[user];
      //  for (int pair_i = 0; pair_i < perUserNum; pair_i++)
      //  {
      //      int negitem = 0;
       //     ptr[(user * perUserNum + pair_i) * row] = user;
       //     int idx=randint_(pos_item.size());
       //     ptr[(user * perUserNum + pair_i) * row + 1] = pos_item[idx];
       //     for (int index = 2; index < neg_num + 2; index++)
       //     {
         //       do
           //     {
        //            negitem = randint_(item_num);
        //        } while (
        //            find(pos_item.begin(), pos_item.end(), negitem) != pos_item.end());
        //        ptr[(user * perUserNum + pair_i) * row + index] = negitem;
        //    }
        //    ptr[(user * perUserNum + pair_i) * row + neg_num+2]=pos_score[idx];
      //  }
 //   }
    int i=0;
    while(i<train_num){
        int tem_user=randint_(user_num);
        std::vector<int> pos_items=allPos[tem_user];
        std::vector<int> pos_items_scores=allPosScore[tem_user];
        if(pos_items.size()==0){
            continue;
        }
        int pos_index=randint_(pos_items.size());
        int pos_item=pos_items[pos_index];
        int pos_item_score=pos_items_scores[pos_index];
        while(true){
        //这里可以考虑加上概率
            int neg_item=randint_(item_num);
//            if(pos_items.find(neg_item)!=pos_items.end()){
              if(find(pos_items.begin(),pos_items.end(),neg_item)!=pos_items.end()){
                continue;
                    }else{
                  ptr[i*row]=tem_user;
                  ptr[i*row+1]=pos_item;
                  ptr[i*row+2]=neg_item;
                  ptr[i*row+3]=pos_item_score;
                  break;
            }
        }
        i+=1;
    }


    return S_array;
}

py::array_t<int> sample_negative_ByUser(std::vector<int> users, int item_num, std::vector<std::vector<int>> allPos, int neg_num)
{
    int row = neg_num + 2;
    int col = users.size();
    py::array_t<int> S_array = py::array_t<int>({col, row});
    py::buffer_info buf_S = S_array.request();
    int *ptr = (int *)buf_S.ptr;

    for (int user_i = 0; user_i < users.size(); user_i++)
    {
        int user = users[user_i];
        std::vector<int> pos_item = allPos[user];
        int negitem = 0;

        ptr[user_i * row] = user;
        ptr[user_i * row + 1] = pos_item[randint_(pos_item.size())];

        for (int neg_i = 2; neg_i < row; neg_i++)
        {
            do
            {
                negitem = randint_(item_num);
            } while (
                find(pos_item.begin(), pos_item.end(), negitem) != pos_item.end());
            ptr[user_i * row + neg_i] = negitem;
        }
    }
    return S_array;
}


py::array_t<int> sample_negative_score_prob(int user_num, int item_num, int train_num, std::vector<std::vector<int>> allPos, std::vector<std::vector<int>> allPosScore,std::vector<float> sum_prob,int neg_num)
{
    //n_user
    int perUserNum = train_num / (user_num+1);
    int row = neg_num + 3;
    py::array_t<int> S_array = py::array_t<int>({train_num, row});
    py::buffer_info buf_S = S_array.request();
    int *ptr = (int *)buf_S.ptr;


    float sum_val=0;
    vector<float> vector1;
    vector1.push_back(0);
    for(int i=0;i<sum_prob.size();i++){
        sum_val+=sum_prob[i];
        vector1.push_back(sum_val);
    }
    sum_prob=vector1;


    int sum_prob_int=(int) sum_prob[sum_prob.size()-1]; //这个值为设定的最大值
    map<int,int> items2cnt;
    int i=0;
    while(i<train_num){
        int tem_user=randint_(user_num);
        std::vector<int> pos_items=allPos[tem_user];
        std::vector<int> pos_items_scores=allPosScore[tem_user];
        if(pos_items.size()==0){
            continue;
        }
        int pos_index=randint_(pos_items.size());
        int pos_item=pos_items[pos_index];
        int pos_item_score=pos_items_scores[pos_index];
        while(true){
        //这里可以考虑加上概率
//            int neg_item=randint_(item_num);
//            if(pos_items.find(neg_item)!=pos_items.end()){
//              float tem_rand_val=randfloat_(sum_prob_int);
              float tem_rand_val=(float)randint_(sum_prob_int);
              int neg_item=search(sum_prob,tem_rand_val); //这个就是采样得到的负类样本
              items2cnt[neg_item]+=1;
              if(find(pos_items.begin(),pos_items.end(),neg_item)!=pos_items.end()){
                    continue;
                    }
                    else{
                  ptr[i*row]=tem_user;
                  ptr[i*row+1]=pos_item;
                  ptr[i*row+2]=neg_item;
                  ptr[i*row+3]=pos_item_score;
                  break;
            }
        }
        i+=1;
    }
//    for(i=0;i<item_num;i++){
//        cout<<"item: "<<i<<" prob: "<<sum_prob[i+1]-sum_prob[i]<<" cnt: "<<items2cnt[i]<<endl;
//    }
    return S_array;
}

void set_seed(unsigned int seed)
{
    srand(seed);
}

using namespace py::literals;

PYBIND11_MODULE(sampling, m)
{
    srand(time(0));
    // srand(2020);
    m.doc() = "example plugin";
    m.def("randint", &randint_, "generate int between [0 end]", "end"_a);
    m.def("seed", &set_seed, "set random seed", "seed"_a);
    m.def("sample_negative", &sample_negative, "sampling negatives for all","user_num"_a, "item_num"_a, "train_num"_a, "allPos"_a, "neg_num"_a);
    m.def("sample_negative_score", &sample_negative_score,"sampling negatives scores for all","user_num"_a, "item_num"_a, "train_num"_a, "allPos"_a,"allPosScore"_a, "neg_num"_a);
    m.def("sample_negative_ByUser", &sample_negative_ByUser, "sampling negatives for given users","users"_a, "item_num"_a, "allPos"_a, "neg_num"_a);
    m.def("sample_negative_score_prob", &sample_negative_score_prob,"sampling negatives scores for all with sum prob","user_num"_a, "item_num"_a, "train_num"_a, "allPos"_a,"allPosScore"_a, "sum_prob"_a, "neg_num"_a);
}