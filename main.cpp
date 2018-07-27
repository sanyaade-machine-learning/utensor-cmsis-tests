#include "mbed.h"
#include "tensor.hpp"
#include "context.hpp"
#include "FATFileSystem.h"
#include "F413ZH_SD_BlockDevice.h"
#include "FullyConnectedOps.hpp"
#include "softmax.hpp"


F413ZH_SD_BlockDevice bd;
FATFileSystem fs("fs");



template <typename T1, typename T2, typename T3>
void test_FC(int scratch_space, T1 x, T2 y){
    Context ctx;
    ctx.gc();

    S_TENSOR A = ctx.add(new RamTensor<T1>({2,2}), "A");
    S_TENSOR B = ctx.add(new RamTensor<T2>({2,1}), "B");
    S_TENSOR C = ctx.add(new RamTensor<T2>({2,1}), "C");
    S_TENSOR out = ctx.add(new RamTensor<T1>({2,1}), "out");
    S_TENSOR b_shift = ctx.add(new RamTensor<uint16_t>({1}), "bshift");
    S_TENSOR o_shift = ctx.add(new RamTensor<uint16_t>({1}), "oshift");
    S_TENSOR scratch = ctx.add(new RamTensor<q15_t>({2}), "scratch"); //dim_vec only needed for arm_fully_connected_q7{_opt}, the rest are 0 

   // A.resize({2,2});
    //T1* a = A->write<T1>(0, sizeof(T1));
    T1* a = A->write<T1>(0, 0);
    a[0] = 1;
    a[1] = 2;
    a[2] = 3;
    a[3] = 4;
    
    T2* b = B->write<T2>(0, 0);
    b[0] = 5;
    b[1] = 6;
    
    T3* c = C->write<T3>(0, 0);
    c[0] = 7;
    c[1] = 8;
    //*(b_shift->write<uint16_t>(0, sizeof(uint16_t))) = 0;
    //*(o_shift->write<uint16_t>(0, sizeof(uint16_t))) = 0;
    *(b_shift->write<uint16_t>(0, 0)) = 0;
    *(o_shift->write<uint16_t>(0, 0)) = 0;

    TNameList inputs = {"B", "A", "C", "bshift", "oshift", "scratch"};
    TNameList outputs = {"out"};
  
    ctx.push_static(hold(new FullyConnectedLayerCmsisOp<T1,T2,T3>()), "FcOp", inputs, outputs);
    ctx.eval();
    
    //S_TENSOR out_s = ctx.get({"out"});
    const T1* out_1 = out->read<T1>(0, 0);
    // Should be [24]
    //           [47]
    printf("[%d]\n\r", out_1[0]);
    printf("[%d]\n\r", out_1[1]);

}

void test_mat_mul(){
    printf("Testing FC\n\r");
    printf("Testing q7_t\n\r");
    test_FC<q7_t, q7_t, q7_t>(2, 0, 0);
    
    printf("Testing q15_t\n\r");
    test_FC<q15_t, q15_t, q15_t>(2, 0, 0);
    
    printf("Testing q15_t q7_t hybrid\n\r");
    test_FC<q15_t, q7_t, q7_t>(2, 0, 0);
}

template <typename T1>
void do_softmax(T1 x){
    Context ctx;
    ctx.gc();

    S_TENSOR A = ctx.add(new RamTensor<T1>({4,1}), "A");
    S_TENSOR out = ctx.add(new RamTensor<T1>({4,1}), "out");

   // A.resize({2,2});
    //T1* a = A->write<T1>(0, sizeof(T1));
    T1* a = A->write<T1>(0, 0);
    a[0] = 1;
    a[1] = 2;
    a[2] = 3;
    a[3] = 4;
    
    TNameList inputs = {"A"};
    TNameList outputs = {"out"};
  
    ctx.push_static(hold(new SoftmaxCmsisOp<T1>()), "SoftmaxOp", inputs, outputs);
    ctx.eval();
    
    //S_TENSOR out_s = ctx.get({"out"});
    const T1* out_1 = out->read<T1>(0, 0);
    // Should be [24]
    //           [47]
    T1 sum = 0;
    for(int i = 0; i < 4; i++)
        sum += out_1[i];

    printf("[%f]\n\r", (float)out_1[0]/((float) sum));
    printf("[%f]\n\r", (float)out_1[1]/((float) sum));
    printf("[%f]\n\r", (float)out_1[2]/((float) sum));
    printf("[%f]\n\r", (float)out_1[3]/((float) sum));

}
void test_softmax(){
    printf("Testing softmax\n\r");
    printf("Testing q7\n\r");
    do_softmax<q7_t>(0);
    printf("Testing q15\n\r");
    do_softmax<q15_t>(0);
}

int main(){
    printf("Starting Test\n\r");
    ON_ERR(bd.init(), "SDBlockDevice init ");
    ON_ERR(fs.mount(&bd), "Mounting the filesystem on \"/fs\". ");

    test_mat_mul();
    test_softmax();


}
