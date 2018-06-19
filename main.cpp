#include "mbed.h"
#include "tensor.hpp"
#include "context.hpp"
#include "FATFileSystem.h"
#include "F413ZH_SD_BlockDevice.h"
#include "FullyConnectedOps.hpp"


F413ZH_SD_BlockDevice bd;
FATFileSystem fs("fs");

Context ctx;

template <typename T1, typename T2, typename T3>
void test_FC(int scratch_space){
    ctx.gc();

    S_TENSOR A = ctx.add(new RamTensor<T1>({2,2}), "A");
    S_TENSOR B = ctx.add(new RamTensor<T2>({2,1}), "B");
    S_TENSOR C = ctx.add(new RamTensor<T2>({2,1}), "C");
    S_TENSOR out = ctx.add(new RamTensor<T3>({2,1}), "out");
    S_TENSOR b_shift = ctx.add(new RamTensor<uint16_t>({1}), "bshift");
    S_TENSOR o_shift = ctx.add(new RamTensor<uint16_t>({1}), "oshift");
    S_TENSOR scratch = ctx.add(new RamTensor<q15_t>({2}), "scratch"); //dim_vec only needed for arm_fully_connected_q7{_opt}, the rest are 0 

   // A.resize({2,2});
    q7_t* a = A->write<T1>(0, sizeof(T1));
    a[0] = 1;
    a[1] = 2;
    a[2] = 3;
    a[3] = 4;
    
    q7_t* b = B->write<T2>(0, sizeof(T2));
    b[0] = 5;
    b[1] = 6;
    
    q7_t* c = C->write<T2>(0, sizeof(T2));
    b[0] = 7;
    b[1] = 8;
    
    *(b_shift->write<uint16_t>(0, sizeof(uint16_t))) = 0;
    *(o_shift->write<uint16_t>(0, sizeof(uint16_t))) = 0;

    TNameList inputs = {"A", "B", "C", "bshift", "oshift", "scratch"};
    TNameList outputs = {"out"};
  
    ctx.push_static(hold(new FullyConnectedLayerCmsisOp<T1,T2,T3>()), "FcOp", inputs, outputs);
    ctx.eval();

}

int main(){
    ON_ERR(bd.init(), "SDBlockDevice init ");
    ON_ERR(fs.mount(&bd), "Mounting the filesystem on \"/fs\". ");


}
