#include<cuda_runtime.h>


__global__ void naiveMM(float *M,float *N,float *out,int width){
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    if(row<width && col<width){
        float val = 0;
        for(int k=0;k<width;k++){
            val+=M[row*width+k]*N[k*width+col];
        }
        out[row*width+col] = val;
    }
}


int main(){
    return 0;
}