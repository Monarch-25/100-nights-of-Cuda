#include<cuda_runtime.h>
#include<iostream>


__global__ void matAdd(const float *A,const float *B,float *C,int N){
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;

    if(col<N && row <N){
        C[N*row + col] = A[N*row + col] + B[N*row + col];
    }
}


int main(){
    float *A;
    float *B;
    float *C;

    const int N = 1024;

    A = (float *)malloc(N*N*sizeof(float));
    B = (float *)malloc(N*N*sizeof(float));
    C = (float *)malloc(N*N*sizeof(float));

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            A[i*N+j] = 1.0f;
            B[i*N+j] = 2.0f;
            C[i*N+j] = 0.0f;
        }
    }

    float *A_d,*B_d,*C_d;
    cudaMalloc((void **)&A_d,N*N*sizeof(float));
    cudaMalloc((void **)&B_d,N*N*sizeof(float));
    cudaMalloc((void **)&C_d,N*N*sizeof(float));

    cudaMemcpy(A_d,A,N*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B,N*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(C_d,C,N*N*sizeof(float),cudaMemcpyHostToDevice);

    dim3 blockDim(32,16);
    dim3 gridDim(ceil(N/32.0),ceil(N/16.0));

    matAdd<<<gridDim,blockDim>>>(A_d,B_d,C_d,N);
    cudaMemcpy(C,C_d,N*N*sizeof(float),cudaMemcpyDeviceToHost);

    printf("Verify\n");
    bool success = true;
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            if((A[N*i+j]+B[N*i+j])!=C[N*i+j]){
                success = false;
                return -1;
            }
        }
    }
    if(success){
        printf("All test cases passed\n");
    }
    printf("Aman Day 2!!\n");

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    free(A);
    free(B);
    free(C);

    return 0;
}

