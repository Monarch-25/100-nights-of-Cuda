#include<cuda_runtime.h>
#include<iostream>

__global__ void vecAddKernel(float *A,float *B,float *C,int n){ //Global callable from host but executed on device
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<n){
        //if interested in checking the order of threads
        // printf("Current thread: %d\n",i); 
        //notice the threads are asychronous but are always in contiguous order of 32
        C[i] =  A[i] + B[i]; 
    }
}

void vecAddGPU(float *A_h,float *B_h,float *C_h,int n){
    size_t size = n*sizeof(float);
    float *A_d,*B_d,*C_d;


    //Allocate memory for A,B,C on device
    cudaMalloc((void **)&A_d,size); //allocates object in the device global memory 
    cudaMalloc((void **)&B_d,size);
    cudaMalloc((void **)&C_d,size);
    //the address is stores in device pointer 
    //the device global memory pointer should not be deferenced in the host code as it can cause exceptions or runtime errors

    //Copy A and B to device
    cudaMemcpy(A_d,A_h,size,cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B_h,size,cudaMemcpyHostToDevice);



    //call kernel to perform actual operation by launching a grid of threads
    //<<<blocks,threads per block>>>
    vecAddKernel<<<ceil(n/256.0),256>>>(A_d,B_d,C_d,n);


    //copy C from the device memory to host memory
    cudaMemcpy(C_h,C_d,size,cudaMemcpyDeviceToHost);
    //free the device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

void fill(float *V,int n){
    for(int i=0;i<n;i++){
        V[i] = rand()%100;
    }
}


int main(){
    int n = 1024;
    float A_h[n];
    float B_h[n];
    float C_h[n];
    fill(A_h,n);
    fill(B_h,n);

    vecAddGPU(A_h,B_h,C_h,n);

    // Verify results
    bool success = true;
    for (int i = 0; i < n; i++) {
        if (A_h[i] + B_h[i] != C_h[i]) {
            success = false;
        } else {
            continue;
        }
    }

    if (success) {
        std::cout << "All tests passed!" << std::endl;
    } else {
        std::cout << "Some tests failed." << std::endl;
    }

    std::cout<<"Aman Day 1\n";
    return 0;
}