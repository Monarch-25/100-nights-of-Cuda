#include<cuda_runtime.h>
#include <iostream>

#define WIDTH 4
#define HEIGHT 4
// #define BLUR_SIZE 1

__global__ void blur2D(unsigned char *Pin, unsigned char *Pout,int width, int height,int BLUR_SIZE){
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;

    if(col<width && row<height){
        int pixVal = 0;
        int pixels = 0;
        for(int blurRow=-BLUR_SIZE;blurRow<BLUR_SIZE+1;blurRow++){
            for(int blurCol=-BLUR_SIZE;blurCol<BLUR_SIZE+1;blurCol++){
                int curRow = row+blurRow;
                int curCol = col+blurCol;
                //check if the pixel is valid
                if(curRow>=0 && curRow<height && curCol<width&&curCol>=0){
                    pixVal += Pin[curRow*width + curCol];
                    pixels +=1;
                }
            }
        }
        Pout[row*width+col] = (unsigned char)(pixVal/pixels);
    }
}



void printImage(const unsigned char* image, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << (int)image[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Input image (4x4)
    unsigned char h_Pin[HEIGHT * WIDTH] = {
        10, 20, 30, 40,
        50, 60, 70, 80,
        90, 100, 110, 120,
        130, 140, 150, 160
    };

    // Output image (4x4)
    unsigned char h_Pout[HEIGHT * WIDTH];

    // Device memory pointers
    unsigned char *d_Pin, *d_Pout;

    // Allocate device memory
    cudaMalloc((void**)&d_Pin, sizeof(unsigned char) * WIDTH * HEIGHT);
    cudaMalloc((void**)&d_Pout, sizeof(unsigned char) * WIDTH * HEIGHT);

    // Copy input image to device
    cudaMemcpy(d_Pin, h_Pin, sizeof(unsigned char) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(2, 2);  // 2x2 threads per block
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (HEIGHT + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    blur2D<<<gridDim, blockDim>>>(d_Pin, d_Pout, WIDTH, HEIGHT, 1);

    // Copy result back to host
    cudaMemcpy(h_Pout, d_Pout, sizeof(unsigned char) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

    // Print the blurred image
    std::cout << "Blurred Image:" << std::endl;
    printImage(h_Pout, WIDTH, HEIGHT);

    // Free device memory
    cudaFree(d_Pin);
    cudaFree(d_Pout);

    return 0;
}