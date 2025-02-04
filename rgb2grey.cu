__global__ void rgbToGrey(cont unsigned char *Pin,unsigned char *Pout,int width,int height){
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    if(row<height and col<width){
        int greyOffset = width*row + col;
        
        //In an rgb image we have three color channels thus the total pixels = greypixel*3
        //Now suppose we have a (2,2) matrix such that:
        // Pixel (0, 0): [R=255, G=0,   B=0]
        // Pixel (0, 1): [R=0,   G=255, B=0]
        // Pixel (1, 0): [R=0,   G=0,   B=255]
        // Pixel (1, 1): [R=128, G=128, B=128]
        //we know that the 2-D visualation is just for our understanding but in the memory they are essential layed 'flat'
        //therefore in memory the data is stored as
        //[255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128]
        //[r  , g, b, r,   g, b, r, g,   b,   r,   g,   b]
        //Hopefully the offset will make more sense now

        int rgbOffset = greyOffset*3;
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        // Pout[greyOffset] = 0.21f*r+0.71f*g+0.07f*b;

        //Floating-point arithmetic can be expensive on GPUs. 
        //By precomputing the weights as integers or fixed-point values we can improve the kernel performance:
        Pout[greyOffset] = (unsigned char)((77 * r + 150 * g + 29 * b) / 256);
    }
}