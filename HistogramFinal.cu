#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include "imageio.h"

__global__ void calculateHistogramStride(unsigned char *d_greyImage, int *d_histogram, int size)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    printf("id:%d\n",blockDim.x);
    if(id<size)
        atomicAdd(&(d_histogram[(id%blockDim.x)*1024+d_greyImage[id]]),1);
    __syncthreads();
}

//Cuda kernel to apply histogram equalization method for image enhacement
__global__ void histogram_equalization(unsigned char *d_greyImage, int *d_histogram,unsigned char *d_enhanced, int size)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<size)
        d_enhanced[id] = d_histogram[d_greyImage[id]];
    __syncthreads();
}

int main(int argc, char** argv)
{
    if(argc != 2){
        printf("Execute code in below format[Image must be in a ppm format]\n");
        printf("%s input_image\n", argv[0]);
        exit(-1);
    }
    
    PPMImage *image;
    char *input_image = argv[1];
    
    //Reading Image
    image = readPPM(input_image);
    int size = sizeof(unsigned char)*(image->x)*(image->y);
    int totalObservation = (image->x)*(image->y);

    //Memory Allocation and initialization of host variables
    unsigned char *h_rgb = (unsigned char *) malloc(size*3);
    unsigned char *h_greyImage = (unsigned char *) malloc(size);
    unsigned char *h_enhanced = (unsigned char *) malloc(size);
    unsigned char *d_greyImage;
    unsigned char *d_enhanced;
    unsigned int *h_histogram = (unsigned int *) malloc(sizeof(unsigned int)*256*1024);
    int *d_histogram;

    //Memory Allocation of cuda variables 
    cudaMalloc(&d_greyImage,size);
    cudaMalloc(&d_enhanced,size);
    cudaMalloc(&d_histogram,sizeof(int)*256);
    cudaMemset(d_histogram, 0, 1024*256*sizeof(int));

    //Cuda Variables to calculate execution time
    cudaEvent_t start,stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //convert RGB image to grayscale for better image enhacement
    getGrayArray(image,h_rgb,h_greyImage);
    cudaMemcpy(d_greyImage,h_greyImage,size,cudaMemcpyHostToDevice);
    printf("Image Dimention: %dx%d pixels\n",imageWidth,imageHeight);

    //Required variables for executing CUDA kernel
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);
    int blocks = prop.multiProcessorCount;
    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (int)ceil((float)size/blockSize);

    cudaEventRecord(start,0);
    calculateHistogramStride<<<gridSize,blockSize>>>(d_greyImage,d_histogram,size);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop);
    printf("Time Required for Creating Histogram: %3.5f ms\n",elapsedTime);

    cudaMemcpy(h_histogram,d_histogram,sizeof(int) * 256 * 1024, cudaMemcpyDeviceToHost);

    int sum = 0;
    for(int i=0;i<1024*256;i++)
    {
        for(int j=0;j<256;j++)
            sum+=h_histogram[i*256+j];
    }
    printf("Total sum: %d\n",sum);
    /*
    //Calculating cummulative probabilities and new gray values for enhanced Image
    float cummulative = 0;
    for(int i=0;i<256;i++)
    {
        //printf("Cummulative Probability of gray image for value %d:%d\n",i,histogram[i]);
        cummulative = cummulative+(h_histogram[i]*1.0/totalObservation);
        h_histogram[i] = cummulative*255;
    }
    cudaMemcpy(d_histogram,h_histogram,sizeof(int) * 256,cudaMemcpyHostToDevice);
    //printf("Total Elements:%d",sum);

    //cudaEventRecord(start,0);
    histogram_equalization<<<gridSize,blockSize>>>(d_greyImage,d_histogram,d_enhanced,size);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop);
    printf("Total time required for Enhancing Image: %3.5f ms\n",elapsedTime);
    cudaMemcpy(h_enhanced,d_enhanced,size,cudaMemcpyDeviceToHost);

    //Create output contrast enhanced Image
    char *token = strtok(input_image, ".");
    const char *filename = strcat(token,".pgm");
    const int maxColorComponentValue = 255;
    FILE *fp;
    const char *comment = "# this is my new binary pgm file";
    fp = fopen(filename, "wb");
    // write header to the file 
    fprintf(fp, "P5\n %s\n %d\n %d\n %d\n", comment, imageWidth, imageHeight,maxColorComponentValue);
    // write image data bytes to the file
    fwrite(h_enhanced, imageHeight*imageWidth, 1, fp);
    fclose(fp);
    printf("OK - file %s saved\n", filename);
    */
    //Deallocating Memories from host and device
    free(h_rgb);
    free(h_greyImage);
    free(h_histogram);
    free(h_enhanced);
    cudaFree(d_greyImage);
    cudaFree(d_histogram);
    cudaFree(d_enhanced);
    return 0;
}