#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include "imageio.h"

int main(int argc, char** argv)
{
    if(argc != 2){
        printf("Execute code in below format[Image must be in a ppm format]\n");
        printf("%s input_image\n", argv[0]);
        exit(-1);
    }

    PPMImage *image;
    char *input_image = argv[1];
    char *output_image = argv[2];
    int histogram[256];
    
    image = readPPM(input_image);
    unsigned char *rgb = (unsigned char *) malloc(sizeof(unsigned char)*(image->x)*(image->y)*3);
    unsigned char *greyImage = (unsigned char *) malloc(sizeof(unsigned char)*(image->x)*(image->y));
    getGrayArray(image,rgb,greyImage);
    printf("Image Dimention: %dx%d pixels\n",imageWidth,imageHeight);
    
    clock_t start, end;
    int i=0,j=0;

    //Starting Clock Timer
    start=clock();

    //Histogram Initialization
    for(i=0;i<256;i++)
        histogram[i]=0;

    //Total Number of Pixels in a image
    int totalObservation = imageHeight*imageWidth;

    //Calculating Histogram
    for(i=0;i<imageHeight;i++)
    {
        for(j=0;j<imageWidth;j++)
        {
            //printf("%d\n",greyImage[i*imageWidth+j]);
            histogram[greyImage[i*imageWidth+j]]++;
        }
    }

    //Calculating new Histogram values from old histogram
    float cummulative = 0;
    for(i=0;i<256;i++)
    {
        //printf("Cummulative Probability of gray image for value %d:%d\n",i,histogram[i]);
        cummulative = cummulative+(histogram[i]*1.0/totalObservation);
        histogram[i] = cummulative*255;
    }

    //Applying Histogram Equalization 
    for(i=0;i<imageHeight;i++)
    {
        for(j=0;j<imageWidth;j++)
        {
            //printf("%d\n",greyImage[i*imageWidth+j]);
            greyImage[i*imageWidth+j] = histogram[greyImage[i*imageWidth+j]];
        }
    }

    //Ending Clock Timer
    end = clock();

    //Time taken by program to execute histogram equalization
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %0.10f millisecs\n", time_spent*1000);


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
    fwrite(greyImage, imageHeight*imageWidth, 1, fp);
    fclose(fp);
    printf("OK - file %s saved\n", filename);
    return 0;
}