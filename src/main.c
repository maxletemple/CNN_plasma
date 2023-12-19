#include "fixed.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <endian.h>
#include <string.h>
#include "cnn.h"



typedef struct _IdxFile
{
    int ndims;
    uint32_t *dims;
    uint8_t *data;
} IdxFile;

/* IdxFile_read(fp)
   Reads all the data from given fp.
*/
IdxFile *IdxFile_read(FILE *fp)
{
    /* Read the file header. */
    struct
    {
        uint16_t magic;
        uint8_t type;
        uint8_t ndims;
        /* big endian */
    } header;
    if (fread(&header, sizeof(header), 1, fp) != 1)
        return NULL;
#if DEBUG_IDXFILE
    fprintf(stderr, "IdxFile_read: magic=%x, type=%x, ndims=%u\n",
            header.magic, header.type, header.ndims);
#endif
    if (header.magic != 0)
        return NULL;
    if (header.type != 0x08)
        return NULL;
    if (header.ndims < 1)
        return NULL;

    /* Read the dimensions. */
    IdxFile *self = (IdxFile *)calloc(1, sizeof(IdxFile));
    if (self == NULL)
        return NULL;
    self->ndims = header.ndims;
    self->dims = (uint32_t *)calloc(self->ndims, sizeof(uint32_t));
    if (self->dims == NULL)
        return NULL;

    if (fread(self->dims, sizeof(uint32_t), self->ndims, fp) == self->ndims)
    {
        uint32_t nbytes = sizeof(uint8_t);
        for (int i = 0; i < self->ndims; i++)
        {
            /* Fix the byte order. */


            uint32_t size = self->dims[i];
#if DEBUG_IDXFILE
            fprintf(stderr, "IdxFile_read: size[%d]=%u\n", i, size);
#endif
            nbytes *= size;
            self->dims[i] = size;
        }
        /* Read the data. */
        self->data = (uint8_t *)malloc(nbytes);
        if (self->data != NULL)
        {
            fread(self->data, sizeof(uint8_t), nbytes, fp);
#if DEBUG_IDXFILE
            fprintf(stderr, "IdxFile_read: read: %lu bytes\n", n);
#endif
        }
    }

    return self;
}

/* IdxFile_destroy(self)
   Release the memory.
*/
void IdxFile_destroy(IdxFile *self)
{
    if (self->dims != NULL)
    {
        free(self->dims);
        self->dims = NULL;
    }
    if (self->data != NULL)
    {
        free(self->data);
        self->data = NULL;
    }
    free(self);
}

/* IdxFile_get1(self, i)
   Get the i-th record of the Idx1 file. (uint8_t)
 */
uint8_t IdxFile_get1(IdxFile *self, int i)
{
    return self->data[i];
}

/* IdxFile_get3(self, i, out)
   Get the i-th record of the Idx3 file. (matrix of uint8_t)
 */
void IdxFile_get3(IdxFile *self, int i, uint8_t *out)
{
    size_t n = self->dims[1] * self->dims[2];
    memcpy(out, &self->data[i * n], n);
}

#define INPUT_DEPTH 1
#define INPUT_WIDTH 28
#define INPUT_HEIGHT 28

#define CONV1_DEPTH 16
#define CONV1_WIDTH 14
#define CONV1_HEIGHT 14
#define CONV1_KERNSIZE 5
#define CONV1_PADDING 2
#define CONV1_STRIDE 2

#define CONV2_DEPTH 32
#define CONV2_WIDTH 7
#define CONV2_HEIGHT 7
#define CONV2_KERNSIZE 5
#define CONV2_PADDING 2
#define CONV2_STRIDE 2

#define FULL1_NEURONS 10


Layer linput;
Layer lconv1;
Layer lconv2;
Layer lfull1;

fixed_t conv1_weights[CONV1_DEPTH * INPUT_DEPTH * CONV1_KERNSIZE * CONV1_KERNSIZE];
fixed_t conv1_biases[CONV1_DEPTH];
fixed_t conv2_weights[CONV2_DEPTH * CONV1_DEPTH * CONV2_KERNSIZE * CONV2_KERNSIZE];
fixed_t conv2_biases[CONV2_DEPTH];
fixed_t full1_weights[FULL1_NEURONS * CONV2_DEPTH * CONV2_WIDTH * CONV2_HEIGHT];
fixed_t full1_biases[FULL1_NEURONS];

fixed_t input[INPUT_DEPTH * INPUT_WIDTH * INPUT_HEIGHT];
fixed_t inter1[CONV1_DEPTH * CONV1_WIDTH * CONV1_HEIGHT];
fixed_t inter2[CONV2_DEPTH * CONV2_WIDTH * CONV2_HEIGHT];
fixed_t output[FULL1_NEURONS];


int main(int argc, char *argv[]) {
    
    layerCreateInput(&linput, INPUT_DEPTH, INPUT_WIDTH, INPUT_HEIGHT);
    layerCreateConv(&lconv1, &linput, CONV1_DEPTH, CONV1_WIDTH, CONV1_HEIGHT, CONV1_KERNSIZE, CONV1_PADDING, CONV1_STRIDE);
    layerCreateConv(&lconv2, &lconv1, CONV2_DEPTH, CONV2_WIDTH, CONV2_HEIGHT, CONV2_KERNSIZE, CONV2_PADDING, CONV2_STRIDE);
    layerCreateFull(&lfull1, &lconv2, FULL1_NEURONS);

    lconv1.weights = conv1_weights;
    lconv1.biases = conv1_biases;
    lconv2.weights = conv2_weights;
    lconv2.biases = conv2_biases;
    lfull1.weights = full1_weights;
    lfull1.biases = full1_biases;

    linput.outputs = input;
    lconv1.outputs = inter1;
    lconv2.outputs = inter2;
    lfull1.outputs = output;
    


    FILE *fp = fopen("weights_fixed.bin", "rb");
    if (fp == NULL){
        printf("Error: cannot open weights.bin\n");
        return 1;
    }
    fread(lconv1.weights, sizeof(fixed_t), 16*1*5*5, fp);
    fread(lconv1.biases, sizeof(fixed_t), 16, fp);
    fread(lconv2.weights, sizeof(fixed_t), 32*16*5*5, fp);
    fread(lconv2.biases, sizeof(fixed_t), 32, fp);
    fread(lfull1.weights, sizeof(fixed_t), 15680, fp);
    fread(lfull1.biases, sizeof(fixed_t), 10, fp);

    fclose(fp);

    
    for (int i = 0; i < 28*28; i++){
        input[i] = FIXED_FROM_FLOAT(1);
    }

    //load first image from mnist
    fp = fopen("data/t10k-images-idx3-ubyte", "rb");
    if (fp == NULL){
        printf("Error: cannot open t10k-images-idx3-ubyte\n");
        return 1;
    }
    IdxFile *idx = IdxFile_read(fp);
    fclose(fp);

    for (int i = 0; i < 28*28; i++){
        //random number between 0 and 1
        input[i] = FIXED_FROM_FLOAT(random() / (float)RAND_MAX);
    
    }

    layerSetInputs(&linput, input);

    for (int i = 0; i < 10; i++){
        printf("%f\n", FIXED_TO_FLOAT(output[i]));
    }

    return 0;
}