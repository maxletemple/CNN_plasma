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

int main(int argc, char *argv[]) {
    
    Layer *linput = layerCreateInput(1, 28, 28);
    Layer *lconv1 = layerCreateConv(linput, 16, 14, 14, 5, 2, 2);
    Layer *lconv2 = layerCreateConv(lconv1, 32, 7, 7, 5, 2, 2);
    Layer *lfull1 = layerCreateFull(lconv2, 10);

    //load weights

    FILE *fp = fopen("weights_fixed.bin", "rb");
    if (fp == NULL){
        printf("Error: cannot open weights.bin\n");
        return 1;
    }
    fread(lconv1->weights, sizeof(fixed_t), 16*1*5*5, fp);
    fread(lconv1->biases, sizeof(fixed_t), 16, fp);
    fread(lconv2->weights, sizeof(fixed_t), 32*16*5*5, fp);
    fread(lconv2->biases, sizeof(fixed_t), 32, fp);
    fread(lfull1->weights, sizeof(fixed_t), 15680, fp);
    fread(lfull1->biases, sizeof(fixed_t), 10, fp);

    fclose(fp);

    fixed_t *input = (fixed_t*)malloc(sizeof(fixed_t)*28*28);
    
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
        input[i] = FIXED_FROM_FLOAT((float)idx->data[i]/255.0f);
    }


    fixed_t *output = (fixed_t*)malloc(sizeof(fixed_t)*10);

    layerSetInputs(linput, input);
    layerGetOutputs(lfull1, output);


    for (int i = 0; i < 10; i++){
        printf("%f\n", FIXED_TO_FLOAT(output[i]));
    }

    return 0;
}