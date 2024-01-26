#include "fixed.h"
#include <stdio.h>
#include "cnn.h"

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

uint8_t img[28 * 28] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84, 185, 159, 151, 60, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 222, 254, 254, 254, 254, 241, 198, 198, 198, 198, 198, 198, 198, 198, 170, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 114, 72, 114, 163, 227, 254, 225, 254, 254, 254, 250, 229, 254, 254, 140, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 66, 14, 67, 67, 67, 59, 21, 236, 254, 106, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 83, 253, 209, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 233, 255, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 129, 254, 238, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 59, 249, 254, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 133, 254, 187, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 205, 248, 58, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 126, 254, 182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 75, 251, 240, 57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 221, 254, 166, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 203, 254, 219, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 254, 254, 77, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 224, 254, 115, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 133, 254, 254, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61, 242, 254, 254, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 254, 254, 219, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 254, 207, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };

void print_fixed(fixed_t x){
    int ent = x >> FIXED_BITS;
    int frac = 0;
    for (int i = 0; i < FIXED_BITS; i++){
        frac |= ((x >> i) & 1) << (FIXED_BITS - i - 1);
    }
    printf("%d.%d\n", ent, frac);

}

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
#define LOAD_WEIGHTS
#ifdef LOAD_WEIGHTS
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

#else
    fixed_t values[5] = {FIXED_FROM_FLOAT(-0.5f), FIXED_FROM_FLOAT(-0.25f), FIXED_FROM_FLOAT(0), FIXED_FROM_FLOAT(0.25f), FIXED_FROM_FLOAT(0.5f)}
    for (int i = 0; i < 16*1*5*5; i++){
        lconv1.weights[i] = values[i % 5];
    }
    for (int i = 0; i < 16; i++){
        lconv1.biases[i] = values[i % 5];
    }
    for (int i = 0; i < 32*16*5*5; i++){
        lconv2.weights[i] = values[i % 5];
    }
    for (int i = 0; i < 32; i++){
        lconv2.biases[i] = values[i % 5];
    }
    for (int i = 0; i < 15680; i++){
        lfull1.weights[i] = values[i % 5];
    }
    for (int i = 0; i < 10; i++){
        lfull1.biases[i] = values[i % 5];
    }
    
#endif
    
    for (int i = 0; i < 28*28; i++){
        fixed_t im = img[i] << FIXED_BITS;
        im = im >> 8;
        input[i] = im;
    }

    layerSetInputs(&linput, input);
    fixed_t sorted[10];
    int sorted_idx[10];
    for (int i = 0; i < 10; i++){
        sorted[i] = output[i];
    }
    for (int i = 0; i < 10; i++){
        sorted_idx[i] = i;
    }
    for (int i = 0; i < 10; i++){
        for (int j = i; j < 10; j++){
            if (sorted[j] > sorted[i]){
                fixed_t tmp = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = tmp;
                int tmp_idx = sorted_idx[i];
                sorted_idx[i] = sorted_idx[j];
                sorted_idx[j] = tmp_idx;
            }
        }
    }
    printf("\nScores:\n");
    for (int i = 0; i < 10; i++){
        printf("%d: %f\n", sorted_idx[i], FIXED_TO_FLOAT(sorted[i]));
    }
    printf("\n");
    return 0;
}