#include "cnn.h"
#include <stdio.h>
#include <stdlib.h>

void LayerCreate(Layer* self, LayerType type, Layer* lprev, int depth, int width, int height, int nbiases, int nweights){

    self->type = type;
    self->lprev = lprev;
    self->lnext = NULL;
    self->layerid = 0;
    if (lprev != NULL){
        self->layerid = lprev->layerid + 1;
        lprev->lnext = self;
    }

    self->depth = depth;
    self->width = width;
    self->height = height;

    self->nbiases = nbiases;

    self->nweights = nweights;

    self->noutputs = depth*width*height;

}

void layerCreateInput(Layer* self, int depth, int width, int height){
    LayerCreate(self, LAYER_INPUT, NULL, depth, width, height, 0, 0);
}

void layerCreateFull(Layer* self, Layer *lprev, int nneurons){
    LayerCreate(self, LAYER_FULL, lprev, nneurons, 1, 1, nneurons, lprev->noutputs * nneurons);
}

void layerCreateConv(Layer* self, Layer* lprev, int depth, int width, int height, int kernsize, int padding, int stride){
    LayerCreate(self, LAYER_CONV, lprev, depth, width, height, depth, depth * lprev->depth * kernsize * kernsize);
    self->conv.kernsize = kernsize;
    self->conv.padding = padding;
    self->conv.stride = stride;
}

void layerDestroy(Layer *l){
    free(l->biases);
    free(l->weights);
    free(l->outputs);
    free(l);
}

void layerSetInputs(Layer *l, fixed_t *inputs){
    if (l->type == LAYER_INPUT){
        for (int i = 0; i < l->noutputs; i++){
            l->outputs[i] = inputs[i];
        }

        //start feedforward
        while (l->lnext != NULL){
            l = l->lnext;
            
            layerFeedForward(l);
        }        
    }
    else{
        printf("Error: layerSetInputs called on non-input layer\n");
    }
    
}

void layerGetOutputs(Layer *l, fixed_t *outputs){
    for (int i = 0; i < l->noutputs; i++){
        outputs[i] = l->outputs[i];
    }
}

void reLU(fixed_t *x){
    if (*x < 0){
        *x = 0;
    }
}

void layerFeedForwConv(Layer* self){
    Layer* lprev = self->lprev;
    int kernsize = self->conv.kernsize;
    int i = 0;
    for (int z1 = 0; z1 < self->depth; z1++){
        //for each output channel

        int qbase = z1 * lprev->depth * kernsize * kernsize;
        for (int y1 = 0; y1 < self->height; y1++){
            //for each output row

            int y0 = self->conv.stride * y1 - self->conv.padding;
            for (int x1 = 0; x1 < self->width; x1 ++){
                //for each output pixel in the row

                int x0 = self->conv.stride * x1 - self->conv.padding;
                //compute the (x1,y1,z1) output element

                //apply the convolutional kernel
                fixed_t sum = self->biases[z1];
                for (int z0 = 0; z0 < lprev->depth; z0++){
                    //for each input channel

                    int pbase = z0 * lprev->width * lprev->height;
                    for (int dy = 0; dy < kernsize; dy++){
                        //for each 

                        int y = y0 + dy;
                        if (y >= 0 && y < lprev->width){
                            int p = pbase + y * lprev->width;
                            int q = qbase + dy * kernsize;
                            for (int dx = 0; dx < kernsize; dx++){
                                int x = x0 + dx;
                                if (x >= 0 && x < lprev->width){
                                    sum += FIXED_MUL(self->weights[q + dx], lprev->outputs[p + x]);
                                }
                            }
                        }
                    }
                }
                reLU(&sum);
                self->outputs[i++] = sum;
            }
        }
    }
}

void layerFeedForwFull(Layer* l){
    int k = 0;
    for (int i = 0; i < l->noutputs; i++){
        fixed_t sum = l->biases[i];
        for (int j = 0; j < l->lprev->noutputs; j++){
            sum += FIXED_MUL(l->weights[k++], l->lprev->outputs[j]);
        }
        l->outputs[i] = sum;
    }
    

    if (l->lnext != NULL){
    } else {
        //relu
        for (int i = 0; i < l->noutputs; i++){
            reLU(&l->outputs[i]);
        }
    }
}


void layerFeedForward(Layer *l){
    if (l->type == LAYER_INPUT){
        return;
    }
    else if (l->type == LAYER_FULL){
        
        layerFeedForwFull(l);
    }
    else if (l->type == LAYER_CONV){
        layerFeedForwConv(l);
        
    }
    else{
        printf("Error: layerFeedForward called on unknown layer type\n");
    }
}

void layerSetWeights(Layer *l, fixed_t *weights){
    for (int i = 0; i < l->nweights; i++){
        l->weights[i] = weights[i];
    }
}

void layerSetBiases(Layer *l, fixed_t *biases){
    for (int i = 0; i < l->nbiases; i++){
        l->biases[i] = biases[i];
    }
}

