#ifndef CNN_H_
#define CNN_H_

#include "fixed.h"

typedef enum _LayerType{
   LAYER_INPUT = 0,
   LAYER_FULL,
   LAYER_CONV,
} LayerType;

typedef struct _Layer{
   LayerType type;
   int layerid;
   struct _Layer* lnext;
   struct _Layer* lprev;
   
   int depth;
   int width;
   int height;

   int nweights;
   fixed_t *weights;
   
   int nbiases;
   fixed_t *biases;

   int noutputs;
   fixed_t *outputs;

   union{
      struct{
      } full;
      struct{
         int kernsize;
         int padding;
         int stride;
      } conv;
   };

} Layer;

Layer* layerCreate(LayerType type, Layer *lprev, int depth, int width, int height, int nbiases, int nweights);
Layer* layerCreateInput(int depth, int width, int height);
Layer* layerCreateFull(Layer *lprev, int nneurons);
Layer* layerCreateConv(Layer* lprev, int depth, int width, int height, int kernsize, int padding, int stride);

void layerDestroy(Layer *l);

void layerSetInputs(Layer *l, fixed_t *inputs);
void layerGetOutputs(Layer *l, fixed_t *outputs);
void layerUpdate(Layer *l);

void layerSetWeights(Layer *l, fixed_t *weights);
void layerSetBiases(Layer *l, fixed_t *biases);

void layerFeedForward(Layer *l);
#endif /* CNN_H_ */