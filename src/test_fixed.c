#include <stdio.h> 
#include <stdint.h>
#include "fixed.h"


void print_fixed(fixed_t x){
    int ent = x >> FIXED_BITS;
    int frac = 0;
    for (int i = 0; i < FIXED_BITS; i++){
        frac += ((x >> i) & 1) << (FIXED_BITS - i - 1);
    }
    printf("%d.%d\n", ent, frac);

}

int main(int argc, char *argv[]) {
    float a = 3.2f;
    float b = 5.8f;

    fixed_t fa = FIXED_FROM_FLOAT(a);
    fixed_t fb = FIXED_FROM_FLOAT(b);

    printf("fa = %f\n", FIXED_TO_FLOAT(fa));
    printf("fb = %f\n", FIXED_TO_FLOAT(fb));

    fixed_t fc = FIXED_DIV(fa, fb);

    printf("fc = %f\n", FIXED_TO_FLOAT(fc));

    printf("255.0f = %d\n", FIXED_FROM_FLOAT(255.0f));

    fixed_t img = FIXED_DIV(224 << FIXED_BITS, 255 << FIXED_BITS);
    printf("img = %f\n", FIXED_TO_FLOAT(img));
    fixed_t max = 1044480;
    printf("max = %f\n", FIXED_TO_FLOAT((fixed_t) 128 << FIXED_BITS));

    return 0;
}