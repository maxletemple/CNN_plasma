#include <stdio.h> 
#include <stdint.h>
#include "fixed.h"


int main(int argc, char *argv[]) {
    float a = 3.2f;
    float b = -5.8f;

    fixed_t fa = FIXED_FROM_FLOAT(a);
    fixed_t fb = FIXED_FROM_FLOAT(b);

    printf("fa = %f\n", FIXED_TO_FLOAT(fa));
    printf("fb = %f\n", FIXED_TO_FLOAT(fb));

    fixed_t fc = FIXED_MUL(fa, fb);
    
    printf("fc = %f\n", FIXED_TO_FLOAT(fc));

    return 0;
}

