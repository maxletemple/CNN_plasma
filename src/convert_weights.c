#include <stdio.h>
#include "fixed.h"

int main(int argc, char *argv[]) {
    FILE *fin = fopen("weights_float.bin", "rb");
    if (fin == NULL) {
        printf("Error opening file\n");
        return 1;
    }
    FILE *fout = fopen("weights_fixed.bin", "wb");
    if (fout == NULL) {
        printf("Error opening file\n");
        return 1;
    }

    float f;
    while (fread(&f, sizeof(float), 1, fin) == 1) {
        fixed_t x = FIXED_FROM_FLOAT(f);
        fwrite(&x, sizeof(fixed_t), 1, fout);
    }

    fclose(fin);
    fclose(fout);
}