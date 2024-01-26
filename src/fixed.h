#ifndef FIXED_H
#define FIXED_H

#include <stdint.h>

typedef int32_t fixed_t;

#define FIXED_BITS 12
#define FIXED_SCALE (1 << FIXED_BITS)
#define FIXED_MASK (FIXED_SCALE - 1)

#define FIXED_INT(x) ((x) >> FIXED_BITS)
#define FIXED_FRAC(x) ((x) & FIXED_MASK)

#define FIXED_FROM_INT(x) ((x) << FIXED_BITS)

#define FIXED_FROM_FLOAT(x) ((fixed_t)((x) * FIXED_SCALE))
#define FIXED_TO_FLOAT(x) ((float)(x) / FIXED_SCALE)

#define FIXED_MUL(x, y) ((((int32_t)(x)) * (int32_t)(y)) >> FIXED_BITS)
#define FIXED_DIV(x, y) ((((int32_t)(x)) << FIXED_BITS) / (int32_t)(y))

#endif // FIXED_H