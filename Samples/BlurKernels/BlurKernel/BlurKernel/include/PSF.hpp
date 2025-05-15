#pragma once

#include <stdint.h>

#ifdef __cplusplus
struct PSF {
#else
typedef struct {
#endif

  uint16_t width;
  uint16_t height;
  float* buffer;

#ifdef __cplusplus
  PSF() : width(0), height(0), buffer(0) {}
};
#else
} PSF;
#endif
