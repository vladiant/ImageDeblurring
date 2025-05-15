/*
 * File:   main.c
 * Author: vantonov
 *
 * Created on March 7, 2013, 3:54 PM
 */

#include <blur_kernel.h>
#include <stdio.h>
#include <stdlib.h>

/*
 *
 */

void printBlurKernel(PSF* inputPSF) {
  if ((inputPSF->width != 0) && (inputPSF->height != 0) &&
      (inputPSF->buffer != nullptr)) {
    float* pData = inputPSF->buffer;
    int row, col;

    for (row = 0; row < inputPSF->height; ++row) {
      for (col = 0; col < inputPSF->width; ++col) {
        printf("%f\t", *pData);
        pData++;
      }

      printf("\n");
    }

    printf("\n");

  } else {
    printf("PSF empty!\n\n");
  }
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
  int row, col;
  float* pData;

  PSF test1 = {};

  printf("test1 not initialized\n");
  printf("width:  %d\n", test1.width);
  printf("height: %d\n", test1.height);
  printf("buffer: %p\n\n", (void*)test1.buffer);

  PSF test2;
  initBlurKernel(&test2, 10, 10);

  // Set test2
  pData = test2.buffer;
  for (row = 0; row < test2.height; ++row) {
    for (col = 0; col < test2.width; ++col) {
      *pData = row + col;
      pData++;
    }
  }

  printf("test2 initialized and set\n");
  printf("width:  %d\n", test2.width);
  printf("height: %d\n", test2.height);
  printf("buffer: %p\n\n", (void*)test2.buffer);
  printBlurKernel(&test2);

  PSF test3;
  cloneBlurKernel(&test3, &test2);

  printf("test3 cloned from test2 \n");
  printf("width:  %d\n", test3.width);
  printf("height: %d\n", test3.height);
  printf("buffer: %p\n\n", (void*)test3.buffer);
  printBlurKernel(&test3);

  releaseBlurKernel(&test2);

  printf("test2 cleared\n");
  printf("width:  %d\n", test2.width);
  printf("height: %d\n", test2.height);
  printf("buffer: %p\n\n", (void*)test2.buffer);
  printBlurKernel(&test2);

  cloneBlurKernel(&test2, &test3);

  printf("test2 cloned from test3 \n");
  printf("width:  %d\n", test2.width);
  printf("height: %d\n", test2.height);
  printf("buffer: %p\n\n", (void*)test2.buffer);
  printBlurKernel(&test2);

  // Set test2
  pData = test2.buffer;
  for (row = 0; row < test2.height; ++row) {
    for (col = 0; col < test2.width; ++col) {
      *pData = row - col;
      pData++;
    }
  }

  copyBlurKernel(&test3, &test2);
  //    regularizeBlurKernel(&test3);
  //    normalizeBlurKernel(&test3);

  printf("test3 copied from test2 \n");
  printf("width:  %d\n", test3.width);
  printf("height: %d\n", test3.height);
  printf("buffer: %p\n\n", (void*)test3.buffer);
  printBlurKernel(&test3);

  regularizeBlurKernel(&test3);

  printf("test3 regularized \n");
  printf("width:  %d\n", test3.width);
  printf("height: %d\n", test3.height);
  printf("buffer: %p\n\n", (void*)test3.buffer);
  printBlurKernel(&test3);

  clearBlurKernel(&test3);

  printf("test3 cleared \n");
  printf("width:  %d\n", test3.width);
  printf("height: %d\n", test3.height);
  printf("buffer: %p\n\n", (void*)test3.buffer);
  printBlurKernel(&test3);

  return (EXIT_SUCCESS);
}
