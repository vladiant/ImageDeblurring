#include "blur_kernel.h"

#include <stdlib.h>
#include <string.h>

#include "local_debug.h"

BlurKernelResult initBlurKernel(PSF* pInputKernel, uint16_t kernelWidth,
                                uint16_t kernelHeight) {
  uint32_t size = kernelWidth * kernelHeight;

  if (size > 0) {
    pInputKernel->width = kernelWidth;
    pInputKernel->height = kernelHeight;
    pInputKernel->buffer =
        (float*)malloc(kernelWidth * kernelHeight * sizeof(float));

    if (pInputKernel->buffer == nullptr) {
      LOGE("initBlurKernel: Unable to allocate %lu bytes!\n",
           size * sizeof(float));
      return BLUR_KERNEL_MEM_ALLOC_FAILED;  // Unable to allocate memory
    }

  } else {
    LOGE("initBlurKernel: invalid size - %u !\n", size);
    return BLUR_KERNEL_INVALID_SIZE;  // Invalid size
  }

  return BLUR_KERNEL_OK;
}

void releaseBlurKernel(PSF* pInputKernel) {
  if (pInputKernel->buffer != nullptr) {
    free(pInputKernel->buffer);
    pInputKernel->buffer = nullptr;
  }

  pInputKernel->width = 0;
  pInputKernel->height = 0;
}

BlurKernelResult copyBlurKernel(PSF* pDstKernel, PSF* pSrcKernel) {
  if (pSrcKernel->width != pDstKernel->width) {
    LOGE("copyBlurKernel: kernels have different widths !\n");
    return BLUR_KERNEL_INEQUAL_WIDTHS;
  }

  if (pSrcKernel->height != pDstKernel->height) {
    LOGE("copyBlurKernel: kernels have different heights !\n");
    return BLUR_KERNEL_INEQUAL_HEIGHTS;
  }

  if (pSrcKernel->buffer == nullptr) {
    LOGE("copyBlurKernel: source kernel empty !\n");
    return BLUR_KERNEL_EMPTY_SOURCE;
  }

  if (pDstKernel->buffer == nullptr) {
    LOGE("copyBlurKernel: destination kernel empty !\n");
    return BLUR_KERNEL_EMPTY_DESTINATION;
  }

  uint32_t size = pSrcKernel->width * pSrcKernel->height;

  memcpy(pDstKernel->buffer, pSrcKernel->buffer, size * sizeof(float));

  return BLUR_KERNEL_OK;
}

BlurKernelResult cloneBlurKernel(PSF* pDstKernel, PSF* pSrcKernel) {
  if (pSrcKernel->buffer == nullptr) {
    LOGE("cloneBlurKernel: source kernel empty !\n");
    return BLUR_KERNEL_EMPTY_SOURCE;
  }

  initBlurKernel(pDstKernel, pSrcKernel->width, pSrcKernel->height);

  BlurKernelResult returnValue =
      initBlurKernel(pDstKernel, pSrcKernel->width, pSrcKernel->height);

  if (returnValue == BLUR_KERNEL_OK) {
    uint32_t size = pSrcKernel->width * pSrcKernel->height;
    memcpy(pDstKernel->buffer, pSrcKernel->buffer, size * sizeof(float));

  } else {
    return returnValue;
  }

  return BLUR_KERNEL_OK;
}

void clearBlurKernel(PSF* pInputKernel) {
  if ((pInputKernel->width != 0) && (pInputKernel->height != 0) &&
      (pInputKernel->buffer != nullptr)) {
    float* pData = pInputKernel->buffer;
    int row, col;

    for (row = 0; row < pInputKernel->height; ++row) {
      for (col = 0; col < pInputKernel->width; ++col) {
        *pData = 0;
        pData++;
      }
    }
  }
}

void regularizeBlurKernel(PSF* pInputKernel) {
  if ((pInputKernel->width != 0) && (pInputKernel->height != 0) &&
      (pInputKernel->buffer != nullptr)) {
    float* pData = pInputKernel->buffer;
    int row, col;
    for (row = 0; row < pInputKernel->height; ++row) {
      for (col = 0; col < pInputKernel->width; ++col) {
        *pData = (*pData < 0) ? 0 : *pData;
        pData++;
      }
    }
  }
}

BlurKernelResult normalizeBlurKernel(PSF* pInputKernel) {
  BlurKernelResult retVal;

  if ((pInputKernel->width != 0) && (pInputKernel->height != 0) &&
      (pInputKernel->buffer != nullptr)) {
    float kernelSum = 0;
    int row, col;
    float* pData = pInputKernel->buffer;

    for (row = 0; row < pInputKernel->height; ++row) {
      for (col = 0; col < pInputKernel->width; ++col) {
        kernelSum += *pData++;
      }
    }

    if (kernelSum > 0) {
      pData = pInputKernel->buffer;
      for (row = 0; row < pInputKernel->height; ++row) {
        for (col = 0; col < pInputKernel->width; ++col) {
          *pData /= kernelSum;
          pData++;
        }
      }

      retVal = BLUR_KERNEL_OK;

    } else {
      retVal = BLUR_KERNEL_NON_POSITIVE_KERNEL_SUM;
    }
  }

  return retVal;
}