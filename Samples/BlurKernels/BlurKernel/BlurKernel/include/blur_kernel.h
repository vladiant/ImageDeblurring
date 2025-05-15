#pragma once

#include "PSF.hpp"

typedef enum {

  BLUR_KERNEL_OK,
  BLUR_KERNEL_EMPTY,
  BLUR_KERNEL_MEM_ALLOC_FAILED,
  BLUR_KERNEL_INVALID_SIZE,
  BLUR_KERNEL_INEQUAL_WIDTHS,
  BLUR_KERNEL_INEQUAL_HEIGHTS,
  BLUR_KERNEL_EMPTY_SOURCE,
  BLUR_KERNEL_EMPTY_DESTINATION,
  BLUR_KERNEL_NON_POSITIVE_KERNEL_SUM

} BlurKernelResult;

#define BLUR_KERNEL_TOL 1e-4f

BlurKernelResult initBlurKernel(PSF* pInputKernel, uint16_t kernelWidth,
                                uint16_t kernelHeight);
void releaseBlurKernel(PSF* pInputKernel);
BlurKernelResult copyBlurKernel(PSF* pDstKernel, PSF* pSrcKernel);
BlurKernelResult cloneBlurKernel(PSF* pDstKernel, PSF* pSrcKernel);

void clearBlurKernel(PSF* pInputKernel);
void regularizeBlurKernel(PSF* pInputKernel);
BlurKernelResult normalizeBlurKernel(PSF* pInputKernel);
