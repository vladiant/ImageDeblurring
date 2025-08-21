/*
 * FftDeblurrer.cpp
 *
 *  Created on: Feb 2, 2015
 *      Author: vantonov
 */

#include "FftDeblurrer.h"

#include <math.h>
#include <string.h>

#include <stdexcept>

namespace Test {
namespace Deblurring {

FftDeblurrer::FftDeblurrer(int imageWidth, int imageHeight)
    : ImageDeblurrer(imageWidth, imageHeight),
      mFftImageWidth(0),
      mFftImageHeight(0),
      mBufferReal(NULL),
      mBufferImaginary(NULL) {
  initalize(imageWidth, imageHeight, NULL);
}

FftDeblurrer::FftDeblurrer(int imageWidth, int imageHeight,
                           void* pExternalMemory)
    : ImageDeblurrer(imageWidth, imageHeight, pExternalMemory),
      mFftImageWidth(0),
      mFftImageHeight(0),
      mBufferReal(NULL),
      mBufferImaginary(NULL) {
  initalize(imageWidth, imageHeight,
            (void*)((intptr_t)pExternalMemory +
                    ImageDeblurrer::getMemorySize(imageWidth, imageHeight)));
}

FftDeblurrer::~FftDeblurrer() { deinitalize(); }

int FftDeblurrer::calculateOptimalFftSize(int size) {
  return (1 << (int)ceil(log(size) / log(2.0)));
}

size_t FftDeblurrer::getMemorySize(int imageWidth, int imageHeight) {
  imageWidth =
      calculateOptimalFftSize((1.0 + 2.0 * BORDERS_PADDING) * imageWidth);
  imageHeight =
      calculateOptimalFftSize((1.0 + 2.0 * BORDERS_PADDING) * imageHeight);

  int bufferSize = imageWidth > imageHeight ? imageWidth : imageHeight;

  int requiredMemorySize = 2 * bufferSize * sizeof(point_value_t);

  requiredMemorySize += ImageDeblurrer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void FftDeblurrer::initalize(int imageWidth, int imageHeight,
                             void* pExternalMemory) {
  mFftImageWidth =
      calculateOptimalFftSize((1.0 + 2.0 * BORDERS_PADDING) * imageWidth);
  mFftImageHeight =
      calculateOptimalFftSize((1.0 + 2.0 * BORDERS_PADDING) * imageHeight);

  int bufferSize =
      mFftImageWidth > mFftImageHeight ? mFftImageWidth : mFftImageHeight;

  if (isExternalMemoryUsed) {
    point_value_t* pDataBuffer = (point_value_t*)pExternalMemory;
    mBufferReal = &pDataBuffer[0];
    mBufferImaginary = &pDataBuffer[bufferSize];
  } else {
    mBufferReal = new point_value_t[bufferSize];
    mBufferImaginary = new point_value_t[bufferSize];
  }
}

void FftDeblurrer::deinitalize() {
  if (!isExternalMemoryUsed) {
    delete[] mBufferReal;
    delete[] mBufferImaginary;
  }

  mFftImageWidth = 0;
  mFftImageHeight = 0;
  mBufferReal = NULL;
  mBufferImaginary = NULL;
}

bool FftDeblurrer::calculateFft2D(point_value_t* c, int nx, int ny, int dir) {
  return calculateFft2D(c, nx, ny, dir, mBufferReal, mBufferImaginary);
}

bool FftDeblurrer::calculateFft2D(point_value_t* c, int nx, int ny, int dir,
                                  point_value_t* realBuffer,
                                  point_value_t* imaginaryBuffer) {
  int i, j;
  int m, twopm;

  // Transform the rows
  if (!calcPowerOfTwo(nx, &m, &twopm) || twopm != nx) return false;
  for (j = 0; j < ny; j++) {
    for (i = 0; i < nx; i++) {
      realBuffer[i] = c[2 * (i + j * nx) + 0];
      imaginaryBuffer[i] = c[2 * (i + j * nx) + 1];
    }
    calculateFft(dir, m, realBuffer, imaginaryBuffer);
    for (i = 0; i < nx; i++) {
      c[2 * (i + j * nx) + 0] = realBuffer[i];
      c[2 * (i + j * nx) + 1] = imaginaryBuffer[i];
    }
  }

  // Transform the columns
  if (!calcPowerOfTwo(ny, &m, &twopm) || twopm != ny) return false;
  for (i = 0; i < nx; i++) {
    for (j = 0; j < ny; j++) {
      realBuffer[j] = c[2 * (i + j * nx) + 0];
      imaginaryBuffer[j] = c[2 * (i + j * nx) + 1];
    }
    calculateFft(dir, m, realBuffer, imaginaryBuffer);
    for (j = 0; j < ny; j++) {
      c[2 * (i + j * nx) + 0] = realBuffer[j];
      c[2 * (i + j * nx) + 1] = imaginaryBuffer[j];
    }
  }

  return true;
}

bool FftDeblurrer::calculateFft(int dir, int m, float* x, float* y) {
  long nn, i, i1, j, k, i2, l, l1, l2;
  float c1, c2, tx, ty, t1, t2, u1, u2, z;

  // Calculate the number of points
  nn = 1;
  for (i = 0; i < m; i++) nn *= 2;

  // Do the bit reversal
  i2 = nn >> 1;
  j = 0;
  for (i = 0; i < nn - 1; i++) {
    if (i < j) {
      tx = x[i];
      ty = y[i];
      x[i] = x[j];
      y[i] = y[j];
      x[j] = tx;
      y[j] = ty;
    }
    k = i2;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }

  // Compute the FFT
  c1 = -1.0;
  c2 = 0.0;
  l2 = 1;
  for (l = 0; l < m; l++) {
    l1 = l2;
    l2 <<= 1;
    u1 = 1.0;
    u2 = 0.0;
    for (j = 0; j < l1; j++) {
      for (i = j; i < nn; i += l2) {
        i1 = i + l1;
        t1 = u1 * x[i1] - u2 * y[i1];
        t2 = u1 * y[i1] + u2 * x[i1];
        x[i1] = x[i] - t1;
        y[i1] = y[i] - t2;
        x[i] += t1;
        y[i] += t2;
      }
      z = u1 * c1 - u2 * c2;
      u2 = u1 * c2 + u2 * c1;
      u1 = z;
    }
    c2 = sqrt((1.0 - c1) / 2.0);
    if (dir == FFT_FORWARD) c2 = -c2;
    c1 = sqrt((1.0 + c1) / 2.0);
  }

  // Scaling for inverse transform
  if (dir == FFT_INVERSE) {
    for (i = 0; i < nn; i++) {
      x[i] /= (float)nn;
      y[i] /= (float)nn;
    }
  }

  return true;
}

bool FftDeblurrer::calcPowerOfTwo(int n, int* m, int* twopm) {
  if (n <= 1) {
    *m = 0;
    *twopm = 1;
    return false;
  }

  *m = 1;
  *twopm = 2;
  do {
    (*m)++;
    (*twopm) *= 2;
  } while (2 * (*twopm) <= n);

  if (*twopm != n)
    return false;
  else
    return true;
}

void FftDeblurrer::convertFromInputToFft(const uint8_t* inputImageData,
                                         int imageWidth, int imageHeight,
                                         int inputImagePpln,
                                         point_value_t* inputBuffer,
                                         int inputBufferStride,
                                         int inputBufferHeight) {
  memset(inputBuffer, 0,
         inputBufferStride * inputBufferHeight * sizeof(point_value_t));

  int offsetX = round(imageWidth * BORDERS_PADDING);
  int offsetY = round(imageWidth * BORDERS_PADDING);

  uint8_t* pInputData = (uint8_t*)inputImageData;
  point_value_t* pOutputData =
      &inputBuffer[2 * offsetX + offsetY * inputBufferStride];

  for (int row = 0; row < imageHeight; row++) {
    for (int col = 0; col < imageWidth; col++) {
      point_value_t value = *pInputData++;
      value /= 255.0;
      *pOutputData++ = value;
      *pOutputData++ = 0;
    }
    pInputData += inputImagePpln - imageWidth;
    pOutputData += (inputBufferStride - 2 * imageWidth);
  }

  return;
}

void FftDeblurrer::convertFftToOutput(const point_value_t* outputBuffer,
                                      int imageWidth, int imageHeight,
                                      int outputBufferStride,
                                      uint8_t* outputImageData,
                                      int outputImagePpln) {
  int offsetX = round(imageWidth * BORDERS_PADDING);
  int offsetY = round(imageWidth * BORDERS_PADDING);

  point_value_t* pInputData =
      (point_value_t*)&outputBuffer[2 * offsetX + offsetY * outputBufferStride];
  uint8_t* pOutputData = outputImageData;

  for (int row = 0; row < imageHeight; row++) {
    for (int col = 0; col < imageWidth; col++) {
      point_value_t value = *pInputData++;
      pInputData++;

      value = round(value * 255.0);
      if (value < 0) {
        value = 0;
      } else if (value > 255) {
        value = 255;
      }
      *pOutputData++ = value;
    }
    pOutputData += outputImagePpln - imageWidth;
    pInputData += (outputBufferStride - 2 * imageWidth);
  }

  return;
}

void FftDeblurrer::prepareBordersOfFftImage(point_value_t* fftImageData,
                                            int imageWidth, int imageHeight,
                                            int fftImageStride) {
  int offsetX = round(imageWidth * BORDERS_PADDING);
  int offsetY = round(imageWidth * BORDERS_PADDING);

  point_value_t* pImageData = NULL;
  point_value_t borderValueRe = 0;
  point_value_t borderValueIm = 0;

  // Set top left corner
  borderValueRe = fftImageData[2 * offsetX + offsetY * fftImageStride + 0];
  borderValueIm = fftImageData[2 * offsetX + offsetY * fftImageStride + 1];
  pImageData = fftImageData;
  for (int row = 0; row < offsetY; row++) {
    for (int col = 0; col < offsetX; col++) {
      *pImageData++ = row * col * borderValueRe / (offsetX * offsetY);
      *pImageData++ = row * col * borderValueIm / (offsetX * offsetY);
    }
    pImageData += (fftImageStride - 2 * offsetX);
  }

  // Set left border
  pImageData = &fftImageData[offsetY * fftImageStride];
  for (int row = offsetY; row < imageHeight + offsetY; row++) {
    borderValueRe = fftImageData[2 * offsetX + row * fftImageStride + 0];
    borderValueIm = fftImageData[2 * offsetX + row * fftImageStride + 1];
    for (int col = 0; col < offsetX; col++) {
      *pImageData++ = col * borderValueRe / offsetX;
      *pImageData++ = col * borderValueIm / offsetX;
    }
    pImageData += (fftImageStride - 2 * offsetX);
  }

  // Set bottom left corner
  borderValueRe =
      fftImageData[2 * offsetX + (offsetY + imageHeight - 1) * fftImageStride +
                   0];
  borderValueIm =
      fftImageData[2 * offsetX + (offsetY + imageHeight - 1) * fftImageStride +
                   1];
  pImageData = &fftImageData[(imageHeight + offsetY) * fftImageStride];
  for (int row = imageHeight + offsetY; row < imageHeight + 2 * offsetY;
       row++) {
    for (int col = 0; col < offsetX; col++) {
      *pImageData++ = (imageHeight + 2 * offsetY - row - 1) * col *
                      borderValueRe / (offsetX * offsetY);
      *pImageData++ = (imageHeight + 2 * offsetY - row - 1) * col *
                      borderValueIm / (offsetX * offsetY);
    }
    pImageData += (fftImageStride - 2 * offsetX);
  }

  // Set top border
  for (int col = offsetX; col < offsetX + imageWidth; col++) {
    pImageData = &fftImageData[2 * col];
    borderValueRe = fftImageData[2 * col + offsetY * fftImageStride + 0];
    borderValueIm = fftImageData[2 * col + offsetY * fftImageStride + 1];

    for (int row = 0; row < offsetY; row++) {
      *pImageData++ = row * borderValueRe / offsetY;
      *pImageData++ = row * borderValueIm / offsetY;
      pImageData += (fftImageStride - 2);
    }
  }

  // Set bottom border
  for (int col = offsetX; col < offsetX + imageWidth; col++) {
    pImageData =
        &fftImageData[2 * col + (imageHeight + offsetY) * fftImageStride];
    borderValueRe =
        fftImageData[2 * col + (imageHeight + offsetY - 1) * fftImageStride +
                     0];
    borderValueIm =
        fftImageData[2 * col + (imageHeight + offsetY - 1) * fftImageStride +
                     1];

    for (int row = 0; row < offsetY; row++) {
      *pImageData++ = (offsetY - row - 1) * borderValueRe / offsetY;
      *pImageData++ = (offsetY - row - 1) * borderValueIm / offsetY;
      pImageData += (fftImageStride - 2);
    }
  }

  // Set top right corner
  borderValueRe = fftImageData[2 * (offsetX + imageWidth - 1) +
                               offsetY * fftImageStride + 0];
  borderValueIm = fftImageData[2 * (offsetX + imageWidth - 1) +
                               offsetY * fftImageStride + 1];
  pImageData = &fftImageData[2 * (offsetX + imageWidth)];
  for (int row = 0; row < offsetY; row++) {
    for (int col = offsetX + imageWidth; col < 2 * offsetX + imageWidth;
         col++) {
      *pImageData++ = row * (2 * offsetX + imageWidth - col - 1) *
                      borderValueRe / (offsetX * offsetY);
      *pImageData++ = row * (2 * offsetX + imageWidth - col - 1) *
                      borderValueIm / (offsetX * offsetY);
    }
    pImageData += (fftImageStride - 2 * offsetX);
  }

  // Set bottom right corner
  borderValueRe =
      fftImageData[2 * (offsetX + imageWidth - 1) +
                   (offsetY + imageHeight - 1) * fftImageStride + 0];
  borderValueIm =
      fftImageData[2 * (offsetX + imageWidth - 1) +
                   (offsetY + imageHeight - 1) * fftImageStride + 1];
  pImageData = &fftImageData[2 * (offsetX + imageWidth) +
                             (offsetY + imageHeight) * fftImageStride];
  for (int row = offsetY + imageHeight; row < 2 * offsetY + imageHeight;
       row++) {
    for (int col = offsetX + imageWidth; col < 2 * offsetX + imageWidth;
         col++) {
      *pImageData++ = (2 * offsetY + imageHeight - row - 1) *
                      (2 * offsetX + imageWidth - col - 1) * borderValueRe /
                      (offsetX * offsetY);
      *pImageData++ = (2 * offsetY + imageHeight - row - 1) *
                      (2 * offsetX + imageWidth - col - 1) * borderValueIm /
                      (offsetX * offsetY);
    }
    pImageData += (fftImageStride - 2 * offsetX);
  }

  // Set right border
  pImageData =
      &fftImageData[2 * (offsetX + imageWidth) + offsetY * fftImageStride];
  for (int row = offsetY; row < imageHeight + offsetY; row++) {
    borderValueRe =
        fftImageData[2 * (offsetX + imageWidth - 1) + row * fftImageStride + 0];
    borderValueIm =
        fftImageData[2 * (offsetX + imageWidth - 1) + row * fftImageStride + 1];
    for (int col = offsetX + imageWidth; col < 2 * offsetX + imageWidth;
         col++) {
      *pImageData++ =
          (2 * offsetX + imageWidth - col - 1) * borderValueRe / offsetX;
      *pImageData++ =
          (2 * offsetX + imageWidth - col - 1) * borderValueIm / offsetX;
    }
    pImageData += (fftImageStride - 2 * offsetX);
  }

  return;
}  // void FftDeblurrer::prepareBordersOfFftImage( ...

void FftDeblurrer::prepareKernelFftImage(const point_value_t* kernelData,
                                         int kernelWidth, int kernelHeight,
                                         int kernelPpln,
                                         point_value_t* kernelFftData,
                                         int fftImageStride,
                                         int fftImageHeight) {
  memset(kernelFftData, 0,
         fftImageStride * fftImageHeight * sizeof(point_value_t));

  int imageWidth = fftImageStride / 2;

  int kernelCenterX = kernelWidth / 2;
  int kernelCenterY = kernelHeight / 2;

  for (int row = 0; row < kernelHeight; row++) {
    for (int col = 0; col < kernelWidth; col++) {
      point_value_t inputKernelValue = kernelData[col + row * kernelPpln];
      if (0 != inputKernelValue) {
        int kernelFftX = col - kernelCenterX;
        if (kernelFftX < 0) {
          kernelFftX += imageWidth;
        } else if (kernelFftX > imageWidth - 1) {
          kernelFftX -= imageWidth;
        }

        int kernelFftY = row - kernelCenterY;
        if (kernelFftY < 0) {
          kernelFftY += fftImageHeight;
        } else if (kernelFftY > fftImageHeight - 1) {
          kernelFftY -= fftImageHeight;
        }

        kernelFftData[2 * kernelFftX + fftImageStride * kernelFftY + 0] =
            inputKernelValue;
        kernelFftData[2 * kernelFftX + fftImageStride * kernelFftY + 1] = 0.0;
      }

    }  // for (int col ...
  }  // for (int row ...

  return;
}  // void FftDeblurrer::prepareKernelFftImage( ...

void FftDeblurrer::prepareKernelFftImage(const SparseBlurKernel& blurKernel,
                                         point_value_t* kernelFftData,
                                         int fftImageStride,
                                         int fftImageHeight) {
  memset(kernelFftData, 0,
         fftImageStride * fftImageHeight * sizeof(point_value_t));

  int imageWidth = fftImageStride / 2;

  float xCoordMin, yCoordMin, xCoordMax, yCoordMax;
  // TODO: Hack, fix!
  const_cast<SparseBlurKernel&>(blurKernel)
      .calcCoordsSpan(&xCoordMin, &yCoordMin, &xCoordMax, &yCoordMax);

  int kernelSpanX = round(xCoordMax - xCoordMin);
  int kernelSpanY = round(yCoordMax - yCoordMin);

  if (kernelSpanX > imageWidth) {
    throw std::invalid_argument("Image width smaller than kernel width!");
  }

  if (kernelSpanY > fftImageHeight) {
    throw std::invalid_argument("Image height smaller than kernel height!");
  }

  std::vector<point_coord_t> kernelPointX;
  std::vector<point_coord_t> kernelPointY;
  std::vector<point_value_t> kernelPointValue;

  // TODO: Hack, fix!
  const_cast<SparseBlurKernel&>(blurKernel)
      .extractKernelPoints(kernelPointX, kernelPointY, kernelPointValue);

  for (int k = 0; k < blurKernel.getKernelSize(); k++) {
    int xCoord = kernelPointX[k];
    int yCoord = kernelPointY[k];

    if (xCoord < 0) {
      xCoord += imageWidth;
    } else if (xCoord > imageWidth - 1) {
      xCoord -= imageWidth;
    }

    if (yCoord < 0) {
      yCoord += fftImageHeight;
    } else if (yCoord > fftImageHeight - 1) {
      yCoord -= fftImageHeight;
    }

    kernelFftData[2 * xCoord + yCoord * fftImageStride + 0] =
        kernelPointValue[k];
    kernelFftData[2 * xCoord + yCoord * fftImageStride + 1] = 0.0;
  }

  return;
}

void FftDeblurrer::multiplyFftImages(const point_value_t* firstImageData,
                                     const point_value_t* secondImageData,
                                     int fftImageWidth, int fftImageHeight,
                                     point_value_t* outputImageData) {
  for (int row = 0; row < fftImageHeight; row++) {
    for (int col = 0; col < fftImageWidth; col++) {
      point_value_t firstRe =
          firstImageData[2 * (col + row * fftImageWidth) + 0];
      point_value_t firstIm =
          firstImageData[2 * (col + row * fftImageWidth) + 1];
      point_value_t secondRe =
          secondImageData[2 * (col + row * fftImageWidth) + 0];
      point_value_t secondIm =
          secondImageData[2 * (col + row * fftImageWidth) + 1];

      outputImageData[2 * (col + row * fftImageWidth) + 0] =
          firstRe * secondRe - firstIm * secondIm;
      outputImageData[2 * (col + row * fftImageWidth) + 1] =
          firstRe * secondIm + firstIm * secondRe;
    }
  }
}

void FftDeblurrer::applyCutOff(point_value_t* inputFftImageData,
                               int fftImageWidth, int fftImageHeight,
                               int cutOffFrequencyX, int cutOffFrequencyY) {
  for (int row = 0; row < fftImageHeight; row++) {
    memset(&inputFftImageData[2 * (cutOffFrequencyX + row * fftImageWidth)], 0,
           4 * (fftImageWidth / 2 - cutOffFrequencyX) * sizeof(point_value_t));
  }

  for (int row = cutOffFrequencyY; row < fftImageHeight - cutOffFrequencyY;
       row++) {
    memset(&inputFftImageData[2 * (0 + row * fftImageWidth)], 0,
           2 * (fftImageWidth) * sizeof(point_value_t));
  }
}

void FftDeblurrer::invertFftMatrix(const point_value_t* inputFftImageData,
                                   int fftImageWidth, int fftImageHeight,
                                   point_value_t* outputFftImageData) {
  for (int row = 0; row < fftImageHeight; row++) {
    for (int col = 0; col < fftImageWidth; col++) {
      point_value_t real =
          inputFftImageData[2 * (col + row * fftImageWidth) + 0];
      point_value_t imaginary =
          inputFftImageData[2 * (col + row * fftImageWidth) + 1];
      point_value_t sum = real * real + imaginary * imaginary;

      if (sum != 0) {
        outputFftImageData[2 * (col + row * fftImageWidth) + 0] = real / sum;
        outputFftImageData[2 * (col + row * fftImageWidth) + 1] =
            -imaginary / sum;
      } else {
        outputFftImageData[2 * (col + row * fftImageWidth) + 0] = 0;
        outputFftImageData[2 * (col + row * fftImageWidth) + 1] = 0;
      }
    }
  }
}

void FftDeblurrer::invertFftMatrix(const point_value_t* inputFftImageData,
                                   int fftImageWidth, int fftImageHeight,
                                   point_value_t minimalValue,
                                   point_value_t* outputFftImageData) {
  for (int row = 0; row < fftImageHeight; row++) {
    for (int col = 0; col < fftImageWidth; col++) {
      point_value_t real =
          inputFftImageData[2 * (col + row * fftImageWidth) + 0];
      point_value_t imaginary =
          inputFftImageData[2 * (col + row * fftImageWidth) + 1];
      point_value_t sum =
          real * real + imaginary * imaginary + minimalValue * minimalValue;

      outputFftImageData[2 * (col + row * fftImageWidth) + 0] = real / sum;
      outputFftImageData[2 * (col + row * fftImageWidth) + 1] =
          -imaginary / sum;
    }
  }
}

void FftDeblurrer::invertFftMatrixRegularized(
    const point_value_t* inputFftImageData,
    const point_value_t* regularizerFftImageData, int fftImageWidth,
    int fftImageHeight, float regularizationWeight,
    point_value_t* outputFftImageData) {
  for (int row = 0; row < fftImageHeight; row++) {
    for (int col = 0; col < fftImageWidth; col++) {
      point_value_t real =
          inputFftImageData[2 * (col + row * fftImageWidth) + 0];
      point_value_t imaginary =
          inputFftImageData[2 * (col + row * fftImageWidth) + 1];

      point_value_t realReg =
          regularizerFftImageData[2 * (col + row * fftImageWidth) + 0];
      point_value_t imaginaryReg =
          regularizerFftImageData[2 * (col + row * fftImageWidth) + 1];

      point_value_t sum = real * real + imaginary * imaginary +
                          regularizationWeight *
                              (realReg * realReg + imaginaryReg * imaginaryReg);

      outputFftImageData[2 * (col + row * fftImageWidth) + 0] = real / sum;
      outputFftImageData[2 * (col + row * fftImageWidth) + 1] =
          -imaginary / sum;
    }
  }

  return;
}

void FftDeblurrer::invertFftMatrixRegularized(
    const point_value_t* inputFftImageData,
    const point_value_t* firstRegularizerFftData,
    const point_value_t* secondRegularizerFftData, int fftImageWidth,
    int fftImageHeight, float regularizationWeight,
    point_value_t* outputFftImageData) {
  for (int row = 0; row < fftImageHeight; row++) {
    for (int col = 0; col < fftImageWidth; col++) {
      point_value_t real =
          inputFftImageData[2 * (col + row * fftImageWidth) + 0];
      point_value_t imaginary =
          inputFftImageData[2 * (col + row * fftImageWidth) + 1];

      point_value_t realFirstReg =
          firstRegularizerFftData[2 * (col + row * fftImageWidth) + 0];
      point_value_t imaginaryFirstReg =
          firstRegularizerFftData[2 * (col + row * fftImageWidth) + 1];

      point_value_t realSecondReg =
          secondRegularizerFftData[2 * (col + row * fftImageWidth) + 0];
      point_value_t imaginarySecondReg =
          secondRegularizerFftData[2 * (col + row * fftImageWidth) + 1];

      point_value_t sum =
          real * real + imaginary * imaginary +
          regularizationWeight * (realFirstReg * realFirstReg +
                                  imaginaryFirstReg * imaginaryFirstReg +
                                  realSecondReg * realSecondReg +
                                  imaginarySecondReg * imaginarySecondReg);

      outputFftImageData[2 * (col + row * fftImageWidth) + 0] = real / sum;
      outputFftImageData[2 * (col + row * fftImageWidth) + 1] =
          -imaginary / sum;
    }
  }

  return;
}  // void FftDeblurrer::invertFftMatrixRegularized( ...

}  // namespace Deblurring
}  // namespace Test
