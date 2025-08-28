/*
 * FftDeblurrer.h
 *
 *  Created on: Feb 2, 2015
 *      Author: vantonov
 */

#include "ImageDeblurrer.h"

#ifndef FFTDEBLURRER_FFTDEBLURRER_H_
#define FFTDEBLURRER_FFTDEBLURRER_H_

namespace Test {
namespace Deblurring {

class FftDeblurrer : public ImageDeblurrer {
 public:
  virtual ~FftDeblurrer();

  static constexpr float BORDERS_PADDING = 0.25;

  enum { FFT_FORWARD = 1, FFT_INVERSE = -1 };

  static size_t getMemorySize(int imageWidth, int imageHeight);

  static int calculateOptimalFftSize(int size);

  static bool calculateFft(int dir, int m, float* x, float* y);

  static bool calculateFft2D(point_value_t* c, int nx, int ny, int dir,
                             point_value_t* realBuffer,
                             point_value_t* imaginaryBuffer);

  static void multiplyFftImages(const point_value_t* firstImageData,
                                const point_value_t* secondImageData,
                                int fftImageWidth, int fftImageHeight,
                                point_value_t* outputImageData);

 protected:
  FftDeblurrer(int imageWidth, int imageHeight);

  FftDeblurrer(int imageWidth, int imageHeight, void* pExternalMemory);

  bool calculateFft2D(point_value_t* c, int nx, int ny, int dir);

  static void convertFromInputToFft(const uint8_t* inputImageData,
                                    int imageWidth, int imageHeight,
                                    int inputImagePpln,
                                    point_value_t* inputBuffer,
                                    int inputBufferStride,
                                    int inputBufferHeight);

  static void convertFftToOutput(const point_value_t* outputBuffer,
                                 int imageWidth, int imageHeight,
                                 int outputBufferStride,
                                 uint8_t* outputImageData, int outputImagePpln);

  static void prepareBordersOfFftImage(point_value_t* fftImageData,
                                       int imageWidth, int imageHeight,
                                       int fftImageWidth);

  static void prepareKernelFftImage(const point_value_t* kernelData,
                                    int kernelWidth, int kernelHeight,
                                    int kernelPpln,
                                    point_value_t* kernelFftData,
                                    int fftImageStride, int fftImageHeight);

  static void prepareKernelFftImage(const SparseBlurKernel& blurKernel,
                                    point_value_t* kernelFftData,
                                    int fftImageStride, int fftImageHeight);

  static void applyCutOff(point_value_t* inputFftImageData, int fftImageWidth,
                          int fftImageHeight, int cutOffFrequencyX,
                          int cutOffFrequencyY);

  static void invertFftMatrix(const point_value_t* inputFftImageData,
                              int fftImageWidth, int fftImageHeight,
                              point_value_t* outputFftImageData);

  void invertFftMatrix(const point_value_t* inputFftImageData,
                       int fftImageWidth, int fftImageHeight,
                       point_value_t minimalValue,
                       point_value_t* outputFftImageData);

  static void invertFftMatrixRegularized(
      const point_value_t* inputFftImageData,
      const point_value_t* regularizerFftImageData, int fftImageWidth,
      int fftImageHeight, float regularizationWeight,
      point_value_t* outputFftImageData);

  static void invertFftMatrixRegularized(
      const point_value_t* inputFftImageData,
      const point_value_t* firstRegularizerFftData,
      const point_value_t* secondRegularizerFftData, int fftImageWidth,
      int fftImageHeight, float regularizationWeight,
      point_value_t* outputFftImageData);

  int mFftImageWidth;
  int mFftImageHeight;

 private:
  // Disable assignment & copy
  FftDeblurrer(const FftDeblurrer& other);
  FftDeblurrer& operator=(const FftDeblurrer& other);

  void initalize(int imageWidth, int imageHeight, void* pExternalMemory);

  void deinitalize();

  static bool calcPowerOfTwo(int n, int* m, int* twopm);

  point_value_t* mBufferReal;
  point_value_t* mBufferImaginary;
};

}  // namespace Deblurring
}  // namespace Test

#endif /* FFTDEBLURRER_FFTDEBLURRER_H_ */
