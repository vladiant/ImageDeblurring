/*
 * CutOffFftDeblurrer.h
 *
 *  Created on: Feb 3, 2015
 *      Author: vantonov
 */

#ifndef FFTDEBLURRER_CUTOFFFFTDEBLURRER_H_
#define FFTDEBLURRER_CUTOFFFFTDEBLURRER_H_

#include "FftDeblurrer.h"

namespace Test {
namespace Deblurring {

class CutOffFftDeblurrer : public FftDeblurrer {
 public:
  CutOffFftDeblurrer(int imageWidth, int imageHeight);

  CutOffFftDeblurrer(int imageWidth, int imageHeight, void* pExternalMemory);

  virtual ~CutOffFftDeblurrer();

  static size_t getMemorySize(int imageWidth, int imageHeight);

  virtual void operator()(const uint8_t* inputImageData, int inputImagePpln,
                          const SparseBlurKernel& currentBlurKernel,
                          uint8_t* outputImageData, int outputImagePpln);

  virtual void operator()(const uint8_t* inputImageData, int inputImagePpln,
                          const point_value_t* kernelData, int kernelWidth,
                          int kernelHeight, int kernelPpln,
                          uint8_t* outputImageData, int outputImagePpln);

  int getCutOffFrequencyX() const { return mCutOffFrequencyX; }

  bool setCutOffFrequencyX(int cutOffFrequencyX);

  int getCutOffFrequencyY() const { return mCutOffFrequencyY; }

  bool setCutOffFrequencyY(int cutOffFrequencyY);

 protected:
  virtual void init(int imageWidth, int imageHeight, void* pExternalMemory);

  virtual void deinit();

 private:
  // Disable assignment & copy
  CutOffFftDeblurrer(const CutOffFftDeblurrer& other);
  CutOffFftDeblurrer& operator=(const CutOffFftDeblurrer& other);

  point_value_t* mWorkImage;
  point_value_t* mKernelImage;
  int mCutOffFrequencyX;
  int mCutOffFrequencyY;
};

}  // namespace Deblurring
}  // namespace Test

#endif /* FFTDEBLURRER_CUTOFFFFTDEBLURRER_H_ */
