/*
 * LsFftDeblurrer.h
 *
 *  Created on: Feb 3, 2015
 *      Author: vantonov
 */

#ifndef FFTDEBLURRER_LSFFTDEBLURRER_H_
#define FFTDEBLURRER_LSFFTDEBLURRER_H_

#include "FftDeblurrer.h"
#include "FftRegularizer.h"

namespace Test {
namespace Deblurring {

class LsFftDeblurrer : public FftDeblurrer {
 public:
  LsFftDeblurrer(int imageWidth, int imageHeight);

  LsFftDeblurrer(int imageWidth, int imageHeight, void* pExternalMemory);

  virtual ~LsFftDeblurrer();

  static size_t getMemorySize(int imageWidth, int imageHeight);

  virtual void operator()(const uint8_t* inputImageData, int inputImagePpln,
                          const SparseBlurKernel& currentBlurKernel,
                          uint8_t* outputImageData, int outputImagePpln);

  virtual void operator()(const uint8_t* inputImageData, int inputImagePpln,
                          const point_value_t* kernelData, int kernelWidth,
                          int kernelHeight, int kernelPpln,
                          uint8_t* outputImageData, int outputImagePpln);

  void setRegularizer(FftRegularizer* regularizer) {
    mRegularizer = regularizer;
  }

 protected:
  virtual void init(int imageWidth, int imageHeight, void* pExternalMemory);

  virtual void deinit();

  void updateHqpmWeights(float beta);

  void calculateDeblurKernel(float beta);

  void applyHqpmWeights(float beta);

  FftRegularizer* mRegularizer;

 private:
  // Disable assignment & copy
  LsFftDeblurrer(const LsFftDeblurrer& other);
  LsFftDeblurrer& operator=(const LsFftDeblurrer& other);

  point_value_t* mWorkImage;
  point_value_t* mKernelImage;
  point_value_t* mInvertedKernelImage;
  point_value_t* mDeblurredImage;
};

}  // namespace Deblurring
}  // namespace Test

#endif /* FFTDEBLURRER_LSFFTDEBLURRER_H_ */
