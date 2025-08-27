/*
 * LrDeblurrer.h
 *
 *  Created on: Jan 23, 2015
 *      Author: vantonov
 */

#ifndef ITERATIVEDEBLURRER_LRDEBLURRER_H_
#define ITERATIVEDEBLURRER_LRDEBLURRER_H_

#include "IterativeDeblurrer.h"

namespace Test {
namespace Deblurring {

class LrDeblurrer : public IterativeDeblurrer {
 public:
  LrDeblurrer(int imageWidth, int imageHeight);

  LrDeblurrer(int imageWidth, int imageHeight, void* pExternalMemory);

  virtual ~LrDeblurrer();

  static size_t getMemorySize(int imageWidth, int imageHeight);

  void operator()(const uint8_t* inputImageData, int inputImagePpln,
                  const SparseBlurKernel& currentBlurKernel,
                  uint8_t* outputImageData, int outputImagePpln);

  void operator()(const uint8_t* inputImageData, int inputImagePpln,
                  const point_value_t* kernelData, int kernelWidth,
                  int kernelHeight, int kernelPpln, uint8_t* outputImageData,
                  int outputImagePpln);

 protected:
  void init(int imageWidth, int imageHeight, void* pExternalMemory);

  void deinit();

  virtual void doIterations(const point_value_t* kernelData, int kernelWidth,
                            int kernelHeight, int kernelPpln);

  virtual void doIterations(const SparseBlurKernel& currentBlurKernel);

  virtual void prepareIterations(const point_value_t* kernelData,
                                 int kernelWidth, int kernelHeight,
                                 int kernelPpln);

  virtual void prepareIterations(const SparseBlurKernel& currentBlurKernel);

  virtual void doRegularization();

 private:
  // Disable assignment & copy
  LrDeblurrer(const LrDeblurrer& other);
  LrDeblurrer& operator=(const LrDeblurrer& other);

  point_value_t* mCurrentDeblurredImage;
  point_value_t* mReblurredImage;
  point_value_t* mBlurredImage;
  point_value_t* mWeightImage;
};

}  // namespace Deblurring
}  // namespace Test

#endif /* ITERATIVEDEBLURRER_LRDEBLURRER_H_ */
