/*
 * CgDeblurrer.h
 *
 *  Created on: Jan 8, 2015
 *      Author: vantonov
 */

#ifndef CGDEBLURRER_CGADEBLURRER_H_
#define CGDEBLURRER_CGADEBLURRER_H_

#include "IterativeDeblurrer.h"

namespace Test {
namespace Deblurring {

class CgDeblurrer : public IterativeDeblurrer {
 public:
  CgDeblurrer(int imageWidth, int imageHeight);

  CgDeblurrer(int imageWidth, int imageHeight, void* pExternalMemory);

  virtual ~CgDeblurrer();

  static size_t getMemorySize(int imageWidth, int imageHeight);

  void operator()(const uint8_t* inputImageData, int inputImagePpln,
                  const SparseBlurKernel& currentBlurKernel,
                  uint8_t* outputImageData, int outputImagePpln);

  void operator()(const uint8_t* inputImageData, int inputImagePpln,
                  const point_value_t* kernelData, int kernelWidth,
                  int kernelHeight, int kernelPpln, uint8_t* outputImageData,
                  int outputImagePpln);

  virtual void operator()(const uint8_t* inputImageData, int inputImagePpln,
                          const GyroBlurKernelBuilder& currentkernelBuilder,
                          uint8_t* outputImageData, int outputImagePpln);

 protected:
  void init(int imageWidth, int imageHeight, void* pExternalMemory);

  void deinit();

  virtual void doIterations(const point_value_t* kernelData, int kernelWidth,
                            int kernelHeight, int kernelPpln);

  virtual void doIterations(const SparseBlurKernel& currentBlurKernel);

  virtual void doIterations(const GyroBlurKernelBuilder& currentkernelBuilder);

  virtual void prepareIterations(const point_value_t* kernelData,
                                 int kernelWidth, int kernelHeight,
                                 int kernelPpln);

  virtual void prepareIterations(const SparseBlurKernel& currentBlurKernel);

  virtual void prepareIterations(
      const GyroBlurKernelBuilder& currentkernelBuilder);

  virtual void doRegularization();

 private:
  // Disable assignment & copy
  CgDeblurrer(const CgDeblurrer& other);
  CgDeblurrer& operator=(const CgDeblurrer& other);

  point_value_t* mCurrentDeblurredImage;
  point_value_t* mResidualImage;
  point_value_t* mPreconditionedImage;
  point_value_t* mBlurredPreconditionedImage;
  point_value_t* mDifferenceResidualImage;
};

}  // namespace Deblurring
}  // namespace Test

#endif /* CGDEBLURRER_CGADEBLURRER_H_ */
