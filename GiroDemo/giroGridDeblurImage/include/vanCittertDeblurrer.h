/*
 * vanCittertDeblurrer.h
 *
 *  Created on: Jan 23, 2015
 *      Author: vantonov
 */

#ifndef ITERATIVEDEBLURRER_VANCITTERTDEBLURRER_H_
#define ITERATIVEDEBLURRER_VANCITTERTDEBLURRER_H_

#include "IterativeDeblurrer.h"

namespace Test {
namespace Deblurring {

class vanCittertDeblurrer : public IterativeDeblurrer {
 public:
  vanCittertDeblurrer(int imageWidth, int imageHeight);

  vanCittertDeblurrer(int imageWidth, int imageHeight, void* pExternalMemory);

  virtual ~vanCittertDeblurrer();

  static size_t getMemorySize(int imageWidth, int imageHeight);

  virtual void operator()(const uint8_t* inputImageData, int inputImagePpln,
                          const SparseBlurKernel& currentBlurKernel,
                          uint8_t* outputImageData, int outputImagePpln);

  virtual void operator()(const uint8_t* inputImageData, int inputImagePpln,
                          const point_value_t* kernelData, int kernelWidth,
                          int kernelHeight, int kernelPpln,
                          uint8_t* outputImageData, int outputImagePpln);

  virtual void operator()(const uint8_t* inputImageData, int inputImagePpln,
                          const GyroBlurKernelBuilder& currentkernelBuilder,
                          uint8_t* outputImageData, int outputImagePpln);

  virtual void operator()(const uint8_t* inputImageData, int inputImagePpln,
                          const SparseBlurKernel* blurKernels,
                          int blurKernelsCount, uint8_t* outputImageData,
                          int outputImagePpln);

  float getBeta() const { return mBeta; }

  void setBeta(float beta) { mBeta = beta; }

 protected:
  void init(int imageWidth, int imageHeight, void* pExternalMemory);

  void deinit();

  virtual void doIterations(const point_value_t* kernelData, int kernelWidth,
                            int kernelHeight, int kernelPpln);

  virtual void doIterations(const SparseBlurKernel& currentBlurKernel);

  virtual void doIterations(const GyroBlurKernelBuilder& currentkernelBuilder);

  virtual void doIterations(const SparseBlurKernel* blurKernels,
                            int blurKernelsCount);

  virtual void prepareIterations(const point_value_t* kernelData,
                                 int kernelWidth, int kernelHeight,
                                 int kernelPpln);

  virtual void prepareIterations(const SparseBlurKernel& currentBlurKernel);

  virtual void prepareIterations(
      const GyroBlurKernelBuilder& currentkernelBuilder);

  virtual void prepareIterations(const SparseBlurKernel* blurKernels,
                                 int blurKernelsCount);

  virtual void doRegularization();

 private:
  // Disable assignment & copy
  vanCittertDeblurrer(const vanCittertDeblurrer& other);
  vanCittertDeblurrer& operator=(const vanCittertDeblurrer& other);

  point_value_t* mCurrentDeblurredImage;
  point_value_t* mResidualImage;
  point_value_t* mReblurredImage;
  point_value_t* mBlurredImage;

  float mBeta;
};

}  // namespace Deblurring
}  // namespace Test

#endif /* ITERATIVEDEBLURRER_VANCITTERTDEBLURRER_H_ */
