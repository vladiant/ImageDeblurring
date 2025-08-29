/*
 * vanCittertTmDeblurrer.h
 *
 *  Created on: Jan 24, 2015
 *      Author: vantonov
 */

#ifndef ITERATIVEDEBLURRER_VANCITTERTTMDEBLURRER_H_
#define ITERATIVEDEBLURRER_VANCITTERTTMDEBLURRER_H_

#include "IterativeDeblurrer.h"

namespace Test {
namespace Deblurring {

class vanCittertTmDeblurrer : public IterativeDeblurrer {
 public:
  vanCittertTmDeblurrer(int imageWidth, int imageHeight);

  vanCittertTmDeblurrer(int imageWidth, int imageHeight, void* pExternalMemory);

  virtual ~vanCittertTmDeblurrer();

  static size_t getMemorySize(int imageWidth, int imageHeight);

  void operator()(const uint8_t* inputImageData, int inputImagePpln,
                  const SparseBlurKernel& currentBlurKernel,
                  uint8_t* outputImageData, int outputImagePpln);

  void operator()(const uint8_t* inputImageData, int inputImagePpln,
                  const point_value_t* kernelData, int kernelWidth,
                  int kernelHeight, int kernelPpln, uint8_t* outputImageData,
                  int outputImagePpln);

  float getBeta() const { return mBeta; }

  void setBeta(float beta) { mBeta = beta; }

  float getRegularizationWeight() const { return mRegularizationWeight; }

  void setRegularizationWeight(float regularizationWeight) {
    mRegularizationWeight = regularizationWeight;
  }

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

  void calculateRegularization(const point_value_t* inputImageData,
                               point_value_t* outputImageData,
                               bool transposeKernel);

  void calculateRegularizationX(const point_value_t* inputImageData,
                                point_value_t* outputImageData,
                                bool transposeKernel);

  void calculateRegularizationY(const point_value_t* inputImageData,
                                point_value_t* outputImageData,
                                bool transposeKernel);

 private:
  // Disable assignment & copy
  vanCittertTmDeblurrer(const vanCittertTmDeblurrer& other);
  vanCittertTmDeblurrer& operator=(const vanCittertTmDeblurrer& other);

  point_value_t* mCurrentDeblurredImage;
  point_value_t* mResidualImage;
  point_value_t* mReblurredImage;
  point_value_t* mBlurredImage;

  float mBeta;

  point_value_t* mRegularizationImage;
  point_value_t* mRegularizationImageTransposed;

  float mRegularizationWeight;
};

}  // namespace Deblurring
}  // namespace Test

#endif /* ITERATIVEDEBLURRER_VANCITTERTTMDEBLURRER_H_ */
