/*
 * FftRegularizer.h
 *
 *  Created on: Feb 4, 2015
 *      Author: vantonov
 */

#ifndef FFTDEBLURRER_FFTREGULARIZER_H_
#define FFTDEBLURRER_FFTREGULARIZER_H_

#include "FftDeblurrer.h"
#include "SparseBlurKernel.h"

namespace Test {
namespace Deblurring {

class FftRegularizer {
 public:
  static constexpr float HQPM_WEIGHT_MINIMAL = 1.0;
  static constexpr float HQPM_WEIGHT_MAXIMAL = 256.0;
  static constexpr float HQPM_WEIGHT_MULTIPLY_STEP =
      2.82842712475;  // 2*sqrt(2)

  FftRegularizer();

  virtual ~FftRegularizer();

  static size_t getMemorySize(int imageWidth, int imageHeight);

  float getRegularizationWeight() const { return mRegularizationWeight; }

  void setRegularizationWeight(float regularizationWeight) {
    mRegularizationWeight = regularizationWeight;
  }

  point_value_t* getRegularizationFftImage() const {
    return mRegularizationFftImage;
  }

  point_value_t* getRegularizationFftImageX() const {
    return mRegularizationFftImageX;
  }

  point_value_t* getRegularizationFftImageY() const {
    return mRegularizationFftImageY;
  }

  bool isHqpmUsed() const {
    return (mHqpmWeightsFftImage != NULL) ||
           (mHqpmWeightsFftImageX != NULL && mHqpmWeightsFftImageY != NULL);
  }

  virtual void calculateHqpmWeights(const point_value_t* inputFftImageData,
                                    float beta) = 0;

  virtual void calculateHqpmWeightsX(const point_value_t* inputFftImageData,
                                     float beta) = 0;

  virtual void calculateHqpmWeightsY(const point_value_t* inputFftImageData,
                                     float beta) = 0;

  float getMaximalHqpmWeight() const { return mMaximalHqpmWeight; }

  void setMaximalHqpmWeight(float maximalHqpmWeight) {
    mMaximalHqpmWeight = maximalHqpmWeight;
  }

  float getMinimalHqpmWeight() const { return mMinimalHqpmWeight; }

  void setMinimalHqpmWeight(float minimalHqpmWeight) {
    mMinimalHqpmWeight = minimalHqpmWeight;
  }

  float getMultiplyStepHqpmWeight() const { return mMultiplyStepHqpmWeight; }

  void setMultiplyStepHqpmWeight(float multiplyStepHqpmWeight) {
    mMultiplyStepHqpmWeight = multiplyStepHqpmWeight;
  }

  point_value_t* getHqpmWeightsFftImage() const { return mHqpmWeightsFftImage; }

  point_value_t* getHqpmWeightsFftImageX() const {
    return mHqpmWeightsFftImageX;
  }

  point_value_t* getHqpmWeightsFftImageY() const {
    return mHqpmWeightsFftImageY;
  }

  void addFftHqpmImage(const point_value_t* inputFftImageData,
                       const point_value_t* kernelFftImageData,
                       const point_value_t* regularizerFftImageData,
                       const point_value_t* weightFftImageData,
                       float regularizationWeight,
                       point_value_t* outputFftImageData);

  void addFftHqpmImage(const point_value_t* inputFftImageData,
                       const point_value_t* kernelFftImageData,
                       const point_value_t* firstRegularizerFftImageData,
                       const point_value_t* secondRegularizerFftImageData,
                       const point_value_t* firstWeightFftImageData,
                       const point_value_t* secondWeightFftImageData,
                       float regularizationWeight,
                       point_value_t* outputFftImageData);

 protected:
  FftRegularizer(int imageWidth, int imageHeight);

  FftRegularizer(int imageWidth, int imageHeight, void* pExternalMemory);

  virtual void setRegularization() = 0;

  virtual void setRegularizationX() = 0;

  virtual void setRegularizationY() = 0;

  virtual void calculateHqpmWeight(const point_value_t* inputImageData,
                                   float beta,
                                   point_value_t* outputImageData) = 0;

  bool calculateFft2D(point_value_t* c, int nx, int ny, int dir);

  void calculateIdentityFft2D(point_value_t* fftImage);

  void calculateLaplaceFft2D(point_value_t* fftImage);

  void calculateGradientXFft2D(point_value_t* fftImage);

  void calculateGradientYFft2D(point_value_t* fftImage);

  bool isExternalMemoryUsed;
  int mFftImageWidth;
  int mFftImageHeight;

  float mRegularizationWeight;

  point_value_t* mRegularizationFftImage;
  point_value_t* mRegularizationFftImageX;
  point_value_t* mRegularizationFftImageY;

  float mMinimalHqpmWeight;
  float mMaximalHqpmWeight;
  float mMultiplyStepHqpmWeight;

  point_value_t* mHqpmWeightsFftImage;
  point_value_t* mHqpmWeightsFftImageX;
  point_value_t* mHqpmWeightsFftImageY;

 private:
  void initalize(int imageWidth, int imageHeight, void* pExternalMemory);

  void deinitalize();

  point_value_t* mBufferReal;
  point_value_t* mBufferImaginary;
};

}  // namespace Deblurring
}  // namespace Test

#endif /* FFTDEBLURRER_FFTREGULARIZER_H_ */
