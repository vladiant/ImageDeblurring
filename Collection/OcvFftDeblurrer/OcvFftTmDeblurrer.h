/*
 * OcvFftTmDeblurrer.h
 *
 *  Created on: Jan 28, 2015
 *      Author: vantonov
 */

#ifndef OCVFFTDEBLURRER_OCVFFTTMDEBLURRER_H_
#define OCVFFTDEBLURRER_OCVFFTTMDEBLURRER_H_

#include <opencv2/core/core.hpp>

#include "ImageDeblurrer.h"

namespace Test {
namespace Deblurring {

class OcvFftTmDeblurrer : public ImageDeblurrer {
 public:
  OcvFftTmDeblurrer(int imageWidth, int imageHeight);

  OcvFftTmDeblurrer(int imageWidth, int imageHeight, void* pExternalMemory);

  ~OcvFftTmDeblurrer();

  static size_t getMemorySize(int imageWidth, int imageHeight, int kernelWidth,
                              int kernelHeight);

  void operator()(const uint8_t* inputImageData, int inputImagePpln,
                  const SparseBlurKernel& currentBlurKernel,
                  uint8_t* outputImageData, int outputImagePpln);

  void operator()(const uint8_t* inputImageData, int inputImagePpln,
                  const point_value_t* kernelData, int kernelWidth,
                  int kernelHeight, int kernelPpln, uint8_t* outputImageData,
                  int outputImagePpln);

  float getRegularizationWeight() const { return mRegularizationWeight; }

  void setRegularizationWeight(float regularizationWeight) {
    mRegularizationWeight = regularizationWeight;
  }

  static constexpr float epsilon = 1e-4;

 protected:
  void init(int imageWidth, int imageHeight, void* pExternalMemory);

  void deinit();

  void buildKernelForFft(const point_value_t* kernelData, int kernelWidth,
                         int kernelHeight, int kernelPpln,
                         point_value_t* kernelFftData);

  void invertFftMatrix(cv::Mat& matrixFft, float minimalValue);

  void invertFftMatrix(cv::Mat& matrixFft);

  void regularizedInvertFftMatrix(cv::Mat& matrixFft,
                                  const cv::Mat& matrixRegularization,
                                  float regularizationWeight);

  void regularizedInvertFftMatrix(cv::Mat& matrixFft,
                                  const cv::Mat& firstMatrixRegularization,
                                  const cv::Mat& secondMatrixRegularization,
                                  float regularizationWeight);

  void setIdentityRegularization();

  void setLaplaceRegularization();

  void setGradientXRegularization();

  void setGradientYRegularization();

 private:
  point_value_t* mInputImageData;
  point_value_t* mKernelImageData;
  point_value_t* mOutputImageData;

  point_value_t* mIdentityRegularizationData;
  point_value_t* mLaplacianRegularizationData;
  point_value_t* mGradientXRegularizationData;
  point_value_t* mGradientYRegularizationData;

  float mRegularizationWeight;
};

}  // namespace Deblurring
}  // namespace Test

#endif /* OCVFFTDEBLURRER_OCVFFTTMDEBLURRER_H_ */
