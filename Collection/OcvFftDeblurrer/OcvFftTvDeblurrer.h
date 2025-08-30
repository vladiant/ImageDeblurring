/*
 * OcvFftTvDeblurrer.h
 *
 *  Created on: Jan 29, 2015
 *      Author: vantonov
 */

#ifndef OCVFFTDEBLURRER_OCVFFTTVDEBLURRER_H_
#define OCVFFTDEBLURRER_OCVFFTTVDEBLURRER_H_

#include <opencv2/core/core.hpp>

#include "ImageDeblurrer.h"

namespace Test {
namespace Deblurring {

class OcvFftTvDeblurrer : public ImageDeblurrer {
 public:
  OcvFftTvDeblurrer(int imageWidth, int imageHeight);

  OcvFftTvDeblurrer(int imageWidth, int imageHeight, void* pExternalMemory);

  ~OcvFftTvDeblurrer();

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

  void calculateFftMatrixForHqpm(const cv::Mat& kernel,
                                 const cv::Mat& firstMatrixRegularization,
                                 const cv::Mat& secondMatrixRegularization,
                                 const cv::Mat& firstWeight,
                                 const cv::Mat& secondWeight,
                                 float lambdaOverBeta, cv::Mat& output);

  void initializeHqpmWeights();

  float WeightCalcAlphaTwoThirds(float x, float beta);

  float WeightCalcTV(float x, float beta);

 private:
  point_value_t* mInputImageData;
  point_value_t* mKernelImageData;
  point_value_t* mOutputImageData;

  point_value_t* mIdentityRegularizationData;
  point_value_t* mLaplacianRegularizationData;
  point_value_t* mGradientXRegularizationData;
  point_value_t* mGradientYRegularizationData;

  point_value_t* mGradientXWeightData;
  point_value_t* mGradientYWeightData;

  float mRegularizationWeight;
};

}  // namespace Deblurring
}  // namespace Test

#endif /* OCVFFTDEBLURRER_OCVFFTTVDEBLURRER_H_ */
