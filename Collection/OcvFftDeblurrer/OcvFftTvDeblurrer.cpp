/*
 * OcvFftTvDeblurrer.cpp
 *
 *  Created on: Jan 29, 2015
 *      Author: vantonov
 */

#include "OcvFftTvDeblurrer.h"

#include <algorithm>

#include "BlurKernelUtils.h"

// TODO: Debug print
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace Test {
namespace Deblurring {

OcvFftTvDeblurrer::OcvFftTvDeblurrer(int imageWidth, int imageHeight)
    : ImageDeblurrer(imageWidth, imageHeight),
      mInputImageData(NULL),
      mKernelImageData(NULL),
      mOutputImageData(NULL),
      mIdentityRegularizationData(NULL),
      mLaplacianRegularizationData(NULL),
      mGradientXRegularizationData(NULL),
      mGradientYRegularizationData(NULL),
      mGradientXWeightData(NULL),
      mGradientYWeightData(NULL),
      mRegularizationWeight(0) {
  init(imageWidth, imageHeight, NULL);
}

OcvFftTvDeblurrer::OcvFftTvDeblurrer(int imageWidth, int imageHeight,
                                     void* pExternalMemory)
    : ImageDeblurrer(imageWidth, imageHeight, pExternalMemory),
      mInputImageData(NULL),
      mKernelImageData(NULL),
      mOutputImageData(NULL),
      mIdentityRegularizationData(NULL),
      mLaplacianRegularizationData(NULL),
      mGradientXRegularizationData(NULL),
      mGradientYRegularizationData(NULL),
      mGradientXWeightData(NULL),
      mGradientYWeightData(NULL),
      mRegularizationWeight(0) {
  init(imageWidth, imageHeight, pExternalMemory);
}

OcvFftTvDeblurrer::~OcvFftTvDeblurrer() { deinit(); }

void OcvFftTvDeblurrer::init(int imageWidth, int imageHeight,
                             void* pExternalMemory) {
  size_t imageBlockSize = imageWidth * imageHeight;

  if (isExternalMemoryUsed) {
    point_value_t* pDataBuffer = (point_value_t*)pExternalMemory;
    mInputImageData = &pDataBuffer[0];
    mKernelImageData = &pDataBuffer[imageBlockSize];
    mOutputImageData = &pDataBuffer[2 * imageBlockSize];
    mIdentityRegularizationData = &pDataBuffer[3 * imageBlockSize];
    mLaplacianRegularizationData = &pDataBuffer[4 * imageBlockSize];
    mGradientXRegularizationData = &pDataBuffer[5 * imageBlockSize];
    mGradientYRegularizationData = &pDataBuffer[6 * imageBlockSize];
    mGradientXWeightData = &pDataBuffer[7 * imageBlockSize];
    mGradientYWeightData = &pDataBuffer[8 * imageBlockSize];
  } else {
    mInputImageData = new point_value_t[imageBlockSize];
    mKernelImageData = new point_value_t[imageBlockSize];
    mOutputImageData = new point_value_t[imageBlockSize];
    mIdentityRegularizationData = new point_value_t[imageBlockSize];
    mLaplacianRegularizationData = new point_value_t[imageBlockSize];
    mGradientXRegularizationData = new point_value_t[imageBlockSize];
    mGradientYRegularizationData = new point_value_t[imageBlockSize];
    mGradientXWeightData = new point_value_t[imageBlockSize];
    mGradientYWeightData = new point_value_t[imageBlockSize];
  }

  setIdentityRegularization();

  setLaplaceRegularization();

  setGradientXRegularization();

  setGradientYRegularization();
}

void OcvFftTvDeblurrer::deinit() {
  if (!isExternalMemoryUsed) {
    delete[] mInputImageData;
    delete[] mKernelImageData;
    delete[] mOutputImageData;

    delete[] mIdentityRegularizationData;
    delete[] mLaplacianRegularizationData;
    delete[] mGradientXRegularizationData;
    delete[] mGradientYRegularizationData;

    delete[] mGradientXWeightData;
    delete[] mGradientYWeightData;
  }

  mInputImageData = NULL;
  mKernelImageData = NULL;
  mOutputImageData = NULL;

  mIdentityRegularizationData = NULL;
  mLaplacianRegularizationData = NULL;
  mGradientXRegularizationData = NULL;
  mGradientYRegularizationData = NULL;

  mGradientXWeightData = NULL;
  mGradientYWeightData = NULL;

  mRegularizationWeight = 0;
}

void OcvFftTvDeblurrer::operator()(const uint8_t* inputImageData,
                                   int inputImagePpln,
                                   const SparseBlurKernel& currentBlurKernel,
                                   uint8_t* outputImageData,
                                   int outputImagePpln) {
  convertFromInput(inputImageData, mImageWidth, mImageHeight, inputImagePpln,
                   mInputImageData);

  extractSparseKernelToImage(const_cast<SparseBlurKernel&>(currentBlurKernel),
                             mKernelImageData, mImageWidth, mImageHeight,
                             mImageWidth);

  cv::Mat inputImage(mImageHeight, mImageWidth, CV_32FC1, mInputImageData);
  cv::Mat kernelImage(mImageHeight, mImageWidth, CV_32FC1, mKernelImageData);
  cv::Mat outputImage(mImageHeight, mImageWidth, CV_32FC1, mOutputImageData);

  // TODO: Regularization matrices
  cv::Mat regularizationMatrixIdentity(mImageHeight, mImageWidth, CV_32FC1,
                                       mIdentityRegularizationData);
  cv::Mat regularizationMatrixLaplacian(mImageHeight, mImageWidth, CV_32FC1,
                                        mLaplacianRegularizationData);
  cv::Mat regularizationMatrixGradientX(mImageHeight, mImageWidth, CV_32FC1,
                                        mGradientXRegularizationData);
  cv::Mat regularizationMatrixGradientY(mImageHeight, mImageWidth, CV_32FC1,
                                        mGradientYRegularizationData);

  initializeHqpmWeights();

  cv::dft(inputImage, inputImage);
  cv::dft(kernelImage, kernelImage);

  //	invertFftMatrix(kernelImage);

  //	invertFftMatrix(kernelImage, epsilon);

  //	regularizedInvertFftMatrix(kernelImage, regularizationMatrixLaplacian,
  //			mRegularizationWeight);

  //	regularizedInvertFftMatrix(kernelImage, regularizationMatrixGradientX,
  //			regularizationMatrixGradientY, mRegularizationWeight);

  cv::Mat weightsGradientX(mImageHeight, mImageWidth, CV_32FC1,
                           mGradientXWeightData);
  cv::Mat weightsGradientY(mImageHeight, mImageWidth, CV_32FC1,
                           mGradientYWeightData);

  cv::Mat initialKernelImage(mImageHeight, mImageWidth, CV_32FC1);
  kernelImage.copyTo(initialKernelImage);

  float beta = 1.0;

  // TODO: Caclulate HQPM weights
  cv::mulSpectrums(inputImage, regularizationMatrixGradientX, weightsGradientX,
                   cv::DFT_COMPLEX_OUTPUT);
  cv::dft(weightsGradientX, weightsGradientX,
          cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
  cv::mulSpectrums(inputImage, regularizationMatrixGradientY, weightsGradientY,
                   cv::DFT_COMPLEX_OUTPUT);
  cv::dft(weightsGradientY, weightsGradientY,
          cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

  point_value_t* pWeightsGradientXData = mGradientXWeightData;
  point_value_t* pWeightsGradientYData = mGradientYWeightData;
  for (int row = 0; row < mImageHeight; row++) {
    for (int col = 0; col < mImageWidth; col++) {
      float valueX = pWeightsGradientXData[col + row * mImageWidth];
      float valueY = pWeightsGradientYData[col + row * mImageWidth];
      valueX = WeightCalcAlphaTwoThirds(valueX, beta);
      valueY = WeightCalcAlphaTwoThirds(valueY, beta);
      pWeightsGradientXData[col + row * mImageWidth] = valueX;
      pWeightsGradientYData[col + row * mImageWidth] = valueY;
    }
  }

  while (beta < 256.0) {
    // TODO: fix!!!
    kernelImage.copyTo(initialKernelImage);

    // TODO: Hqpm step
    cv::dft(weightsGradientX, weightsGradientX);
    cv::dft(weightsGradientY, weightsGradientY);
    cv::Mat offsetHqpmMatrix(mImageHeight, mImageWidth, CV_32FC1);
    offsetHqpmMatrix = cv::Scalar(0);
    calculateFftMatrixForHqpm(kernelImage, regularizationMatrixGradientX,
                              regularizationMatrixGradientY, weightsGradientX,
                              weightsGradientY, (beta * mRegularizationWeight),
                              offsetHqpmMatrix);

    regularizedInvertFftMatrix(kernelImage, regularizationMatrixGradientX,
                               regularizationMatrixGradientY,
                               mRegularizationWeight * beta);

    cv::addWeighted(kernelImage, 1.0, offsetHqpmMatrix,
                    1.0 / (beta * mRegularizationWeight), 0.0, kernelImage);

    cv::mulSpectrums(inputImage, kernelImage, outputImage,
                     cv::DFT_COMPLEX_OUTPUT);

    // TODO: Caclulate HQPM weights
    cv::mulSpectrums(outputImage, regularizationMatrixGradientX,
                     weightsGradientX, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(weightsGradientX, weightsGradientX,
            cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    cv::mulSpectrums(outputImage, regularizationMatrixGradientY,
                     weightsGradientY, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(weightsGradientY, weightsGradientY,
            cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

    point_value_t* pWeightsGradientXData = mGradientXWeightData;
    point_value_t* pWeightsGradientYData = mGradientYWeightData;
    for (int row = 0; row < mImageHeight; row++) {
      for (int col = 0; col < mImageWidth; col++) {
        float valueX = pWeightsGradientXData[col + row * mImageWidth];
        float valueY = pWeightsGradientYData[col + row * mImageWidth];
        valueX = WeightCalcAlphaTwoThirds(valueX, beta);
        valueY = WeightCalcAlphaTwoThirds(valueY, beta);
        pWeightsGradientXData[col + row * mImageWidth] = valueX;
        pWeightsGradientYData[col + row * mImageWidth] = valueY;
      }
    }

    // TODO: Debug prints
    //		cv::dft(outputImage, outputImage,
    //				cv::DFT_INVERSE | cv::DFT_SCALE |
    // cv::DFT_REAL_OUTPUT); 		cv::imshow("outputImage", outputImage);
    cv::imshow("weightsGradientX", weightsGradientX);
    cv::imshow("weightsGradientY", weightsGradientY);
    cv::waitKey(0);
    std::cout << "beta: " << beta << std::endl;

    // TODO: fix!!!
    initialKernelImage.copyTo(kernelImage);

    beta *= 2 * sqrt(2);
  }

  cv::dft(outputImage, outputImage,
          cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

  convertToOutput(mOutputImageData, mImageWidth, mImageHeight, outputImagePpln,
                  outputImageData);
}

void OcvFftTvDeblurrer::operator()(const uint8_t* inputImageData,
                                   int inputImagePpln,
                                   const point_value_t* kernelData,
                                   int kernelWidth, int kernelHeight,
                                   int kernelPpln, uint8_t* outputImageData,
                                   int outputImagePpln) {
  convertFromInput(inputImageData, mImageWidth, mImageHeight, inputImagePpln,
                   mInputImageData);

  buildKernelForFft(kernelData, kernelWidth, kernelHeight, kernelPpln,
                    mKernelImageData);

  cv::Mat inputImage(mImageHeight, mImageWidth, CV_32FC1, mInputImageData);
  cv::Mat kernelImage(mImageHeight, mImageWidth, CV_32FC1, mKernelImageData);
  cv::Mat outputImage(mImageHeight, mImageWidth, CV_32FC1, mOutputImageData);

  // TODO: Regularization matrices
  cv::Mat regularizationMatrixIdentity(mImageHeight, mImageWidth, CV_32FC1,
                                       mIdentityRegularizationData);
  cv::Mat regularizationMatrixLaplacian(mImageHeight, mImageWidth, CV_32FC1,
                                        mLaplacianRegularizationData);
  cv::Mat regularizationMatrixGradientX(mImageHeight, mImageWidth, CV_32FC1,
                                        mGradientXRegularizationData);
  cv::Mat regularizationMatrixGradientY(mImageHeight, mImageWidth, CV_32FC1,
                                        mGradientYRegularizationData);

  initializeHqpmWeights();

  cv::dft(kernelImage, kernelImage);
  cv::dft(inputImage, inputImage);

  //	invertFftMatrix(kernelImage);

  //	invertFftMatrix(kernelImage, epsilon);

  //	regularizedInvertFftMatrix(kernelImage, regularizationMatrixLaplacian,
  //			mRegularizationWeight);

  //	regularizedInvertFftMatrix(kernelImage, regularizationMatrixGradientX,
  //			regularizationMatrixGradientY, mRegularizationWeight);

  cv::Mat weightsGradientX(mImageHeight, mImageWidth, CV_32FC1,
                           mGradientXWeightData);
  cv::Mat weightsGradientY(mImageHeight, mImageWidth, CV_32FC1,
                           mGradientYWeightData);

  cv::Mat initialKernelImage(mImageHeight, mImageWidth, CV_32FC1);
  kernelImage.copyTo(initialKernelImage);

  float beta = 1.0;

  // TODO: Caclulate HQPM weights
  cv::mulSpectrums(inputImage, regularizationMatrixGradientX, weightsGradientX,
                   cv::DFT_COMPLEX_OUTPUT);
  cv::dft(weightsGradientX, weightsGradientX,
          cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
  cv::mulSpectrums(inputImage, regularizationMatrixGradientY, weightsGradientY,
                   cv::DFT_COMPLEX_OUTPUT);
  cv::dft(weightsGradientY, weightsGradientY,
          cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

  point_value_t* pWeightsGradientXData = mGradientXWeightData;
  point_value_t* pWeightsGradientYData = mGradientYWeightData;
  for (int row = 0; row < mImageHeight; row++) {
    for (int col = 0; col < mImageWidth; col++) {
      float valueX = pWeightsGradientXData[col + row * mImageWidth];
      float valueY = pWeightsGradientYData[col + row * mImageWidth];
      valueX = WeightCalcAlphaTwoThirds(valueX, beta);
      valueY = WeightCalcAlphaTwoThirds(valueY, beta);
      pWeightsGradientXData[col + row * mImageWidth] = valueX;
      pWeightsGradientYData[col + row * mImageWidth] = valueY;
    }
  }

  while (beta < 256.0) {
    // TODO: fix!!!
    kernelImage.copyTo(initialKernelImage);

    // TODO: Hqpm step
    cv::dft(weightsGradientX, weightsGradientX);
    cv::dft(weightsGradientY, weightsGradientY);
    cv::Mat offsetHqpmMatrix(mImageHeight, mImageWidth, CV_32FC1);
    offsetHqpmMatrix = cv::Scalar(0);
    calculateFftMatrixForHqpm(kernelImage, regularizationMatrixGradientX,
                              regularizationMatrixGradientY, weightsGradientX,
                              weightsGradientY, (beta * mRegularizationWeight),
                              offsetHqpmMatrix);

    regularizedInvertFftMatrix(kernelImage, regularizationMatrixGradientX,
                               regularizationMatrixGradientY,
                               mRegularizationWeight * beta);

    cv::addWeighted(kernelImage, 1.0, offsetHqpmMatrix,
                    1.0 / (beta * mRegularizationWeight), 0.0, kernelImage);

    cv::mulSpectrums(inputImage, kernelImage, outputImage,
                     cv::DFT_COMPLEX_OUTPUT);

    // TODO: Caclulate HQPM weights
    cv::mulSpectrums(outputImage, regularizationMatrixGradientX,
                     weightsGradientX, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(weightsGradientX, weightsGradientX,
            cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    cv::mulSpectrums(outputImage, regularizationMatrixGradientY,
                     weightsGradientY, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(weightsGradientY, weightsGradientY,
            cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

    point_value_t* pWeightsGradientXData = mGradientXWeightData;
    point_value_t* pWeightsGradientYData = mGradientYWeightData;
    for (int row = 0; row < mImageHeight; row++) {
      for (int col = 0; col < mImageWidth; col++) {
        float valueX = pWeightsGradientXData[col + row * mImageWidth];
        float valueY = pWeightsGradientYData[col + row * mImageWidth];
        valueX = WeightCalcAlphaTwoThirds(valueX, beta);
        valueY = WeightCalcAlphaTwoThirds(valueY, beta);
        pWeightsGradientXData[col + row * mImageWidth] = valueX;
        pWeightsGradientYData[col + row * mImageWidth] = valueY;
      }
    }

    // TODO: Debug prints
    //		cv::dft(outputImage, outputImage,
    //				cv::DFT_INVERSE | cv::DFT_SCALE |
    // cv::DFT_REAL_OUTPUT); 		cv::imshow("outputImage", outputImage);
    cv::imshow("weightsGradientX", weightsGradientX);
    cv::imshow("weightsGradientY", weightsGradientY);
    cv::waitKey(0);
    std::cout << "beta: " << beta << std::endl;

    // TODO: fix!!!
    initialKernelImage.copyTo(kernelImage);

    beta *= 2 * sqrt(2);
  }

  cv::dft(outputImage, outputImage,
          cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

  convertToOutput(mOutputImageData, mImageWidth, mImageHeight, outputImagePpln,
                  outputImageData);
}

size_t OcvFftTvDeblurrer::getMemorySize(int imageWidth, int imageHeight,
                                        [[maybe_unused]] int kernelWidth,
                                        [[maybe_unused]] int kernelHeight) {
  return sizeof(point_value_t) * 3 * imageWidth * imageHeight;
}

void OcvFftTvDeblurrer::buildKernelForFft(const point_value_t* kernelData,
                                          int kernelWidth, int kernelHeight,
                                          int kernelPpln,
                                          point_value_t* kernelFftData) {
  // Clear image for kernel.
  memset(kernelFftData, 0, mImageWidth * mImageHeight * sizeof(point_value_t));

  int kernelCenterX = kernelWidth / 2;
  int kernelCenterY = kernelHeight / 2;

  for (int row = 0; row < kernelHeight; row++) {
    for (int col = 0; col < kernelWidth; col++) {
      point_value_t inputKernelValue = kernelData[col + row * kernelPpln];
      if (0 != inputKernelValue) {
        int kernelFftX = col - kernelCenterX;
        if (kernelFftX < 0) {
          kernelFftX += mImageWidth;
        } else if (kernelFftX > mImageWidth - 1) {
          kernelFftX -= mImageWidth;
        }

        int kernelFftY = row - kernelCenterY;
        if (kernelFftY < 0) {
          kernelFftY += mImageHeight;
        } else if (kernelFftY > mImageHeight - 1) {
          kernelFftY -= mImageHeight;
        }

        kernelFftData[kernelFftX + mImageWidth * kernelFftY] = inputKernelValue;
      }

    }  // for (int col ...
  }  // for (int row ...
}

void OcvFftTvDeblurrer::invertFftMatrix(cv::Mat& matrixFft,
                                        float minimalValue) {
  float realValue, imaginaryValue,
      sum;  // temporal variables for matrix inversion

  uchar* matrixData = matrixFft.data;

  size_t matrixStep = matrixFft.step;

  int matrixWidth = matrixFft.cols;
  int matrixHeight = matrixFft.rows;

  int matrixHalfWidth =
      ((matrixWidth % 2 == 0) ? matrixWidth - 2 : matrixWidth - 1);
  int matrixHalfHeight =
      ((matrixHeight % 2 == 0) ? matrixHeight - 2 : matrixHeight - 1);

  // sets upper left
  float upperLeftValue = ((float*)matrixData)[0];
  ((float*)matrixData)[0] =
      upperLeftValue / (upperLeftValue * upperLeftValue + minimalValue);

  // set first column
  for (int row = 1; row < matrixHalfHeight; row += 2) {
    realValue = ((float*)(matrixData + row * matrixStep))[0];
    imaginaryValue = ((float*)(matrixData + (row + 1) * matrixStep))[0];
    sum =
        realValue * realValue + imaginaryValue * imaginaryValue + minimalValue;
    ((float*)(matrixData + row * matrixStep))[0] = realValue / sum;
    ((float*)(matrixData + (row + 1) * matrixStep))[0] = -imaginaryValue / sum;
  }

  // sets down left if needed
  if (matrixHeight % 2 == 0) {
    float downLeftValue =
        ((float*)(matrixData + (matrixHeight - 1) * matrixStep))[0];
    ((float*)(matrixData + (matrixHeight - 1) * matrixStep))[0] =
        downLeftValue / (downLeftValue * downLeftValue + minimalValue);
  }

  if (matrixWidth % 2 == 0) {
    // sets upper right
    float upperLeftValue = ((float*)matrixData)[matrixWidth - 1];
    ((float*)matrixData)[matrixWidth - 1] =
        upperLeftValue / (upperLeftValue * upperLeftValue + minimalValue);

    // set last column
    for (int row = 1; row < matrixHalfHeight; row += 2) {
      realValue = ((float*)(matrixData + row * matrixStep))[matrixWidth - 1];
      imaginaryValue =
          ((float*)(matrixData + (row + 1) * matrixStep))[matrixWidth - 1];
      sum = realValue * realValue + imaginaryValue * imaginaryValue +
            minimalValue;
      ((float*)(matrixData + row * matrixStep))[matrixWidth - 1] =
          realValue / sum;
      ((float*)(matrixData + (row + 1) * matrixStep))[matrixWidth - 1] =
          -imaginaryValue / sum;
    }

    // sets down right
    if (matrixHeight % 2 == 0) {
      float downRightValue =
          ((float*)(matrixData +
                    (matrixHeight - 1) * matrixStep))[matrixWidth - 1];
      ((float*)(matrixData +
                (matrixHeight - 1) * matrixStep))[matrixWidth - 1] =
          downRightValue / (downRightValue * downRightValue + minimalValue);
    }
  }

  for (int row = 0; row < matrixHeight; row++) {
    for (int col = 1; col < matrixHalfWidth; col += 2) {
      realValue = ((float*)(matrixData + row * matrixStep))[col];
      imaginaryValue = ((float*)(matrixData + row * matrixStep))[col + 1];
      sum = realValue * realValue + imaginaryValue * imaginaryValue +
            minimalValue;
      ((float*)(matrixData + row * matrixStep))[col] = realValue / sum;
      ((float*)(matrixData + row * matrixStep))[col + 1] =
          (-imaginaryValue / sum);
    }
  }
}

void OcvFftTvDeblurrer::invertFftMatrix(cv::Mat& matrixFft) {
  float realValue, imaginaryValue,
      sum;  // temporal variables for matrix inversion

  uchar* matrixData = matrixFft.data;

  size_t matrixStep = matrixFft.step;

  int matrixWidth = matrixFft.cols;
  int matrixHeight = matrixFft.rows;

  int matrixHalfWidth =
      ((matrixWidth % 2 == 0) ? matrixWidth - 2 : matrixWidth - 1);
  int matrixHalfHeight =
      ((matrixHeight % 2 == 0) ? matrixHeight - 2 : matrixHeight - 1);

  // sets upper left
  float upperLeftValue = ((float*)matrixData)[0];
  if (upperLeftValue != 0) {
    ((float*)matrixData)[0] = 1.0 / upperLeftValue;
  } else {
    ((float*)matrixData)[0] = 0.0;
  }

  // set first column
  for (int row = 1; row < matrixHalfHeight; row += 2) {
    realValue = ((float*)(matrixData + row * matrixStep))[0];
    imaginaryValue = ((float*)(matrixData + (row + 1) * matrixStep))[0];
    sum = realValue * realValue + imaginaryValue * imaginaryValue;
    if (sum != 0) {
      ((float*)(matrixData + row * matrixStep))[0] = realValue / sum;
      ((float*)(matrixData + (row + 1) * matrixStep))[0] =
          -imaginaryValue / sum;
    } else {
      ((float*)(matrixData + row * matrixStep))[0] = 0.0;
      ((float*)(matrixData + (row + 1) * matrixStep))[0] = 0.0;
    }
  }

  // sets down left if needed
  if (matrixHeight % 2 == 0) {
    float downLeftValue =
        ((float*)(matrixData + (matrixHeight - 1) * matrixStep))[0];
    if (downLeftValue != 0) {
      ((float*)(matrixData + (matrixHeight - 1) * matrixStep))[0] =
          1.0 / downLeftValue;
    } else {
      ((float*)(matrixData + (matrixHeight - 1) * matrixStep))[0] = 0.0;
    }
  }

  if (matrixWidth % 2 == 0) {
    // sets upper right
    float upperLeftValue = ((float*)matrixData)[matrixWidth - 1];
    if (upperLeftValue != 0) {
      ((float*)matrixData)[matrixWidth - 1] = 1.0 / upperLeftValue;
    } else {
      ((float*)matrixData)[matrixWidth - 1] = 0.0;
    }

    // set last column
    for (int row = 1; row < matrixHalfHeight; row += 2) {
      realValue = ((float*)(matrixData + row * matrixStep))[matrixWidth - 1];
      imaginaryValue =
          ((float*)(matrixData + (row + 1) * matrixStep))[matrixWidth - 1];
      sum = realValue * realValue + imaginaryValue * imaginaryValue;
      if (sum != 0) {
        ((float*)(matrixData + row * matrixStep))[matrixWidth - 1] =
            realValue / sum;
        ((float*)(matrixData + (row + 1) * matrixStep))[matrixWidth - 1] =
            -imaginaryValue / sum;
      } else {
        ((float*)(matrixData + row * matrixStep))[matrixWidth - 1] = 0.0;
        ((float*)(matrixData + (row + 1) * matrixStep))[matrixWidth - 1] = 0.0;
      }
    }

    // sets down right
    if (matrixHeight % 2 == 0) {
      float downRightValue =
          ((float*)(matrixData +
                    (matrixHeight - 1) * matrixStep))[matrixWidth - 1];
      if (downRightValue != 0) {
        ((float*)(matrixData +
                  (matrixHeight - 1) * matrixStep))[matrixWidth - 1] =
            1.0 / downRightValue;
      } else {
        ((float*)(matrixData +
                  (matrixHeight - 1) * matrixStep))[matrixWidth - 1] = 0.0;
      }
    }
  }

  for (int row = 0; row < matrixHeight; row++) {
    for (int col = 1; col < matrixHalfWidth; col += 2) {
      realValue = ((float*)(matrixData + row * matrixStep))[col];
      imaginaryValue = ((float*)(matrixData + row * matrixStep))[col + 1];
      sum = realValue * realValue + imaginaryValue * imaginaryValue;
      if (sum != 0) {
        ((float*)(matrixData + row * matrixStep))[col] = realValue / sum;
        ((float*)(matrixData + row * matrixStep))[col + 1] =
            (-imaginaryValue / sum);
      } else {
        ((float*)(matrixData + row * matrixStep))[col] = 0.0;
        ((float*)(matrixData + row * matrixStep))[col + 1] = 0.0;
      }
    }
  }
}

void OcvFftTvDeblurrer::regularizedInvertFftMatrix(
    cv::Mat& matrixFft, const cv::Mat& matrixRegularization,
    float regularizationWeight) {
  float realValue, imaginaryValue, realRegularizationValue,
      imaginaryRegularizationValue,
      sum;  // temporal variables for matrix inversion

  uchar* matrixData = matrixFft.data;
  uchar* regularizationData = matrixRegularization.data;

  size_t matrixStep = matrixFft.step;

  int matrixWidth = matrixFft.cols;
  int matrixHeight = matrixFft.rows;

  int matrixHalfWidth =
      ((matrixWidth % 2 == 0) ? matrixWidth - 2 : matrixWidth - 1);
  int matrixHalfHeight =
      ((matrixHeight % 2 == 0) ? matrixHeight - 2 : matrixHeight - 1);

  // sets upper left
  float upperLeftValue = ((float*)matrixData)[0];
  float upperLeftRegularizationValue = ((float*)regularizationData)[0];
  ((float*)matrixData)[0] =
      upperLeftValue / (upperLeftValue * upperLeftValue +
                        regularizationWeight * (upperLeftRegularizationValue *
                                                upperLeftRegularizationValue));

  // set first column
  for (int row = 1; row < matrixHalfHeight; row += 2) {
    realValue = ((float*)(matrixData + row * matrixStep))[0];
    imaginaryValue = ((float*)(matrixData + (row + 1) * matrixStep))[0];
    realRegularizationValue =
        ((float*)(regularizationData + row * matrixStep))[0];
    imaginaryRegularizationValue =
        ((float*)(regularizationData + (row + 1) * matrixStep))[0];
    sum = realValue * realValue + imaginaryValue * imaginaryValue +
          regularizationWeight *
              (realRegularizationValue * realRegularizationValue +
               imaginaryRegularizationValue * imaginaryRegularizationValue);
    ((float*)(matrixData + row * matrixStep))[0] = realValue / sum;
    ((float*)(matrixData + (row + 1) * matrixStep))[0] = -imaginaryValue / sum;
  }

  // sets down left if needed
  if (matrixHeight % 2 == 0) {
    float downLeftValue =
        ((float*)(matrixData + (matrixHeight - 1) * matrixStep))[0];
    float downLeftRegularizationValue =
        ((float*)(regularizationData + (matrixHeight - 1) * matrixStep))[0];
    ((float*)(matrixData + (matrixHeight - 1) * matrixStep))[0] =
        downLeftValue / (downLeftValue * downLeftValue +
                         regularizationWeight * (downLeftRegularizationValue *
                                                 downLeftRegularizationValue));
  }

  if (matrixWidth % 2 == 0) {
    // sets upper right
    float upperLeftValue = ((float*)matrixData)[matrixWidth - 1];
    float upperLeftRegularizationValue =
        ((float*)regularizationData)[matrixWidth - 1];
    ((float*)matrixData)[matrixWidth - 1] =
        upperLeftValue / (upperLeftValue * upperLeftValue +
                          regularizationWeight * upperLeftRegularizationValue *
                              upperLeftRegularizationValue);

    // set last column
    for (int row = 1; row < matrixHalfHeight; row += 2) {
      realValue = ((float*)(matrixData + row * matrixStep))[matrixWidth - 1];
      imaginaryValue =
          ((float*)(matrixData + (row + 1) * matrixStep))[matrixWidth - 1];
      realRegularizationValue =
          ((float*)(regularizationData + row * matrixStep))[matrixWidth - 1];
      imaginaryRegularizationValue =
          ((float*)(regularizationData +
                    (row + 1) * matrixStep))[matrixWidth - 1];
      sum = realValue * realValue + imaginaryValue * imaginaryValue +
            regularizationWeight *
                (realRegularizationValue * realRegularizationValue +
                 imaginaryRegularizationValue * imaginaryRegularizationValue);
      ((float*)(matrixData + row * matrixStep))[matrixWidth - 1] =
          realValue / sum;
      ((float*)(matrixData + (row + 1) * matrixStep))[matrixWidth - 1] =
          -imaginaryValue / sum;
    }

    // sets down right
    if (matrixHeight % 2 == 0) {
      float downRightValue =
          ((float*)(matrixData +
                    (matrixHeight - 1) * matrixStep))[matrixWidth - 1];
      float downRightRegularizedValue =
          ((float*)(regularizationData +
                    (matrixHeight - 1) * matrixStep))[matrixWidth - 1];
      ((float*)(matrixData +
                (matrixHeight - 1) * matrixStep))[matrixWidth - 1] =
          downRightValue / (downRightValue * downRightValue +
                            regularizationWeight * (downRightRegularizedValue *
                                                    downRightRegularizedValue));
    }
  }

  for (int row = 0; row < matrixHeight; row++) {
    for (int col = 1; col < matrixHalfWidth; col += 2) {
      realValue = ((float*)(matrixData + row * matrixStep))[col];
      imaginaryValue = ((float*)(matrixData + row * matrixStep))[col + 1];
      realRegularizationValue =
          ((float*)(regularizationData + row * matrixStep))[col];
      imaginaryRegularizationValue =
          ((float*)(regularizationData + row * matrixStep))[col + 1];
      sum = realValue * realValue + imaginaryValue * imaginaryValue +
            regularizationWeight *
                (realRegularizationValue * realRegularizationValue +
                 imaginaryRegularizationValue * imaginaryRegularizationValue);
      ((float*)(matrixData + row * matrixStep))[col] = realValue / sum;
      ((float*)(matrixData + row * matrixStep))[col + 1] =
          (-imaginaryValue / sum);
    }
  }
}

void OcvFftTvDeblurrer::regularizedInvertFftMatrix(
    cv::Mat& matrixFft, const cv::Mat& firstMatrixRegularization,
    const cv::Mat& secondMatrixRegularization, float regularizationWeight) {
  float realValue, imaginaryValue, firstRealRegularizationValue,
      firstImaginaryRegularizationValue, secondRealRegularizationValue,
      secondImaginaryRegularizationValue,
      sum;  // temporal variables for matrix inversion

  uchar* matrixData = matrixFft.data;
  uchar* firstRegularizationData = firstMatrixRegularization.data;
  uchar* secondRegularizationData = secondMatrixRegularization.data;

  size_t matrixStep = matrixFft.step;

  int matrixWidth = matrixFft.cols;
  int matrixHeight = matrixFft.rows;

  int matrixHalfWidth =
      ((matrixWidth % 2 == 0) ? matrixWidth - 2 : matrixWidth - 1);
  int matrixHalfHeight =
      ((matrixHeight % 2 == 0) ? matrixHeight - 2 : matrixHeight - 1);

  // sets upper left
  float upperLeftValue = ((float*)matrixData)[0];
  float upperLeftRegularizationFirstValue =
      ((float*)firstRegularizationData)[0];
  float upperLeftRegularizationSecondValue =
      ((float*)secondRegularizationData)[0];
  ((float*)matrixData)[0] =
      upperLeftValue /
      (upperLeftValue * upperLeftValue +
       regularizationWeight * (upperLeftRegularizationFirstValue *
                                   upperLeftRegularizationFirstValue +
                               upperLeftRegularizationSecondValue *
                                   upperLeftRegularizationSecondValue));

  // set first column
  for (int row = 1; row < matrixHalfHeight; row += 2) {
    realValue = ((float*)(matrixData + row * matrixStep))[0];
    imaginaryValue = ((float*)(matrixData + (row + 1) * matrixStep))[0];
    firstRealRegularizationValue =
        ((float*)(firstRegularizationData + row * matrixStep))[0];
    firstImaginaryRegularizationValue =
        ((float*)(firstRegularizationData + (row + 1) * matrixStep))[0];
    secondRealRegularizationValue =
        ((float*)(secondRegularizationData + row * matrixStep))[0];
    secondImaginaryRegularizationValue =
        ((float*)(secondRegularizationData + (row + 1) * matrixStep))[0];
    sum = realValue * realValue + imaginaryValue * imaginaryValue +
          regularizationWeight *
              (firstRealRegularizationValue * firstRealRegularizationValue +
               firstImaginaryRegularizationValue *
                   firstImaginaryRegularizationValue +
               secondRealRegularizationValue * secondRealRegularizationValue +
               secondImaginaryRegularizationValue *
                   secondImaginaryRegularizationValue);
    ((float*)(matrixData + row * matrixStep))[0] = realValue / sum;
    ((float*)(matrixData + (row + 1) * matrixStep))[0] = -imaginaryValue / sum;
  }

  // sets down left if needed
  if (matrixHeight % 2 == 0) {
    float downLeftValue =
        ((float*)(matrixData + (matrixHeight - 1) * matrixStep))[0];
    float downLeftRegularizationFirstValue = ((
        float*)(firstRegularizationData + (matrixHeight - 1) * matrixStep))[0];
    float downLeftRegularizationSecondValue = ((
        float*)(secondRegularizationData + (matrixHeight - 1) * matrixStep))[0];
    ((float*)(matrixData + (matrixHeight - 1) * matrixStep))[0] =
        downLeftValue /
        (downLeftValue * downLeftValue +
         regularizationWeight * (downLeftRegularizationFirstValue *
                                     downLeftRegularizationFirstValue +
                                 downLeftRegularizationSecondValue *
                                     downLeftRegularizationSecondValue));
  }

  if (matrixWidth % 2 == 0) {
    // sets upper right
    float upperLeftValue = ((float*)matrixData)[matrixWidth - 1];
    float upperLeftRegularizationFirstValue =
        ((float*)firstRegularizationData)[matrixWidth - 1];
    float upperLeftRegularizationSecondValue =
        ((float*)secondRegularizationData)[matrixWidth - 1];
    ((float*)matrixData)[matrixWidth - 1] =
        upperLeftValue /
        (upperLeftValue * upperLeftValue +
         regularizationWeight * (upperLeftRegularizationFirstValue *
                                     upperLeftRegularizationFirstValue +
                                 upperLeftRegularizationSecondValue *
                                     upperLeftRegularizationSecondValue));

    // set last column
    for (int row = 1; row < matrixHalfHeight; row += 2) {
      realValue = ((float*)(matrixData + row * matrixStep))[matrixWidth - 1];
      imaginaryValue =
          ((float*)(matrixData + (row + 1) * matrixStep))[matrixWidth - 1];
      firstRealRegularizationValue = ((
          float*)(firstRegularizationData + row * matrixStep))[matrixWidth - 1];
      firstImaginaryRegularizationValue =
          ((float*)(firstRegularizationData +
                    (row + 1) * matrixStep))[matrixWidth - 1];
      secondRealRegularizationValue =
          ((float*)(secondRegularizationData +
                    row * matrixStep))[matrixWidth - 1];
      secondImaginaryRegularizationValue =
          ((float*)(secondRegularizationData +
                    (row + 1) * matrixStep))[matrixWidth - 1];
      sum = realValue * realValue + imaginaryValue * imaginaryValue +
            regularizationWeight *
                (firstRealRegularizationValue * firstRealRegularizationValue +
                 firstImaginaryRegularizationValue *
                     firstImaginaryRegularizationValue +
                 secondRealRegularizationValue * secondRealRegularizationValue +
                 secondImaginaryRegularizationValue *
                     secondImaginaryRegularizationValue);
      ((float*)(matrixData + row * matrixStep))[matrixWidth - 1] =
          realValue / sum;
      ((float*)(matrixData + (row + 1) * matrixStep))[matrixWidth - 1] =
          -imaginaryValue / sum;
    }

    // sets down right
    if (matrixHeight % 2 == 0) {
      float downRightValue =
          ((float*)(matrixData +
                    (matrixHeight - 1) * matrixStep))[matrixWidth - 1];
      float downRightRegularizedFirstValue =
          ((float*)(firstRegularizationData +
                    (matrixHeight - 1) * matrixStep))[matrixWidth - 1];
      float downRightRegularizedSecondValue =
          ((float*)(secondRegularizationData +
                    (matrixHeight - 1) * matrixStep))[matrixWidth - 1];
      ((float*)(matrixData +
                (matrixHeight - 1) * matrixStep))[matrixWidth - 1] =
          downRightValue /
          (downRightValue * downRightValue +
           regularizationWeight * (downRightRegularizedFirstValue *
                                       downRightRegularizedFirstValue +
                                   downRightRegularizedSecondValue *
                                       downRightRegularizedSecondValue));
    }
  }

  for (int row = 0; row < matrixHeight; row++) {
    for (int col = 1; col < matrixHalfWidth; col += 2) {
      realValue = ((float*)(matrixData + row * matrixStep))[col];
      imaginaryValue = ((float*)(matrixData + row * matrixStep))[col + 1];
      firstRealRegularizationValue =
          ((float*)(firstRegularizationData + row * matrixStep))[col];
      firstImaginaryRegularizationValue =
          ((float*)(firstRegularizationData + row * matrixStep))[col + 1];
      secondRealRegularizationValue =
          ((float*)(secondRegularizationData + row * matrixStep))[col];
      secondImaginaryRegularizationValue =
          ((float*)(secondRegularizationData + row * matrixStep))[col + 1];
      sum = realValue * realValue + imaginaryValue * imaginaryValue +
            regularizationWeight *
                (firstRealRegularizationValue * firstRealRegularizationValue +
                 firstImaginaryRegularizationValue *
                     firstImaginaryRegularizationValue +
                 secondRealRegularizationValue * secondRealRegularizationValue +
                 secondImaginaryRegularizationValue *
                     secondImaginaryRegularizationValue);
      ((float*)(matrixData + row * matrixStep))[col] = realValue / sum;
      ((float*)(matrixData + row * matrixStep))[col + 1] =
          (-imaginaryValue / sum);
    }
  }
}

void OcvFftTvDeblurrer::setIdentityRegularization() {
  cv::Mat regularizationMatrixIdentity(mImageHeight, mImageWidth, CV_32FC1,
                                       mIdentityRegularizationData);
  regularizationMatrixIdentity = cv::Scalar(0);
  regularizationMatrixIdentity.at<float>(cv::Point(0, 0)) = 1.0;
  cv::dft(regularizationMatrixIdentity, regularizationMatrixIdentity);
}

void OcvFftTvDeblurrer::setLaplaceRegularization() {
  cv::Mat regularizationMatrixLaplacian(mImageHeight, mImageWidth, CV_32FC1,
                                        mLaplacianRegularizationData);
  regularizationMatrixLaplacian = cv::Scalar(0);
  regularizationMatrixLaplacian.at<float>(cv::Point(0, 0)) = 4.0;
  regularizationMatrixLaplacian.at<float>(cv::Point(0, 1)) = -1.0;
  regularizationMatrixLaplacian.at<float>(cv::Point(1, 0)) = -1.0;
  regularizationMatrixLaplacian.at<float>(
      cv::Point(0, regularizationMatrixLaplacian.rows - 1)) =
      regularizationMatrixLaplacian.at<float>(cv::Point(0, 1));
  regularizationMatrixLaplacian.at<float>(
      cv::Point(regularizationMatrixLaplacian.cols - 1, 0)) =
      regularizationMatrixLaplacian.at<float>(cv::Point(1, 0));
  regularizationMatrixLaplacian.at<float>(
      cv::Point(regularizationMatrixLaplacian.cols - 1,
                regularizationMatrixLaplacian.rows - 1)) =
      regularizationMatrixLaplacian.at<float>(cv::Point(1, 1));
  cv::dft(regularizationMatrixLaplacian, regularizationMatrixLaplacian);
}

void OcvFftTvDeblurrer::setGradientXRegularization() {
  cv::Mat regularizationMatrixGradientX(mImageHeight, mImageWidth, CV_32FC1,
                                        mGradientXRegularizationData);
  regularizationMatrixGradientX = cv::Scalar(0);
  regularizationMatrixGradientX.at<float>(
      cv::Point(1, regularizationMatrixGradientX.rows - 1)) = 3.0;
  regularizationMatrixGradientX.at<float>(cv::Point(1, 0)) = 10.0;
  regularizationMatrixGradientX.at<float>(cv::Point(1, 1)) = 3.0;
  regularizationMatrixGradientX.at<float>(
      cv::Point(regularizationMatrixGradientX.cols - 1,
                regularizationMatrixGradientX.rows - 1)) = -3.0;
  regularizationMatrixGradientX.at<float>(
      cv::Point(regularizationMatrixGradientX.cols - 1, 0)) = -10.0;
  regularizationMatrixGradientX.at<float>(
      cv::Point(regularizationMatrixGradientX.cols - 1, 1)) = -3.0;
  cv::dft(regularizationMatrixGradientX, regularizationMatrixGradientX);
}

void OcvFftTvDeblurrer::setGradientYRegularization() {
  cv::Mat regularizationMatrixGradientY(mImageHeight, mImageWidth, CV_32FC1,
                                        mGradientYRegularizationData);
  regularizationMatrixGradientY = cv::Scalar(0);

  regularizationMatrixGradientY.at<float>(
      cv::Point(regularizationMatrixGradientY.cols - 1, 1)) = 3.0;
  regularizationMatrixGradientY.at<float>(cv::Point(0, 1)) = 10.0;
  regularizationMatrixGradientY.at<float>(cv::Point(1, 1)) = 3.0;

  regularizationMatrixGradientY.at<float>(
      cv::Point(regularizationMatrixGradientY.cols - 1,
                regularizationMatrixGradientY.rows - 1)) = -3.0;
  regularizationMatrixGradientY.at<float>(
      cv::Point(0, regularizationMatrixGradientY.rows - 1)) = -10.0;
  regularizationMatrixGradientY.at<float>(
      cv::Point(1, regularizationMatrixGradientY.rows - 1)) = -3.0;
  cv::dft(regularizationMatrixGradientY, regularizationMatrixGradientY);
}

void OcvFftTvDeblurrer::calculateFftMatrixForHqpm(
    const cv::Mat& kernel, const cv::Mat& firstMatrixRegularization,
    const cv::Mat& secondMatrixRegularization, const cv::Mat& firstWeight,
    const cv::Mat& secondWeight, float lambdaOverBeta, cv::Mat& output) {
  float reV, imV, firstReRegV, firstImRegV, secondReRegV, secondImRegV,
      firstReW, firstImW, secondReW, secondImW,
      sum;  // temporal variables for matrix inversion

  float* kernelData = (float*)kernel.data;
  float* firstRegData = (float*)firstMatrixRegularization.data;
  float* secondRegData = (float*)secondMatrixRegularization.data;

  float* firstWData = (float*)firstWeight.data;
  float* secondWData = (float*)secondWeight.data;

  float* outputData = (float*)output.data;

  int imagePpln = kernel.step / sizeof(float);

  int matrixWidth = kernel.cols;
  int matrixHeight = kernel.rows;

  int matrixHalfWidth =
      ((matrixWidth % 2 == 0) ? matrixWidth - 2 : matrixWidth - 1);
  int matrixHalfHeight =
      ((matrixHeight % 2 == 0) ? matrixHeight - 2 : matrixHeight - 1);

  // sets upper left
  float upperLeftValue = kernelData[0];
  float upperLeftRegFirstValue = firstRegData[0];
  float upperLeftRegSecondValue = secondRegData[0];
  float upperLeftWFirstValue = firstWData[0];
  float upperLeftWSecondValue = secondWData[0];
  outputData[0] = (lambdaOverBeta * upperLeftValue +
                   upperLeftRegFirstValue * upperLeftWFirstValue +
                   upperLeftRegSecondValue * upperLeftWSecondValue) /
                  (lambdaOverBeta * upperLeftValue * upperLeftValue +
                   upperLeftRegFirstValue * upperLeftRegFirstValue +
                   upperLeftRegSecondValue * upperLeftRegSecondValue);

  // set first column
  for (int row = 1; row < matrixHalfHeight; row += 2) {
    reV = (kernelData + row * imagePpln)[0];
    imV = (kernelData + (row + 1) * imagePpln)[0];
    firstReRegV = (firstRegData + row * imagePpln)[0];
    firstImRegV = (firstRegData + (row + 1) * imagePpln)[0];
    secondReRegV = (secondRegData + row * imagePpln)[0];
    secondImRegV = (secondRegData + (row + 1) * imagePpln)[0];
    firstReW = (firstWData + row * imagePpln)[0];
    firstImW = (firstWData + (row + 1) * imagePpln)[0];
    secondReW = (secondWData + row * imagePpln)[0];
    secondImW = (secondWData + (row + 1) * imagePpln)[0];
    sum = lambdaOverBeta * (reV * reV + imV * imV) + firstReRegV * firstReRegV +
          firstImRegV * firstImRegV + secondReRegV * secondReRegV +
          secondImRegV * secondImRegV;
    (outputData + row * imagePpln)[0] =
        (lambdaOverBeta * reV + firstReRegV * firstReW +
         firstImRegV * firstImW + secondReRegV * secondReW +
         secondImRegV * secondImW) /
        sum;
    (outputData + (row + 1) * imagePpln)[0] =
        (-lambdaOverBeta * imV + firstReRegV * firstImW -
         firstImRegV * firstReW + secondReRegV * secondImW -
         secondImRegV * secondReW) /
        sum;
  }

  // sets down left if needed
  if (matrixHeight % 2 == 0) {
    float downLeftValue = (kernelData + (matrixHeight - 1) * imagePpln)[0];
    float downLeftRegFirstValue =
        (firstRegData + (matrixHeight - 1) * imagePpln)[0];
    float downLeftRegSecondValue =
        (secondRegData + (matrixHeight - 1) * imagePpln)[0];
    float downLeftWFirstValue =
        (firstWData + (matrixHeight - 1) * imagePpln)[0];
    float downLeftWSecondValue =
        (secondWData + (matrixHeight - 1) * imagePpln)[0];
    (outputData + (matrixHeight - 1) * imagePpln)[0] =
        (lambdaOverBeta * downLeftValue +
         downLeftRegFirstValue * downLeftWFirstValue +
         downLeftRegSecondValue * downLeftWSecondValue) /
        (downLeftValue * downLeftValue * lambdaOverBeta +
         downLeftRegFirstValue * downLeftRegFirstValue +
         downLeftRegSecondValue * downLeftRegSecondValue);
  }

  if (matrixWidth % 2 == 0) {
    // sets upper right
    float upperRightValue = kernelData[matrixWidth - 1];
    float upperRightRegFirstValue = firstRegData[matrixWidth - 1];
    float upperRightRegSecondValue = secondWData[matrixWidth - 1];
    float upperRightWFirstValue = firstWData[matrixWidth - 1];
    float upperRightWSecondValue = secondRegData[matrixWidth - 1];
    outputData[matrixWidth - 1] =
        (lambdaOverBeta * upperRightValue +
         upperRightRegFirstValue * upperRightWFirstValue +
         upperRightRegSecondValue * upperRightWSecondValue) /
        (lambdaOverBeta * upperRightValue * upperRightValue +
         upperRightRegFirstValue * upperRightRegFirstValue +
         upperRightRegSecondValue * upperRightRegSecondValue);

    // set last column
    for (int row = 1; row < matrixHalfHeight; row += 2) {
      firstReRegV = (firstRegData + row * imagePpln)[matrixWidth - 1];
      firstImRegV = (firstRegData + (row + 1) * imagePpln)[matrixWidth - 1];
      secondReRegV = (secondRegData + row * imagePpln)[matrixWidth - 1];
      secondImRegV = (secondRegData + (row + 1) * imagePpln)[matrixWidth - 1];
      firstReW = (firstWData + row * imagePpln)[matrixWidth - 1];
      firstImW = (firstWData + (row + 1) * imagePpln)[matrixWidth - 1];
      secondReW = (secondWData + row * imagePpln)[matrixWidth - 1];
      secondImW = (secondWData + (row + 1) * imagePpln)[matrixWidth - 1];
      reV = (kernelData + row * imagePpln)[matrixWidth - 1];
      imV = (kernelData + (row + 1) * imagePpln)[matrixWidth - 1];
      reV += firstReRegV * firstReW + firstImRegV * firstImW +
             secondReRegV * secondReW + secondImRegV * secondImW;
      imV += firstReRegV * firstImW - firstImRegV * firstReW +
             secondReRegV * secondImW - secondImRegV * secondReW;
      sum = lambdaOverBeta * (reV * reV + imV * imV) +
            (firstReRegV * firstReRegV + firstImRegV * firstImRegV +
             secondReRegV * secondReRegV + secondImRegV * secondImRegV);
      (outputData + row * imagePpln)[matrixWidth - 1] = reV / sum;
      (outputData + (row + 1) * imagePpln)[matrixWidth - 1] = -imV / sum;
    }

    // sets down right
    if (matrixHeight % 2 == 0) {
      float downRightValue =
          (kernelData + (matrixHeight - 1) * imagePpln)[matrixWidth - 1];
      float downRightRegFirstValue =
          (firstRegData + (matrixHeight - 1) * imagePpln)[matrixWidth - 1];
      float downRightRegSecondValue =
          (secondRegData + (matrixHeight - 1) * imagePpln)[matrixWidth - 1];
      float downRightWFirstValue =
          (firstWData + (matrixHeight - 1) * imagePpln)[matrixWidth - 1];
      float downRightWSecondValue =
          (secondWData + (matrixHeight - 1) * imagePpln)[matrixWidth - 1];
      (outputData + (matrixHeight - 1) * imagePpln)[matrixWidth - 1] =
          (lambdaOverBeta * downRightValue +
           downRightRegFirstValue * downRightWFirstValue +
           downRightRegSecondValue * downRightWSecondValue) /
          (lambdaOverBeta * downRightValue * downRightValue +
           downRightRegFirstValue * downRightRegFirstValue +
           downRightRegSecondValue * downRightRegSecondValue);
    }
  }

  for (int row = 0; row < matrixHeight; row++) {
    for (int col = 1; col < matrixHalfWidth; col += 2) {
      firstReRegV = (firstRegData + row * imagePpln)[col];
      firstImRegV = (firstRegData + row * imagePpln)[col + 1];
      secondReRegV = (secondRegData + row * imagePpln)[col];
      secondImRegV = (secondRegData + row * imagePpln)[col + 1];
      firstReW = (firstRegData + row * imagePpln)[col];
      firstImW = (firstRegData + row * imagePpln)[col + 1];
      secondReW = (secondRegData + row * imagePpln)[col];
      secondImW = (secondRegData + row * imagePpln)[col + 1];
      reV = (kernelData + row * imagePpln)[col];
      imV = (kernelData + row * imagePpln)[col + 1];
      reV += firstReRegV * firstReW + firstImRegV * firstImW +
             secondReRegV * secondReW + secondImRegV * secondImW;
      imV += firstReRegV * firstImW - firstImRegV * firstReW +
             secondReRegV * secondImW - secondImRegV * secondReW;
      sum = lambdaOverBeta * (reV * reV + imV * imV) +
            firstReRegV * firstReRegV + firstImRegV * firstImRegV +
            secondReRegV * secondReRegV + secondImRegV * secondImRegV;
      (outputData + row * imagePpln)[col] = reV / sum;
      (outputData + row * imagePpln)[col + 1] = (-imV / sum);
    }
  }
}

void OcvFftTvDeblurrer::initializeHqpmWeights() {
  cv::Mat weightsGradientX(mImageHeight, mImageWidth, CV_32FC1,
                           mGradientXWeightData);
  weightsGradientX = cv::Scalar(1);
  cv::Mat weightsGradientY(mImageHeight, mImageWidth, CV_32FC1,
                           mGradientYWeightData);
  weightsGradientY = cv::Scalar(1);
}

float OcvFftTvDeblurrer::WeightCalcAlphaTwoThirds(float x, float beta) {
  float value = 0;  // return value
  float m = 8.0 / (27 * beta * beta * beta);
  float t1 = 0.25 * x * x;
  float t2 = 27.0 * x * x * x * x * m * m - 256.0 * m * m * m;
  if (t2 < 0) return (value);
  float t3, t4, t5, t6;
  if (m != 0) {
    t3 = 9.0 * x * x * m;
    t4 = exp(log(sqrt(3 * t2) + t3) / 3.0);
    t5 = t4 * 0.381571414;
    t6 = 3.494321859 * m / t4;
  } else {
    t3 = 0;
    t4 = 0;
    t5 = 0;
    t6 = 0;
  }

  float t7 = sqrt(t1 + t5 + t6);
  float t8 = (x != 0) ? x * x * x * 0.25 / t7 : 0;

  float det1 = 2 * t1 - t5 - t6 + t8;
  float det2 = det1 - 2 * t8;

  float r1, r2, r3, r4, r;

  float c1 = abs(x) / 2.0, c2 = abs(x);

  if (det1 >= 0) {
    r3 = 0.75 * x + 0.5 * (-t7 - sqrt(det1));
    r4 = 0.75 * x + 0.5 * (-t7 + sqrt(det1));
    r = std::max(r3, r4);

    if (det2 >= 0) {
      r1 = 0.75 * x + 0.5 * (t7 - sqrt(det2));
      r2 = 0.75 * x + 0.5 * (t7 + sqrt(det2));
      r = std::max(r, r1);
      r = std::max(r, r2);
    }

    if ((abs(r) >= c1) && (abs(r) <= c2))
      value = r;
    else
      value = 0;
  } else {
    value = 0;
  }

  return (value);
}

// TODO: Complete
float OcvFftTvDeblurrer::WeightCalcTV(float x, float beta) {
  float value = 0;  // return value
  float m = 8.0 / (27 * beta * beta * beta);
  float t1 = 0.25 * x * x;
  float t2 = 27.0 * x * x * x * x * m * m - 256.0 * m * m * m;
  if (t2 < 0) return (value);
  float t3, t4, t5, t6;
  if (m != 0) {
    t3 = 9.0 * x * x * m;
    t4 = exp(log(sqrt(3 * t2) + t3) / 3.0);
    t5 = t4 * 0.381571414;
    t6 = 3.494321859 * m / t4;
  } else {
    t3 = 0;
    t4 = 0;
    t5 = 0;
    t6 = 0;
  }

  float t7 = sqrt(t1 + t5 + t6);
  float t8 = (x != 0) ? x * x * x * 0.25 / t7 : 0;

  float det1 = 2 * t1 - t5 - t6 + t8;
  float det2 = det1 - 2 * t8;

  float r1, r2, r3, r4, r;

  float c1 = abs(x) / 2.0, c2 = abs(x);

  if (det1 >= 0) {
    r3 = 0.75 * x + 0.5 * (-t7 - sqrt(det1));
    r4 = 0.75 * x + 0.5 * (-t7 + sqrt(det1));
    r = std::max(r3, r4);

    if (det2 >= 0) {
      r1 = 0.75 * x + 0.5 * (t7 - sqrt(det2));
      r2 = 0.75 * x + 0.5 * (t7 + sqrt(det2));
      r = std::max(r, r1);
      r = std::max(r, r2);
    }

    if ((abs(r) >= c1) && (abs(r) <= c2))
      value = r;
    else
      value = 0;
  } else {
    value = 0;
  }

  return (value);
}

}  // namespace Deblurring
}  // namespace Test
