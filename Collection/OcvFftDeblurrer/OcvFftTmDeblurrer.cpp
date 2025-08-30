/*
 * OcvFftTmDeblurrer.cpp
 *
 *  Created on: Jan 28, 2015
 *      Author: vantonov
 */

#include "OcvFftTmDeblurrer.h"

#include "BlurKernelUtils.h"

// TODO: Debug print
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace Test {
namespace Deblurring {

OcvFftTmDeblurrer::OcvFftTmDeblurrer(int imageWidth, int imageHeight)
    : ImageDeblurrer(imageWidth, imageHeight),
      mInputImageData(NULL),
      mKernelImageData(NULL),
      mOutputImageData(NULL),
      mIdentityRegularizationData(NULL),
      mLaplacianRegularizationData(NULL),
      mGradientXRegularizationData(NULL),
      mGradientYRegularizationData(NULL),
      mRegularizationWeight(0) {
  init(imageWidth, imageHeight, NULL);
}

OcvFftTmDeblurrer::OcvFftTmDeblurrer(int imageWidth, int imageHeight,
                                     void* pExternalMemory)
    : ImageDeblurrer(imageWidth, imageHeight, pExternalMemory),
      mInputImageData(NULL),
      mKernelImageData(NULL),
      mOutputImageData(NULL),
      mIdentityRegularizationData(NULL),
      mLaplacianRegularizationData(NULL),
      mGradientXRegularizationData(NULL),
      mGradientYRegularizationData(NULL),
      mRegularizationWeight(0) {
  init(imageWidth, imageHeight, pExternalMemory);
}

OcvFftTmDeblurrer::~OcvFftTmDeblurrer() { deinit(); }

void OcvFftTmDeblurrer::init(int imageWidth, int imageHeight,
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
  } else {
    mInputImageData = new point_value_t[imageBlockSize];
    mKernelImageData = new point_value_t[imageBlockSize];
    mOutputImageData = new point_value_t[imageBlockSize];
    mIdentityRegularizationData = new point_value_t[imageBlockSize];
    mLaplacianRegularizationData = new point_value_t[imageBlockSize];
    mGradientXRegularizationData = new point_value_t[imageBlockSize];
    mGradientYRegularizationData = new point_value_t[imageBlockSize];
  }

  setIdentityRegularization();

  setLaplaceRegularization();

  setGradientXRegularization();

  setGradientYRegularization();
}

void OcvFftTmDeblurrer::deinit() {
  if (!isExternalMemoryUsed) {
    delete[] mInputImageData;
    delete[] mKernelImageData;
    delete[] mOutputImageData;
    delete[] mIdentityRegularizationData;
    delete[] mLaplacianRegularizationData;
    delete[] mGradientXRegularizationData;
    delete[] mGradientYRegularizationData;
  }

  mInputImageData = NULL;
  mKernelImageData = NULL;
  mOutputImageData = NULL;

  mIdentityRegularizationData = NULL;
  mLaplacianRegularizationData = NULL;
  mGradientXRegularizationData = NULL;
  mGradientYRegularizationData = NULL;

  mRegularizationWeight = 0;
}

void OcvFftTmDeblurrer::operator()(const uint8_t* inputImageData,
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

  cv::dft(inputImage, inputImage);
  cv::dft(kernelImage, kernelImage);

  //	invertFftMatrix(kernelImage);

  //	invertFftMatrix(kernelImage, epsilon);

  //	regularizedInvertFftMatrix(kernelImage, regularizationMatrixLaplacian,
  //			mRegularizationWeight);

  regularizedInvertFftMatrix(kernelImage, regularizationMatrixGradientX,
                             regularizationMatrixGradientY,
                             mRegularizationWeight);

  cv::mulSpectrums(inputImage, kernelImage, outputImage,
                   cv::DFT_COMPLEX_OUTPUT);
  cv::dft(outputImage, outputImage,
          cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

  convertToOutput(mOutputImageData, mImageWidth, mImageHeight, outputImagePpln,
                  outputImageData);
}

void OcvFftTmDeblurrer::operator()(const uint8_t* inputImageData,
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

  cv::dft(inputImage, inputImage);
  cv::dft(kernelImage, kernelImage);

  //	invertFftMatrix(kernelImage);

  //	invertFftMatrix(kernelImage, epsilon);

  //	regularizedInvertFftMatrix(kernelImage, regularizationMatrixLaplacian,
  //			mRegularizationWeight);

  regularizedInvertFftMatrix(kernelImage, regularizationMatrixGradientX,
                             regularizationMatrixGradientY,
                             mRegularizationWeight);

  cv::mulSpectrums(inputImage, kernelImage, outputImage,
                   cv::DFT_COMPLEX_OUTPUT);
  cv::dft(outputImage, outputImage,
          cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

  convertToOutput(mOutputImageData, mImageWidth, mImageHeight, outputImagePpln,
                  outputImageData);
}

size_t OcvFftTmDeblurrer::getMemorySize(int imageWidth, int imageHeight,
                                        [[maybe_unused]] int kernelWidth,
                                        [[maybe_unused]] int kernelHeight) {
  return sizeof(point_value_t) * 3 * imageWidth * imageHeight;
}

void OcvFftTmDeblurrer::buildKernelForFft(const point_value_t* kernelData,
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

void OcvFftTmDeblurrer::invertFftMatrix(cv::Mat& matrixFft,
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

void OcvFftTmDeblurrer::invertFftMatrix(cv::Mat& matrixFft) {
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

void OcvFftTmDeblurrer::regularizedInvertFftMatrix(
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
        upperLeftValue /
        (upperLeftValue * upperLeftValue +
         upperLeftRegularizationValue * upperLeftRegularizationValue);

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

void OcvFftTmDeblurrer::regularizedInvertFftMatrix(
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
         upperLeftRegularizationFirstValue * upperLeftRegularizationFirstValue +
         upperLeftRegularizationSecondValue *
             upperLeftRegularizationSecondValue);

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
          ((float*)(firstRegularizationData +
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

void OcvFftTmDeblurrer::setIdentityRegularization() {
  cv::Mat regularizationMatrixIdentity(mImageHeight, mImageWidth, CV_32FC1,
                                       mIdentityRegularizationData);
  regularizationMatrixIdentity = cv::Scalar(0);
  regularizationMatrixIdentity.at<float>(cv::Point(0, 0)) = 1.0;
  cv::dft(regularizationMatrixIdentity, regularizationMatrixIdentity);
}

void OcvFftTmDeblurrer::setLaplaceRegularization() {
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

void OcvFftTmDeblurrer::setGradientXRegularization() {
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

void OcvFftTmDeblurrer::setGradientYRegularization() {
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

}  // namespace Deblurring
}  // namespace Test
