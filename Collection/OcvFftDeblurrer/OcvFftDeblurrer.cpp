/*
 * OcvFftDeblurrer.cpp
 *
 *  Created on: 05.01.2015
 *      Author: vladiant
 */

#include "OcvFftDeblurrer.h"

#include <math.h>

#include "BlurKernelUtils.h"

// TODO: Debug
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace Test {
namespace Deblurring {

OcvFftDeblurrer::OcvFftDeblurrer(int imageWidth, int imageHeight)
    : ImageDeblurrer(imageWidth, imageHeight),
      mInputImageData(NULL),
      mKernelImageData(NULL),
      mOutputImageData(NULL),
      mWeightImageData(NULL) {
  init(imageWidth, imageHeight, NULL);
}

OcvFftDeblurrer::OcvFftDeblurrer(int imageWidth, int imageHeight,
                                 void* pExternalMemory)
    : ImageDeblurrer(imageWidth, imageHeight, pExternalMemory),
      mInputImageData(NULL),
      mKernelImageData(NULL),
      mOutputImageData(NULL),
      mWeightImageData(NULL) {
  init(imageWidth, imageHeight, pExternalMemory);
}

OcvFftDeblurrer::~OcvFftDeblurrer() { deinit(); }

void OcvFftDeblurrer::init(int imageWidth, int imageHeight,
                           void* pExternalMemory) {
  size_t imageBlockSize = imageWidth * imageHeight;

  if (isExternalMemoryUsed) {
    point_value_t* pDataBuffer = (point_value_t*)pExternalMemory;
    mInputImageData = &pDataBuffer[0];
    mKernelImageData = &pDataBuffer[imageBlockSize];
    mOutputImageData = &pDataBuffer[2 * imageBlockSize];
    mWeightImageData = &pDataBuffer[3 * imageBlockSize];
  } else {
    mInputImageData = new point_value_t[imageBlockSize];
    mKernelImageData = new point_value_t[imageBlockSize];
    mOutputImageData = new point_value_t[imageBlockSize];
    mWeightImageData = new point_value_t[imageBlockSize];
  }
}

void OcvFftDeblurrer::deinit() {
  if (!isExternalMemoryUsed) {
    delete[] mInputImageData;
    delete[] mKernelImageData;
    delete[] mOutputImageData;
    delete[] mWeightImageData;
  }

  mInputImageData = NULL;
  mKernelImageData = NULL;
  mOutputImageData = NULL;
  mWeightImageData = NULL;
}

void OcvFftDeblurrer::operator()(const uint8_t* inputImageData,
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

  // TODO: Weighting
  cv::Mat weightImage(mImageHeight, mImageWidth, CV_32FC1, mWeightImageData);
  weightImage = cv::Scalar(1.0);
  applyDFTWindow(weightImage);
  applyDFTWindow(inputImage);

  // TODO: Debug prints
  cv::imshow("inputImage", inputImage);
  cv::imshow("mKernelImageData", kernelImage);
  cv::imshow("weightImage", weightImage);
  cv::waitKey(0);

  cv::dft(weightImage, weightImage);

  cv::dft(inputImage, inputImage);
  cv::dft(kernelImage, kernelImage);
  invertFftMatrix(kernelImage, epsilon);
  cv::mulSpectrums(inputImage, kernelImage, outputImage,
                   cv::DFT_COMPLEX_OUTPUT);

  // TODO: Weighting
  cv::mulSpectrums(weightImage, kernelImage, weightImage,
                   cv::DFT_COMPLEX_OUTPUT);

  cv::dft(outputImage, outputImage,
          cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

  // TODO: Weighting
  cv::dft(weightImage, weightImage,
          cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
  cv::divide(outputImage, weightImage, outputImage);
  //	cv::divide(cv::Scalar(1.0), weightImage, weightImage);
  //	cv::threshold(weightImage, weightImage, 50.0, 0.0, cv::THRESH_TRUNC);
  //	cv::multiply(outputImage, weightImage, outputImage);

  convertToOutput(mOutputImageData, mImageWidth, mImageHeight, outputImagePpln,
                  outputImageData);
}

void OcvFftDeblurrer::operator()(const uint8_t* inputImageData,
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

  // TODO: Weighting
  //	cv::Mat weightImage(mImageHeight, mImageWidth, CV_32FC1,
  // mWeightImageData);
  cv::Mat weightImage(inputImage.rows, inputImage.cols, CV_32FC1,
                      mWeightImageData);
  weightImage = cv::Scalar(1.0);
  applyDFTWindow(weightImage);
  applyDFTWindow(inputImage);

  // TODO: Debug prints
  cv::imshow("inputImage", inputImage);
  cv::imshow("weightImage", weightImage);
  cv::waitKey(0);

  cv::dft(weightImage, weightImage);

  cv::dft(inputImage, inputImage);
  cv::dft(kernelImage, kernelImage);
  invertFftMatrix(kernelImage, epsilon);
  //	invertFftMatrix(kernelImage);
  cv::mulSpectrums(inputImage, kernelImage, outputImage,
                   cv::DFT_COMPLEX_OUTPUT);

  // TODO: Weighting
  cv::mulSpectrums(weightImage, kernelImage, weightImage,
                   cv::DFT_COMPLEX_OUTPUT);

  cv::dft(outputImage, outputImage,
          cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

  // TODO: Weighting
  cv::dft(weightImage, weightImage,
          cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
  cv::divide(outputImage, weightImage, outputImage);
  //	cv::divide(cv::Scalar(1.0), weightImage, weightImage);
  //	cv::threshold(weightImage, weightImage, 50.0, 0.0, cv::THRESH_TRUNC);
  //	cv::multiply(outputImage, weightImage, outputImage);

  convertToOutput(mOutputImageData, mImageWidth, mImageHeight, outputImagePpln,
                  outputImageData);
}

size_t OcvFftDeblurrer::getMemorySize(int imageWidth, int imageHeight,
                                      [[maybe_unused]] int kernelWidth,
                                      [[maybe_unused]] int kernelHeight) {
  return sizeof(point_value_t) * 3 * imageWidth * imageHeight;
}

void OcvFftDeblurrer::buildKernelForFft(const point_value_t* kernelData,
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

void OcvFftDeblurrer::invertFftMatrix(cv::Mat& matrixFft, float gamma) {
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
      upperLeftValue / (upperLeftValue * upperLeftValue + gamma);

  // set first column
  for (int row = 1; row < matrixHalfHeight; row += 2) {
    realValue = ((float*)(matrixData + row * matrixStep))[0];
    imaginaryValue = ((float*)(matrixData + (row + 1) * matrixStep))[0];
    sum = realValue * realValue + imaginaryValue * imaginaryValue + gamma;
    ((float*)(matrixData + row * matrixStep))[0] = realValue / sum;
    ((float*)(matrixData + (row + 1) * matrixStep))[0] = -imaginaryValue / sum;
  }

  // sets down left if needed
  if (matrixHeight % 2 == 0) {
    float downLeftValue =
        ((float*)(matrixData + (matrixHeight - 1) * matrixStep))[0];
    ((float*)(matrixData + (matrixHeight - 1) * matrixStep))[0] =
        downLeftValue / (downLeftValue * downLeftValue + gamma);
  }

  if (matrixWidth % 2 == 0) {
    // sets upper right
    float upperLeftValue = ((float*)matrixData)[matrixWidth - 1];
    ((float*)matrixData)[matrixWidth - 1] =
        upperLeftValue / (upperLeftValue * upperLeftValue + gamma);

    // set last column
    for (int row = 1; row < matrixHalfHeight; row += 2) {
      realValue = ((float*)(matrixData + row * matrixStep))[matrixWidth - 1];
      imaginaryValue =
          ((float*)(matrixData + (row + 1) * matrixStep))[matrixWidth - 1];
      sum = realValue * realValue + imaginaryValue * imaginaryValue + gamma;
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
          downRightValue / (downRightValue * downRightValue + gamma);
    }
  }

  for (int row = 0; row < matrixHeight; row++) {
    for (int col = 1; col < matrixHalfWidth; col += 2) {
      realValue = ((float*)(matrixData + row * matrixStep))[col];
      imaginaryValue = ((float*)(matrixData + row * matrixStep))[col + 1];
      sum = realValue * realValue + imaginaryValue * imaginaryValue + gamma;
      ((float*)(matrixData + row * matrixStep))[col] = realValue / sum;
      ((float*)(matrixData + row * matrixStep))[col + 1] =
          (-imaginaryValue / sum);
    }
  }
}

void OcvFftDeblurrer::invertFftMatrix(cv::Mat& matrixFft) {
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

void OcvFftDeblurrer::applyDFTWindow(cv::Mat& imageFft) {
  int rows = imageFft.rows;
  int cols = imageFft.cols;
  float* pImageData = (float*)imageFft.data;
  int imagePpln = imageFft.step / sizeof(float);

  for (int row = 0; row < rows; row++) {
    double rowWeight = 0.402 - 0.498 * cos(2 * M_PI * row / (rows - 1)) +
                       0.098 * cos(4 * M_PI * row / (rows - 1)) +
                       0.001 * cos(6 * M_PI * row / (rows - 1));
    for (int col = 0; col < cols; col++) {
      double colWeight = 0.402 - 0.498 * cos(2 * M_PI * col / (cols - 1)) +
                         0.098 * cos(4 * M_PI * col / (cols - 1)) +
                         0.001 * cos(6 * M_PI * col / (cols - 1));
      double pixelValue = (pImageData + row * imagePpln)[col];
      pixelValue *= rowWeight * colWeight;
      // TODO: Restore
      //			(pImageData + row * imagePpln)[col] =
      // pixelValue;
    }
  }
}

void BartlettHannWindow(cv::Mat& imageFft) {
  int rows = imageFft.rows;
  int cols = imageFft.cols;
  float* pImageData = (float*)imageFft.data;
  int imagePpln = imageFft.step / sizeof(float);

  for (int row = 0; row < rows; row++) {
    double rowWeight = 0.62 - 0.48 * fabs(1.0 * row / (rows - 1) - 0.5) -
                       0.38 * cos(2 * M_PI * row / (rows - 1));

    for (int col = 0; col < cols; col++) {
      double colWeight = 0.62 - 0.48 * fabs(1.0 * col / (cols - 1) - 0.5) -
                         0.38 * cos(2 * M_PI * col / (cols - 1));
      double pixelValue = (pImageData + row * imagePpln)[col];
      pixelValue *= rowWeight * colWeight;
      (pImageData + row * imagePpln)[col] = pixelValue;
    }
  }
}

void BlackmanWindow(cv::Mat& imageFft) {
  int rows = imageFft.rows;
  int cols = imageFft.cols;
  float* pImageData = (float*)imageFft.data;
  int imagePpln = imageFft.step / sizeof(float);

  for (int row = 0; row < rows; row++) {
    double rowWeight = 0.42 - 0.5 * cos(2 * M_PI * row / (rows - 1)) +
                       0.08 * cos(4 * M_PI * row / (rows - 1));

    for (int col = 0; col < cols; col++) {
      double colWeight = 0.42 - 0.5 * cos(2 * M_PI * col / (cols - 1)) +
                         0.08 * cos(4 * M_PI * col / (cols - 1));
      double pixelValue = (pImageData + row * imagePpln)[col];
      pixelValue *= rowWeight * colWeight;
      //			(pImageData + row * imagePpln)[col] =
      // pixelValue;
    }
  }
}

void HannWindow(cv::Mat& imageFft) {
  int rows = imageFft.rows;
  int cols = imageFft.cols;
  float* pImageData = (float*)imageFft.data;
  int imagePpln = imageFft.step / sizeof(float);

  for (int row = 0; row < rows; row++) {
    double rowWeight = 0.5 * (1.0 - cos(2 * M_PI * row / (rows - 1)));

    for (int col = 0; col < cols; col++) {
      double colWeight = 0.5 * (1.0 - cos(2 * M_PI * col / (cols - 1)));
      double pixelValue = (pImageData + row * imagePpln)[col];
      pixelValue *= rowWeight * colWeight;
      (pImageData + row * imagePpln)[col] = pixelValue;
    }
  }
}

void BartlettWindow(cv::Mat& imageFft) {
  int rows = imageFft.rows;
  int cols = imageFft.cols;
  float* pImageData = (float*)imageFft.data;
  int imagePpln = imageFft.step / sizeof(float);

  for (int row = 0; row < rows; row++) {
    double rowWeight = 1.0 - fabs(row - (rows - 1) / 2.0) * 2.0 / (rows - 1);

    for (int col = 0; col < cols; col++) {
      double colWeight = 1.0 - fabs(col - (cols - 1) / 2.0) * 2.0 / (cols - 1);
      double pixelValue = (pImageData + row * imagePpln)[col];
      pixelValue *= rowWeight * colWeight;
      (pImageData + row * imagePpln)[col] = pixelValue;
    }
  }
}

void SineWindow(cv::Mat& imageFft) {
  int rows = imageFft.rows;
  int cols = imageFft.cols;
  float* pImageData = (float*)imageFft.data;
  int imagePpln = imageFft.step / sizeof(float);

  for (int row = 0; row < rows; row++) {
    double rowWeight = sin(row * M_PI / (rows - 1));

    for (int col = 0; col < cols; col++) {
      double colWeight = sin(col * M_PI / (cols - 1));
      double pixelValue = (pImageData + row * imagePpln)[col];
      pixelValue *= rowWeight * colWeight;
      (pImageData + row * imagePpln)[col] = pixelValue;
    }
  }
}

void KaiserBesselWindow(cv::Mat& imageFft) {
  int rows = imageFft.rows;
  int cols = imageFft.cols;
  float* pImageData = (float*)imageFft.data;
  int imagePpln = imageFft.step / sizeof(float);

  for (int row = 0; row < rows; row++) {
    double rowWeight = 0.402 - 0.498 * cos(2 * M_PI * row / (rows - 1)) +
                       0.098 * cos(4 * M_PI * row / (rows - 1)) +
                       0.001 * cos(6 * M_PI * row / (rows - 1));
    for (int col = 0; col < cols; col++) {
      double colWeight = 0.402 - 0.498 * cos(2 * M_PI * col / (cols - 1)) +
                         0.098 * cos(4 * M_PI * col / (cols - 1)) +
                         0.001 * cos(6 * M_PI * col / (cols - 1));
      double pixelValue = (pImageData + row * imagePpln)[col];
      pixelValue *= rowWeight * colWeight;
      (pImageData + row * imagePpln)[col] = pixelValue;
    }
  }
}

}  // namespace Deblurring
}  // namespace Test
