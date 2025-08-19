/*
 * vanCittertDeblurrer.cpp
 *
 *  Created on: Jan 23, 2015
 *      Author: vantonov
 */

#include "vanCittertDeblurrer.h"

#include <string.h>

namespace Test {
namespace Deblurring {

vanCittertDeblurrer::vanCittertDeblurrer(int imageWidth, int imageHeight)
    : IterativeDeblurrer(imageWidth, imageHeight),
      mCurrentDeblurredImage(NULL),
      mResidualImage(NULL),
      mReblurredImage(NULL),
      mBlurredImage(NULL),
      mBeta(1.0) {
  init(imageWidth, imageHeight, NULL);
}

vanCittertDeblurrer::vanCittertDeblurrer(int imageWidth, int imageHeight,
                                         void* pExternalMemory)
    : IterativeDeblurrer(imageWidth, imageHeight, pExternalMemory),
      mCurrentDeblurredImage(NULL),
      mResidualImage(NULL),
      mReblurredImage(NULL),
      mBlurredImage(NULL),
      mBeta(1.0) {
  init(imageWidth, imageHeight,
       (void*)((intptr_t)pExternalMemory +
               IterativeDeblurrer::getMemorySize(imageWidth, imageHeight)));
}

vanCittertDeblurrer::~vanCittertDeblurrer() { deinit(); }

void vanCittertDeblurrer::operator()(const uint8_t* inputImageData,
                                     int inputImagePpln,
                                     const SparseBlurKernel& currentBlurKernel,
                                     uint8_t* outputImageData,
                                     int outputImagePpln) {
  convertFromInput(inputImageData, mImageWidth, mImageHeight, inputImagePpln,
                   mBlurredImage);

  prepareIterations(currentBlurKernel);

  if (mRegularizer != NULL) {
    mRegularizer->prepareIrls(mCurrentDeblurredImage);
  }

  double normIrls = 1;
  int irlsIteration = 1;
  do {
    doIterations(currentBlurKernel);

    if (mRegularizer != NULL && mRegularizer->isIrlsUsed() &&
        irlsIteration < mRegularizer->getMaxIrlsIterations()) {
      mRegularizer->calculateIrlsWeights(mCurrentDeblurredImage);
      mRegularizer->calculateIrlsWeightsX(mCurrentDeblurredImage);
      mRegularizer->calculateIrlsWeightsY(mCurrentDeblurredImage);

      normIrls =
          calculateL2NormOfDifference(mCurrentDeblurredImage,
                                      mRegularizer->getOldDeblurredImage()) /
          (mImageWidth * mImageHeight);

      // TODO: Debug print
      //			std::cout << "Iteration IRLS: " << irlsIteration
      //<< "  norm IRLS: "
      //					<< normIrls << std::endl;

      if (normIrls < mRegularizer->getMinimalNormIrls()) {
        break;
      }

      mRegularizer->setOldDeblurredImage(mCurrentDeblurredImage);

    } else {
      break;
    }

    irlsIteration++;
  } while (true);

  convertToOutput(mCurrentDeblurredImage, mImageWidth, mImageHeight,
                  outputImagePpln, outputImageData);
}

void vanCittertDeblurrer::operator()(const uint8_t* inputImageData,
                                     int inputImagePpln,
                                     const point_value_t* kernelData,
                                     int kernelWidth, int kernelHeight,
                                     int kernelPpln, uint8_t* outputImageData,
                                     int outputImagePpln) {
  convertFromInput(inputImageData, mImageWidth, mImageHeight, inputImagePpln,
                   mBlurredImage);

  prepareIterations(kernelData, kernelWidth, kernelHeight, kernelPpln);

  if (mRegularizer != NULL) {
    mRegularizer->prepareIrls(mCurrentDeblurredImage);
  }

  double normIrls = 1;
  int irlsIteration = 1;
  do {
    doIterations(kernelData, kernelWidth, kernelHeight, kernelPpln);

    if (mRegularizer != NULL && mRegularizer->isIrlsUsed() &&
        irlsIteration < mRegularizer->getMaxIrlsIterations()) {
      mRegularizer->calculateIrlsWeights(mCurrentDeblurredImage);
      mRegularizer->calculateIrlsWeightsX(mCurrentDeblurredImage);
      mRegularizer->calculateIrlsWeightsY(mCurrentDeblurredImage);

      normIrls =
          calculateL2NormOfDifference(mCurrentDeblurredImage,
                                      mRegularizer->getOldDeblurredImage()) /
          (mImageWidth * mImageHeight);

      // TODO: Debug print
      //			std::cout << "Iteration IRLS: " << irlsIteration
      //<< "  norm IRLS: "
      //					<< normIrls << std::endl;

      if (normIrls < mRegularizer->getMinimalNormIrls()) {
        break;
      }

      mRegularizer->setOldDeblurredImage(mCurrentDeblurredImage);

    } else {
      break;
    }

    irlsIteration++;
  } while (true);

  convertToOutput(mCurrentDeblurredImage, mImageWidth, mImageHeight,
                  outputImagePpln, outputImageData);
}

void vanCittertDeblurrer::operator()(
    const uint8_t* inputImageData, int inputImagePpln,
    const GyroBlurKernelBuilder& currentkernelBuilder, uint8_t* outputImageData,
    int outputImagePpln) {
  convertFromInput(inputImageData, mImageWidth, mImageHeight, inputImagePpln,
                   mBlurredImage);

  prepareIterations(currentkernelBuilder);

  if (mRegularizer != NULL) {
    mRegularizer->prepareIrls(mCurrentDeblurredImage);
  }

  double normIrls = 1;
  int irlsIteration = 1;
  do {
    doIterations(currentkernelBuilder);

    if (mRegularizer != NULL && mRegularizer->isIrlsUsed() &&
        irlsIteration < mRegularizer->getMaxIrlsIterations()) {
      mRegularizer->calculateIrlsWeights(mCurrentDeblurredImage);
      mRegularizer->calculateIrlsWeightsX(mCurrentDeblurredImage);
      mRegularizer->calculateIrlsWeightsY(mCurrentDeblurredImage);

      normIrls =
          calculateL2NormOfDifference(mCurrentDeblurredImage,
                                      mRegularizer->getOldDeblurredImage()) /
          (mImageWidth * mImageHeight);

      // TODO: Debug print
      //			std::cout << "Iteration IRLS: " << irlsIteration
      //<< "  norm IRLS: "
      //					<< normIrls << std::endl;

      if (normIrls < mRegularizer->getMinimalNormIrls()) {
        break;
      }

      mRegularizer->setOldDeblurredImage(mCurrentDeblurredImage);

    } else {
      break;
    }

    irlsIteration++;
  } while (true);

  convertToOutput(mCurrentDeblurredImage, mImageWidth, mImageHeight,
                  outputImagePpln, outputImageData);
}

size_t vanCittertDeblurrer::getMemorySize(int imageWidth, int imageHeight) {
  int requiredMemorySize = 4 * imageWidth * imageHeight * sizeof(point_value_t);

  requiredMemorySize +=
      IterativeDeblurrer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void vanCittertDeblurrer::init(int imageWidth, int imageHeight,
                               void* pExternalMemory) {
  size_t imageBlockSize = imageWidth * imageHeight;

  if (isExternalMemoryUsed) {
    point_value_t* pDataBuffer = (point_value_t*)pExternalMemory;

    mCurrentDeblurredImage = &pDataBuffer[0];
    mResidualImage = &pDataBuffer[imageBlockSize];
    mReblurredImage = &pDataBuffer[2 * imageBlockSize];
    mBlurredImage = &pDataBuffer[3 * imageBlockSize];
  } else {
    mCurrentDeblurredImage = new point_value_t[imageBlockSize];
    mResidualImage = new point_value_t[imageBlockSize];
    mReblurredImage = new point_value_t[imageBlockSize];
    mBlurredImage = new point_value_t[imageBlockSize];
  }
}

void vanCittertDeblurrer::deinit() {
  if (!isExternalMemoryUsed) {
    delete[] mCurrentDeblurredImage;
    delete[] mResidualImage;
    delete[] mReblurredImage;
    delete[] mBlurredImage;
  }

  mCurrentDeblurredImage = NULL;
  mResidualImage = NULL;
  mReblurredImage = NULL;
  mBlurredImage = NULL;
}

void vanCittertDeblurrer::doIterations(const point_value_t* kernelData,
                                       int kernelWidth, int kernelHeight,
                                       int kernelPpln) {
  mProcessStatus = CONVERGENCE_OK;

  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  //	double initialNorm;

  mCurrentIteration = 0;

  do {
    blur(mCurrentDeblurredImage, kernelData, kernelWidth, kernelHeight,
         kernelPpln, mReblurredImage, false);

    subtractImages(mBlurredImage, mReblurredImage, mResidualImage);

    //		if (residualNorm < bestNorm) {
    //			bestNorm = residualNorm;
    //			deblurredImage.copyTo(bestRestoredImage);
    //		}

    blur(mResidualImage, kernelData, kernelWidth, kernelHeight, kernelPpln,
         mReblurredImage, true);

    doRegularization();

    mCurrentNorm =
        calculateL2Norm(mReblurredImage) / (mImageWidth * mImageHeight);

    addWeightedImages(mCurrentDeblurredImage, 1, mReblurredImage, mBeta, 0,
                      mCurrentDeblurredImage);

    // TODO: Add as debug print!
    //		std::cout << " Iteration: " << mCurrentIteration << " Norm: "
    //				<< mCurrentNorm << std::endl;

    mCurrentIteration++;
  } while ((mCurrentNorm > mMinimalNorm) &&
           (mCurrentIteration < mMaxIterations));

  if (mCurrentNorm >= mMinimalNorm) {
    mProcessStatus = MAX_ITERATIONS_REACHED;
  }

  return;
}

void vanCittertDeblurrer::doIterations(
    const SparseBlurKernel& currentBlurKernel) {
  mProcessStatus = CONVERGENCE_OK;

  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  //	double initialNorm;

  mCurrentIteration = 0;

  do {
    blur(mCurrentDeblurredImage, currentBlurKernel, mReblurredImage, false);

    subtractImages(mBlurredImage, mReblurredImage, mResidualImage);

    //		if (residualNorm < bestNorm) {
    //			bestNorm = residualNorm;
    //			deblurredImage.copyTo(bestRestoredImage);
    //		}

    blur(mResidualImage, currentBlurKernel, mReblurredImage, true);

    doRegularization();

    mCurrentNorm =
        calculateL2Norm(mReblurredImage) / (mImageWidth * mImageHeight);

    addWeightedImages(mCurrentDeblurredImage, 1, mReblurredImage, mBeta, 0,
                      mCurrentDeblurredImage);

    // TODO: Add as debug print!
    //		std::cout << " Iteration: " << mCurrentIteration << " Norm: "
    //				<< mCurrentNorm << std::endl;

    mCurrentIteration++;
  } while ((mCurrentNorm > mMinimalNorm) &&
           (mCurrentIteration < mMaxIterations));

  if (mCurrentNorm >= mMinimalNorm) {
    mProcessStatus = MAX_ITERATIONS_REACHED;
  }

  return;
}

void vanCittertDeblurrer::doIterations(
    const GyroBlurKernelBuilder& currentkernelBuilder) {
  mProcessStatus = CONVERGENCE_OK;

  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  //	double initialNorm;

  mCurrentIteration = 0;

  do {
    blur(mCurrentDeblurredImage, currentkernelBuilder, mReblurredImage, false);

    subtractImages(mBlurredImage, mReblurredImage, mResidualImage);

    //		if (residualNorm < bestNorm) {
    //			bestNorm = residualNorm;
    //			deblurredImage.copyTo(bestRestoredImage);
    //		}

    blur(mResidualImage, currentkernelBuilder, mReblurredImage, true);

    doRegularization();

    mCurrentNorm =
        calculateL2Norm(mReblurredImage) / (mImageWidth * mImageHeight);

    addWeightedImages(mCurrentDeblurredImage, 1, mReblurredImage, mBeta, 0,
                      mCurrentDeblurredImage);

    // TODO: Add as debug print!
    //		std::cout << " Iteration: " << mCurrentIteration << " Norm: "
    //				<< mCurrentNorm << std::endl;

    mCurrentIteration++;
  } while ((mCurrentNorm > mMinimalNorm) &&
           (mCurrentIteration < mMaxIterations));

  if (mCurrentNorm >= mMinimalNorm) {
    mProcessStatus = MAX_ITERATIONS_REACHED;
  }

  return;
}

void vanCittertDeblurrer::prepareIterations(const point_value_t* kernelData,
                                            int kernelWidth, int kernelHeight,
                                            int kernelPpln) {
  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  // initial approximation
  memcpy(mCurrentDeblurredImage, mBlurredImage, imageBlockSize);
  memset(mResidualImage, 0, imageBlockSize);
  memset(mReblurredImage, 0, imageBlockSize);
}

void vanCittertDeblurrer::prepareIterations(
    const SparseBlurKernel& currentBlurKernel) {
  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  // initial approximation
  memcpy(mCurrentDeblurredImage, mBlurredImage, imageBlockSize);
  memset(mResidualImage, 0, imageBlockSize);
  memset(mReblurredImage, 0, imageBlockSize);
}

void vanCittertDeblurrer::prepareIterations(
    const GyroBlurKernelBuilder& currentkernelBuilder) {
  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  // initial approximation
  memcpy(mCurrentDeblurredImage, mBlurredImage, imageBlockSize);
  memset(mResidualImage, 0, imageBlockSize);
  memset(mReblurredImage, 0, imageBlockSize);
}

void vanCittertDeblurrer::doRegularization() {
  if (NULL == mRegularizer) {
    return;
  }

  float regularizationWeight = mRegularizer->getRegularizationWeight();

  if (0.0 == regularizationWeight) {
    return;
  }

  point_value_t* regularizationImage = mRegularizer->getRegularizationImage();
  point_value_t* regularizationImageTransposed =
      mRegularizer->getRegularizationImageTransposed();

  point_value_t* irlsWeights = mRegularizer->getIrlsWeights();
  point_value_t* irlsWeightsX = mRegularizer->getIrlsWeightsX();
  Test::Deblurring::point_value_t* irlsWeightsY =
      mRegularizer->getIrlsWeightsY();

  mRegularizer->calculateRegularization(mCurrentDeblurredImage,
                                        regularizationImage, false);
  mRegularizer->calculateRegularization(regularizationImage,
                                        regularizationImageTransposed, true);

  if (irlsWeights != NULL) {
    multiplyImages(regularizationImageTransposed, irlsWeights,
                   regularizationImageTransposed);
  }

  addWeightedImages(mReblurredImage, 1.0, regularizationImageTransposed,
                    -regularizationWeight, 0.0, mReblurredImage);

  mRegularizer->calculateRegularizationX(mCurrentDeblurredImage,
                                         regularizationImage, false);
  mRegularizer->calculateRegularizationX(regularizationImage,
                                         regularizationImageTransposed, true);

  if (irlsWeights != NULL) {
    multiplyImages(regularizationImageTransposed, irlsWeights,
                   regularizationImageTransposed);
  }

  if (irlsWeightsX != NULL) {
    multiplyImages(regularizationImageTransposed, irlsWeightsX,
                   regularizationImageTransposed);
  }

  addWeightedImages(mReblurredImage, 1.0, regularizationImageTransposed,
                    -regularizationWeight, 0.0, mReblurredImage);

  mRegularizer->calculateRegularizationY(mCurrentDeblurredImage,
                                         regularizationImage, false);
  mRegularizer->calculateRegularizationY(regularizationImage,
                                         regularizationImageTransposed, true);

  if (irlsWeights != NULL) {
    multiplyImages(regularizationImageTransposed, irlsWeights,
                   regularizationImageTransposed);
  }

  if (irlsWeightsY != NULL) {
    multiplyImages(regularizationImageTransposed, irlsWeightsY,
                   regularizationImageTransposed);
  }

  addWeightedImages(mReblurredImage, 1.0, regularizationImageTransposed,
                    -regularizationWeight, 0.0, mReblurredImage);
}

}  // namespace Deblurring
}  // namespace Test
