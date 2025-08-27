/*
 * LrDeblurrer.cpp
 *
 *  Created on: Jan 23, 2015
 *      Author: vantonov
 */

#include "LrDeblurrer.h"

#include <string.h>

namespace Test {
namespace Deblurring {

LrDeblurrer::LrDeblurrer(int imageWidth, int imageHeight)
    : IterativeDeblurrer(imageWidth, imageHeight),
      mCurrentDeblurredImage(NULL),
      mReblurredImage(NULL),
      mWeightImage(NULL) {
  init(imageWidth, imageHeight, NULL);
}

LrDeblurrer::LrDeblurrer(int imageWidth, int imageHeight, void* pExternalMemory)
    : IterativeDeblurrer(imageWidth, imageHeight, pExternalMemory),
      mCurrentDeblurredImage(NULL),
      mReblurredImage(NULL),
      mWeightImage(NULL) {
  init(imageWidth, imageHeight,
       (void*)((intptr_t)pExternalMemory +
               IterativeDeblurrer::getMemorySize(imageWidth, imageHeight)));
}

LrDeblurrer::~LrDeblurrer() { deinit(); }

void LrDeblurrer::operator()(const uint8_t* inputImageData, int inputImagePpln,
                             const SparseBlurKernel& currentBlurKernel,
                             uint8_t* outputImageData, int outputImagePpln) {
  convertFromInput(inputImageData, mImageWidth, mImageHeight, inputImagePpln,
                   mCurrentDeblurredImage);

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

void LrDeblurrer::operator()(const uint8_t* inputImageData, int inputImagePpln,
                             const point_value_t* kernelData, int kernelWidth,
                             int kernelHeight, int kernelPpln,
                             uint8_t* outputImageData, int outputImagePpln) {
  convertFromInput(inputImageData, mImageWidth, mImageHeight, inputImagePpln,
                   mCurrentDeblurredImage);

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

size_t LrDeblurrer::getMemorySize(int imageWidth, int imageHeight) {
  int requiredMemorySize = 4 * imageWidth * imageHeight * sizeof(point_value_t);

  requiredMemorySize +=
      IterativeDeblurrer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void LrDeblurrer::init(int imageWidth, int imageHeight, void* pExternalMemory) {
  size_t imageBlockSize = imageWidth * imageHeight;

  if (isExternalMemoryUsed) {
    point_value_t* pDataBuffer = (point_value_t*)pExternalMemory;
    mCurrentDeblurredImage = &pDataBuffer[0];
    mReblurredImage = &pDataBuffer[imageBlockSize];
    mBlurredImage = &pDataBuffer[2 * imageBlockSize];
    mWeightImage = &pDataBuffer[3 * imageBlockSize];
  } else {
    mCurrentDeblurredImage = new point_value_t[imageBlockSize];
    mReblurredImage = new point_value_t[imageBlockSize];
    mBlurredImage = new point_value_t[imageBlockSize];
    mWeightImage = new point_value_t[imageBlockSize];
  }
}

void LrDeblurrer::deinit() {
  if (!isExternalMemoryUsed) {
    delete[] mCurrentDeblurredImage;
    delete[] mReblurredImage;
    delete[] mBlurredImage;
    delete[] mWeightImage;
  }

  mCurrentDeblurredImage = NULL;
  mReblurredImage = NULL;
  mBlurredImage = NULL;
  mWeightImage = NULL;
}

void LrDeblurrer::doIterations(const point_value_t* kernelData, int kernelWidth,
                               int kernelHeight, int kernelPpln) {
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
         kernelPpln, mWeightImage, false);

    blur(mWeightImage, kernelData, kernelWidth, kernelHeight, kernelPpln,
         mReblurredImage, true);

    doRegularization();

    divideImages(mBlurredImage, mReblurredImage, mWeightImage);

    // Norm calculation first part
    memcpy(mReblurredImage, mCurrentDeblurredImage, imageBlockSize);

    multiplyImages(mWeightImage, mCurrentDeblurredImage,
                   mCurrentDeblurredImage);

    // Norm calculation second part
    mCurrentNorm =
        calculateL2NormOfDifference(mCurrentDeblurredImage, mReblurredImage) /
        (mImageWidth * mImageHeight);

    //		if (residualNorm < bestNorm) {
    //			bestNorm = residualNorm;
    //			deblurredImage.copyTo(bestRestoredImage);
    //		}

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

void LrDeblurrer::doIterations(const SparseBlurKernel& currentBlurKernel) {
  mProcessStatus = CONVERGENCE_OK;

  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  //	double initialNorm;

  mCurrentIteration = 0;

  do {
    blur(mCurrentDeblurredImage, currentBlurKernel, mWeightImage, false);

    blur(mWeightImage, currentBlurKernel, mReblurredImage, true);

    doRegularization();

    divideImages(mBlurredImage, mReblurredImage, mWeightImage);

    // Norm calculation first part
    memcpy(mReblurredImage, mCurrentDeblurredImage, imageBlockSize);

    multiplyImages(mWeightImage, mCurrentDeblurredImage,
                   mCurrentDeblurredImage);

    // Norm calculation second part
    mCurrentNorm =
        calculateL2NormOfDifference(mCurrentDeblurredImage, mReblurredImage) /
        (mImageWidth * mImageHeight);

    //		if (residualNorm < bestNorm) {
    //			bestNorm = residualNorm;
    //			deblurredImage.copyTo(bestRestoredImage);
    //		}

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

void LrDeblurrer::prepareIterations(const point_value_t* kernelData,
                                    int kernelWidth, int kernelHeight,
                                    int kernelPpln) {
  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  // initial approximation
  blur(mCurrentDeblurredImage, kernelData, kernelWidth, kernelHeight,
       kernelPpln, mBlurredImage, true);
  memset(mReblurredImage, 0, imageBlockSize);
}

void LrDeblurrer::prepareIterations(const SparseBlurKernel& currentBlurKernel) {
  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  // initial approximation
  blur(mCurrentDeblurredImage, currentBlurKernel, mBlurredImage, true);
  memset(mReblurredImage, 0, imageBlockSize);
}

void LrDeblurrer::doRegularization() {
  if (NULL == mRegularizer) {
    return;
  }

  float regularizationWeight = mRegularizer->getRegularizationWeight();

  if (0.0 == regularizationWeight) {
    return;
  }

  // TODO: There is problem with division and noise!

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
                    regularizationWeight, 0.0, mReblurredImage);

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
                    regularizationWeight, 0.0, mReblurredImage);

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
                    regularizationWeight, 0.0, mReblurredImage);
}

}  // namespace Deblurring
}  // namespace Test
