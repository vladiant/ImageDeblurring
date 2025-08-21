/*
 * CgaDeblurrer.cpp
 *
 *  Created on: Jan 8, 2015
 *      Author: vantonov
 */

#include "CgDeblurrer.h"

#include <string.h>

namespace Test {
namespace Deblurring {

CgDeblurrer::CgDeblurrer(int imageWidth, int imageHeight)
    : IterativeDeblurrer(imageWidth, imageHeight),
      mCurrentDeblurredImage(NULL),
      mResidualImage(NULL),
      mPreconditionedImage(NULL),
      mBlurredPreconditionedImage(NULL),
      mDifferenceResidualImage(NULL) {
  init(imageWidth, imageHeight, NULL);
}

CgDeblurrer::CgDeblurrer(int imageWidth, int imageHeight, void* pExternalMemory)
    : IterativeDeblurrer(imageWidth, imageHeight, pExternalMemory),
      mCurrentDeblurredImage(NULL),
      mResidualImage(NULL),
      mPreconditionedImage(NULL),
      mBlurredPreconditionedImage(NULL),
      mDifferenceResidualImage(NULL) {
  init(imageWidth, imageHeight,
       (void*)((intptr_t)pExternalMemory +
               IterativeDeblurrer::getMemorySize(imageWidth, imageHeight)));
}

CgDeblurrer::~CgDeblurrer() { deinit(); }

void CgDeblurrer::operator()(const uint8_t* inputImageData, int inputImagePpln,
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

void CgDeblurrer::operator()(const uint8_t* inputImageData, int inputImagePpln,
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

void CgDeblurrer::operator()(const uint8_t* inputImageData, int inputImagePpln,
                             const GyroBlurKernelBuilder& currentkernelBuilder,
                             uint8_t* outputImageData, int outputImagePpln) {
  convertFromInput(inputImageData, mImageWidth, mImageHeight, inputImagePpln,
                   mCurrentDeblurredImage);

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

void CgDeblurrer::operator()(const uint8_t* inputImageData, int inputImagePpln,
                             const SparseBlurKernel* blurKernels,
                             int blurKernelsCount, uint8_t* outputImageData,
                             int outputImagePpln) {
  convertFromInput(inputImageData, mImageWidth, mImageHeight, inputImagePpln,
                   mCurrentDeblurredImage);

  prepareIterations(blurKernels, blurKernelsCount);

  if (mRegularizer != NULL) {
    mRegularizer->prepareIrls(mCurrentDeblurredImage);
  }

  double normIrls = 1;
  int irlsIteration = 1;
  do {
    doIterations(blurKernels, blurKernelsCount);

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

size_t CgDeblurrer::getMemorySize(int imageWidth, int imageHeight) {
  int requiredMemorySize = 5 * imageWidth * imageHeight * sizeof(point_value_t);

  requiredMemorySize +=
      IterativeDeblurrer::getMemorySize(imageWidth, imageHeight);

  return requiredMemorySize;
}

void CgDeblurrer::init(int imageWidth, int imageHeight, void* pExternalMemory) {
  size_t imageBlockSize = imageWidth * imageHeight;

  if (isExternalMemoryUsed) {
    point_value_t* pDataBuffer = (point_value_t*)pExternalMemory;
    mCurrentDeblurredImage = &pDataBuffer[0];
    mResidualImage = &pDataBuffer[imageBlockSize];
    mPreconditionedImage = &pDataBuffer[2 * imageBlockSize];
    mBlurredPreconditionedImage = &pDataBuffer[3 * imageBlockSize];
    mDifferenceResidualImage = &pDataBuffer[4 * imageBlockSize];
  } else {
    mCurrentDeblurredImage = new point_value_t[imageBlockSize];
    mResidualImage = new point_value_t[imageBlockSize];
    mPreconditionedImage = new point_value_t[imageBlockSize];
    mBlurredPreconditionedImage = new point_value_t[imageBlockSize];
    mDifferenceResidualImage = new point_value_t[imageBlockSize];
  }
}

void CgDeblurrer::deinit() {
  if (!isExternalMemoryUsed) {
    delete[] mCurrentDeblurredImage;
    delete[] mResidualImage;
    delete[] mPreconditionedImage;
    delete[] mBlurredPreconditionedImage;
    delete[] mDifferenceResidualImage;
  }

  mCurrentDeblurredImage = NULL;
  mResidualImage = NULL;
  mPreconditionedImage = NULL;
  mBlurredPreconditionedImage = NULL;
  mDifferenceResidualImage = NULL;
}

void CgDeblurrer::doIterations(const point_value_t* kernelData, int kernelWidth,
                               int kernelHeight, int kernelPpln) {
  mProcessStatus = CONVERGENCE_OK;

  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  double preconditionWeight, updateWeight;
  double bestNorm =
      calculateL2Norm(mResidualImage) / (mImageWidth * mImageHeight);

  mCurrentIteration = 0;

  do {
    // Ap_k
    blur(mPreconditionedImage, kernelData, kernelWidth, kernelHeight,
         kernelPpln, mDifferenceResidualImage, false);
    blur(mDifferenceResidualImage, kernelData, kernelWidth, kernelHeight,
         kernelPpln, mBlurredPreconditionedImage, true);

    doRegularization();

    // beta_k first part
    preconditionWeight =
        calculateDotProductOfImages(mResidualImage, mResidualImage);

    // alpha_k
    updateWeight = calculateDotProductOfImages(mPreconditionedImage,
                                               mBlurredPreconditionedImage);
    if (0 == updateWeight) {
      mProcessStatus = ITERATION_FAILED;
      return;
    }
    updateWeight = preconditionWeight / updateWeight;

    // x_k
    addWeightedImages(mCurrentDeblurredImage, 1.0, mPreconditionedImage,
                      updateWeight, 0.0, mCurrentDeblurredImage);

    // r_k
    memcpy(mDifferenceResidualImage, mResidualImage, imageBlockSize);
    addWeightedImages(mResidualImage, 1.0, mBlurredPreconditionedImage,
                      -updateWeight, 0.0, mResidualImage);
    subtractImages(mResidualImage, mDifferenceResidualImage,
                   mDifferenceResidualImage);

    // norm calculation
    mCurrentNorm =
        calculateL2Norm(mResidualImage) / (mImageWidth * mImageHeight);

    if (mUseBestIteration && mCurrentNorm < bestNorm) {
      bestNorm = mCurrentNorm;
      memcpy(getBestIterationImage(), mCurrentDeblurredImage, imageBlockSize);
    }

    // beta_k second part
    preconditionWeight =
        calculateDotProductOfImages(mResidualImage, mDifferenceResidualImage) /
        preconditionWeight;

    // p_k
    addWeightedImages(mResidualImage, 1.0, mPreconditionedImage,
                      1.0 * preconditionWeight, 0.0, mPreconditionedImage);

    // TODO: Add as debug print!
    //		std::cout << " Iteration: " << mCurrentIteration << " Norm: "
    //				<< mCurrentNorm << " preconditionWeight: " <<
    // preconditionWeight
    //				<< " updateWeight: " << updateWeight <<
    // std::endl;

    mCurrentIteration++;
  } while ((mCurrentNorm > mMinimalNorm) &&
           (mCurrentIteration < mMaxIterations));

  if (mCurrentNorm >= mMinimalNorm) {
    mProcessStatus = MAX_ITERATIONS_REACHED;
  }

  if (mUseBestIteration) {
    mCurrentNorm = bestNorm;
    memcpy(mCurrentDeblurredImage, getBestIterationImage(), imageBlockSize);
  }

  return;
}

void CgDeblurrer::doIterations(const SparseBlurKernel& currentBlurKernel) {
  mProcessStatus = CONVERGENCE_OK;

  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  double preconditionWeight, updateWeight;
  double bestNorm =
      calculateL2Norm(mResidualImage) / (mImageWidth * mImageHeight);

  mCurrentIteration = 0;

  do {
    // Ap_k
    blur(mPreconditionedImage, currentBlurKernel, mDifferenceResidualImage,
         false);
    blur(mDifferenceResidualImage, currentBlurKernel,
         mBlurredPreconditionedImage, true);

    doRegularization();

    // beta_k first part
    preconditionWeight =
        calculateDotProductOfImages(mResidualImage, mResidualImage);

    // alpha_k
    updateWeight = calculateDotProductOfImages(mPreconditionedImage,
                                               mBlurredPreconditionedImage);
    if (0 == updateWeight) {
      mProcessStatus = ITERATION_FAILED;
      return;
    }
    updateWeight = preconditionWeight / updateWeight;

    // x_k
    addWeightedImages(mCurrentDeblurredImage, 1.0, mPreconditionedImage,
                      updateWeight, 0.0, mCurrentDeblurredImage);

    // r_k
    memcpy(mDifferenceResidualImage, mResidualImage, imageBlockSize);
    addWeightedImages(mResidualImage, 1.0, mBlurredPreconditionedImage,
                      -updateWeight, 0.0, mResidualImage);
    subtractImages(mResidualImage, mDifferenceResidualImage,
                   mDifferenceResidualImage);

    // norm calculation
    mCurrentNorm =
        calculateL2Norm(mResidualImage) / (mImageWidth * mImageHeight);

    if (mUseBestIteration && mCurrentNorm < bestNorm) {
      bestNorm = mCurrentNorm;
      memcpy(getBestIterationImage(), mCurrentDeblurredImage, imageBlockSize);
    }

    // beta_k second part
    preconditionWeight =
        calculateDotProductOfImages(mResidualImage, mDifferenceResidualImage) /
        preconditionWeight;

    // p_k
    addWeightedImages(mResidualImage, 1.0, mPreconditionedImage,
                      1.0 * preconditionWeight, 0.0, mPreconditionedImage);

    // TODO: Add as debug print!
    //		std::cout << " Iteration: " << mCurrentIteration << " Norm: "
    //				<< mCurrentNorm << " preconditionWeight: " <<
    // preconditionWeight
    //				<< " updateWeight: " << updateWeight <<
    // std::endl;

    mCurrentIteration++;
  } while ((mCurrentNorm > mMinimalNorm) &&
           (mCurrentIteration < mMaxIterations));

  if (mCurrentNorm >= mMinimalNorm) {
    mProcessStatus = MAX_ITERATIONS_REACHED;
  }

  if (mUseBestIteration) {
    mCurrentNorm = bestNorm;
    memcpy(mCurrentDeblurredImage, getBestIterationImage(), imageBlockSize);
  }

  return;
}

void CgDeblurrer::doIterations(
    const GyroBlurKernelBuilder& currentkernelBuilder) {
  mProcessStatus = CONVERGENCE_OK;

  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  double preconditionWeight, updateWeight;
  double bestNorm =
      calculateL2Norm(mResidualImage) / (mImageWidth * mImageHeight);

  mCurrentIteration = 0;

  do {
    // Ap_k
    blur(mPreconditionedImage, currentkernelBuilder, mDifferenceResidualImage,
         false);
    blur(mDifferenceResidualImage, currentkernelBuilder,
         mBlurredPreconditionedImage, true);

    doRegularization();

    // beta_k first part
    preconditionWeight =
        calculateDotProductOfImages(mResidualImage, mResidualImage);

    // alpha_k
    updateWeight = calculateDotProductOfImages(mPreconditionedImage,
                                               mBlurredPreconditionedImage);
    if (0 == updateWeight) {
      mProcessStatus = ITERATION_FAILED;
      return;
    }
    updateWeight = preconditionWeight / updateWeight;

    // x_k
    addWeightedImages(mCurrentDeblurredImage, 1.0, mPreconditionedImage,
                      updateWeight, 0.0, mCurrentDeblurredImage);

    // r_k
    memcpy(mDifferenceResidualImage, mResidualImage, imageBlockSize);
    addWeightedImages(mResidualImage, 1.0, mBlurredPreconditionedImage,
                      -updateWeight, 0.0, mResidualImage);
    subtractImages(mResidualImage, mDifferenceResidualImage,
                   mDifferenceResidualImage);

    // norm calculation
    mCurrentNorm =
        calculateL2Norm(mResidualImage) / (mImageWidth * mImageHeight);

    if (mUseBestIteration && mCurrentNorm < bestNorm) {
      bestNorm = mCurrentNorm;
      memcpy(getBestIterationImage(), mCurrentDeblurredImage, imageBlockSize);
    }

    // beta_k second part
    preconditionWeight =
        calculateDotProductOfImages(mResidualImage, mDifferenceResidualImage) /
        preconditionWeight;

    // p_k
    addWeightedImages(mResidualImage, 1.0, mPreconditionedImage,
                      1.0 * preconditionWeight, 0.0, mPreconditionedImage);

    // TODO: Add as debug print!
    //		std::cout << " Iteration: " << mCurrentIteration << " Norm: "
    //				<< mCurrentNorm << " preconditionWeight: " <<
    // preconditionWeight
    //				<< " updateWeight: " << updateWeight <<
    // std::endl;

    mCurrentIteration++;
  } while ((mCurrentNorm > mMinimalNorm) &&
           (mCurrentIteration < mMaxIterations));

  if (mCurrentNorm >= mMinimalNorm) {
    mProcessStatus = MAX_ITERATIONS_REACHED;
  }

  if (mUseBestIteration) {
    mCurrentNorm = bestNorm;
    memcpy(mCurrentDeblurredImage, getBestIterationImage(), imageBlockSize);
  }

  return;
}

void CgDeblurrer::doIterations(const SparseBlurKernel* blurKernels,
                               int blurKernelsCount) {
  mProcessStatus = CONVERGENCE_OK;

  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  double preconditionWeight, updateWeight;
  double bestNorm =
      calculateL2Norm(mResidualImage) / (mImageWidth * mImageHeight);

  mCurrentIteration = 0;

  do {
    // Ap_k
    blur(mPreconditionedImage, blurKernels, blurKernelsCount,
         mDifferenceResidualImage, false);
    blur(mDifferenceResidualImage, blurKernels, blurKernelsCount,
         mBlurredPreconditionedImage, true);

    doRegularization();

    // beta_k first part
    preconditionWeight =
        calculateDotProductOfImages(mResidualImage, mResidualImage);

    // alpha_k
    updateWeight = calculateDotProductOfImages(mPreconditionedImage,
                                               mBlurredPreconditionedImage);
    if (0 == updateWeight) {
      mProcessStatus = ITERATION_FAILED;
      return;
    }
    updateWeight = preconditionWeight / updateWeight;

    // x_k
    addWeightedImages(mCurrentDeblurredImage, 1.0, mPreconditionedImage,
                      updateWeight, 0.0, mCurrentDeblurredImage);

    // r_k
    memcpy(mDifferenceResidualImage, mResidualImage, imageBlockSize);
    addWeightedImages(mResidualImage, 1.0, mBlurredPreconditionedImage,
                      -updateWeight, 0.0, mResidualImage);
    subtractImages(mResidualImage, mDifferenceResidualImage,
                   mDifferenceResidualImage);

    // norm calculation
    mCurrentNorm =
        calculateL2Norm(mResidualImage) / (mImageWidth * mImageHeight);

    if (mUseBestIteration && mCurrentNorm < bestNorm) {
      bestNorm = mCurrentNorm;
      memcpy(getBestIterationImage(), mCurrentDeblurredImage, imageBlockSize);
    }

    // beta_k second part
    preconditionWeight =
        calculateDotProductOfImages(mResidualImage, mDifferenceResidualImage) /
        preconditionWeight;

    // p_k
    addWeightedImages(mResidualImage, 1.0, mPreconditionedImage,
                      1.0 * preconditionWeight, 0.0, mPreconditionedImage);

    // TODO: Add as debug print!
    //		std::cout << " Iteration: " << mCurrentIteration << " Norm: "
    //				<< mCurrentNorm << " preconditionWeight: " <<
    // preconditionWeight
    //				<< " updateWeight: " << updateWeight <<
    // std::endl;

    mCurrentIteration++;
  } while ((mCurrentNorm > mMinimalNorm) &&
           (mCurrentIteration < mMaxIterations));

  if (mCurrentNorm >= mMinimalNorm) {
    mProcessStatus = MAX_ITERATIONS_REACHED;
  }

  if (mUseBestIteration) {
    mCurrentNorm = bestNorm;
    memcpy(mCurrentDeblurredImage, getBestIterationImage(), imageBlockSize);
  }

  return;
}

void CgDeblurrer::prepareIterations(const point_value_t* kernelData,
                                    int kernelWidth, int kernelHeight,
                                    int kernelPpln) {
  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  // initial approximation of the residual
  memcpy(mResidualImage, mCurrentDeblurredImage, imageBlockSize);
  blur(mCurrentDeblurredImage, kernelData, kernelWidth, kernelHeight,
       kernelPpln, mBlurredPreconditionedImage, false);
  subtractImages(mResidualImage, mBlurredPreconditionedImage,
                 mDifferenceResidualImage);
  blur(mDifferenceResidualImage, kernelData, kernelWidth, kernelHeight,
       kernelPpln, mResidualImage, true);

  memset(mDifferenceResidualImage, 0, imageBlockSize);

  // initial approximation of preconditioner
  memcpy(mPreconditionedImage, mResidualImage, imageBlockSize);

  // initial approximation of preconditioned blurred image
  memset(mBlurredPreconditionedImage, 0, imageBlockSize);
}

void CgDeblurrer::prepareIterations(const SparseBlurKernel& currentBlurKernel) {
  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  // initial approximation of the residual
  memcpy(mResidualImage, mCurrentDeblurredImage, imageBlockSize);
  blur(mCurrentDeblurredImage, currentBlurKernel, mBlurredPreconditionedImage,
       false);
  subtractImages(mResidualImage, mBlurredPreconditionedImage,
                 mDifferenceResidualImage);
  blur(mDifferenceResidualImage, currentBlurKernel, mResidualImage, true);

  memset(mDifferenceResidualImage, 0, imageBlockSize);

  // initial approximation of preconditioner
  memcpy(mPreconditionedImage, mResidualImage, imageBlockSize);

  // initial approximation of preconditioned blurred image
  memset(mBlurredPreconditionedImage, 0, imageBlockSize);
}

void CgDeblurrer::prepareIterations(
    const GyroBlurKernelBuilder& currentkernelBuilder) {
  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  // initial approximation of the residual
  memcpy(mResidualImage, mCurrentDeblurredImage, imageBlockSize);
  blur(mCurrentDeblurredImage, currentkernelBuilder,
       mBlurredPreconditionedImage, false);
  subtractImages(mResidualImage, mBlurredPreconditionedImage,
                 mDifferenceResidualImage);
  blur(mDifferenceResidualImage, currentkernelBuilder, mResidualImage, true);

  memset(mDifferenceResidualImage, 0, imageBlockSize);

  // initial approximation of preconditioner
  memcpy(mPreconditionedImage, mResidualImage, imageBlockSize);

  // initial approximation of preconditioned blurred image
  memset(mBlurredPreconditionedImage, 0, imageBlockSize);
}

void CgDeblurrer::prepareIterations(const SparseBlurKernel* blurKernels,
                                    int blurKernelsCount) {
  size_t imageBlockSize = mImageWidth * mImageHeight * sizeof(point_value_t);

  if (0 == imageBlockSize) {
    mProcessStatus = NOT_INITIALIZED;
    return;
  }

  // initial approximation of the residual
  memcpy(mResidualImage, mCurrentDeblurredImage, imageBlockSize);
  blur(mCurrentDeblurredImage, blurKernels, blurKernelsCount,
       mBlurredPreconditionedImage, false);
  subtractImages(mResidualImage, mBlurredPreconditionedImage,
                 mDifferenceResidualImage);
  blur(mDifferenceResidualImage, blurKernels, blurKernelsCount, mResidualImage,
       true);

  memset(mDifferenceResidualImage, 0, imageBlockSize);

  // initial approximation of preconditioner
  memcpy(mPreconditionedImage, mResidualImage, imageBlockSize);

  // initial approximation of preconditioned blurred image
  memset(mBlurredPreconditionedImage, 0, imageBlockSize);
}

void CgDeblurrer::doRegularization() {
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

  mRegularizer->calculateRegularization(mPreconditionedImage,
                                        regularizationImage, false);
  mRegularizer->calculateRegularization(regularizationImage,
                                        regularizationImageTransposed, true);

  if (irlsWeights != NULL) {
    multiplyImages(regularizationImageTransposed, irlsWeights,
                   regularizationImageTransposed);
  }

  addWeightedImages(mBlurredPreconditionedImage, 1.0,
                    regularizationImageTransposed, regularizationWeight, 0.0,
                    mBlurredPreconditionedImage);

  mRegularizer->calculateRegularizationX(mPreconditionedImage,
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

  addWeightedImages(mBlurredPreconditionedImage, 1.0,
                    regularizationImageTransposed, regularizationWeight, 0.0,
                    mBlurredPreconditionedImage);

  mRegularizer->calculateRegularizationY(mPreconditionedImage,
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

  addWeightedImages(mBlurredPreconditionedImage, 1.0,
                    regularizationImageTransposed, regularizationWeight, 0.0,
                    mBlurredPreconditionedImage);
}

}  // namespace Deblurring
}  // namespace Test
