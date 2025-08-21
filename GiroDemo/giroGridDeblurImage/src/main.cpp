/*
 * main.cpp
 *
 *  Created on: Jul 18, 2014
 *      Author: vladiant
 */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "BlurKernelUtils.h"
#include "CgDeblurrer.h"
#include "CutOffFftDeblurrer.h"
#include "GradientFftRegularizer.h"
#include "GradientsIterationRegularizer.h"
#include "GyroBlurKernelBuilder.h"
#include "GyroBlurParams.h"
#include "GyroDataReader.h"
#include "HLFftRegularizer.h"
#include "HLIterationRegularizer.h"
#include "HLTwoThirdsFftRegularizer.h"
#include "InverseFftDeblurrer.h"
#include "LaplaceFftRegularizer.h"
#include "LaplaceIterationRegularizer.h"
#include "LrDeblurrer.h"
#include "LsFftDeblurrer.h"
#include "MaskLaplaceIterationRegularizer.h"
#include "SparseBlurKernel.h"
#include "TmFftRegularizer.h"
#include "TmIterationRegularizer.h"
#include "TvFftRegularizer.h"
#include "TvIterationRegularizer.h"
#include "vanCittertDeblurrer.h"

int main(int argc, char* argv[]) {
  // *** Sample calibration data *** //
  Test::Math::Versor<float> gyroSpaceToCameraSpaceQuaternion =
      Test::Math::Versor<float>(-0.0470602, 0.698666, 0.71093, -0.0650423);
  bool isZAxisInvertedInGyroSpace = true;

  // Field of view for intrinsic camera matrix K
  const float fovX = 2657.2 / 2952;  // Experimentally measured
  const float fovY = 2637.0 / 1944;

  // Delay time.
  Test::Time frameTimestampDelayMinusGyroTimestampDelay =
      -3.78381e+06 - Test::timeFromSeconds<float>(0.0019);
  Test::Time delayTime = frameTimestampDelayMinusGyroTimestampDelay;

  // Exposure time
  Test::Time exposureDuration =
      Test::timeFromSeconds<float>(0.04999);  // 0.04999

  // Time for readout in seconds
  const Test::Time readoutTime = Test::timeFromSeconds<float>(0.03283);

  // *** End of sample calibration data *** //

  // Reading gyro data from JPEG file
  const int numberGyroSamples = 200;
  const char* IMAGE_FILENAME = "giro_data_image.jpg";

  std::string imageFilename;
  if (argc > 1) {
    imageFilename = argv[1];
  } else {
    imageFilename = IMAGE_FILENAME;
  }

  Test::Deblurring::GyroDataReader gyroDataReader(numberGyroSamples);

  std::cout << "Reading data from image file: " << imageFilename << std::endl;

  gyroDataReader.readDataFromJPEGFile(imageFilename);

  if (gyroDataReader.getStatus() != gyroDataReader.GYRO_DATA_READER_INIT_OK) {
    std::cerr << "Error reading gyro data from file " << imageFilename << " !"
              << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Correcting angular velocities. " << std::endl;

  gyroDataReader.correctAngularVelocities(gyroSpaceToCameraSpaceQuaternion,
                                          isZAxisInvertedInGyroSpace);

  std::cout << "Reading capture timestamp. " << std::endl;

  Test::Time captureTimeStamp = gyroDataReader.getCaptureTimestamp();

  std::cout << "Capture timestamp: " << captureTimeStamp << std::endl;

  // Load image and print sampling points.
  cv::Mat initialImage = cv::imread(/*"blurred.bmp"*/ imageFilename);

  if (initialImage.empty()) {
    std::cerr << "Error reading image data from file " << imageFilename << " !"
              << std::endl;
    return EXIT_FAILURE;
  }

  cv::namedWindow(imageFilename, cv::WINDOW_NORMAL);
  cv::imshow(imageFilename, initialImage);

  int imageWidth = initialImage.cols;
  int imageHeight = initialImage.rows;
  int imageChannels = initialImage.channels();

  // Set parameters of the gyro blur kernel.
  Test::Deblurring::GyroBlurParams gyroBlurParams;
  gyroBlurParams.setCameraParams(imageWidth, imageHeight, fovX * imageWidth,
                                 fovY * imageHeight);
  gyroBlurParams.setTimeParams(captureTimeStamp, exposureDuration, readoutTime,
                               delayTime);

  // Read versors.
  std::vector<Test::OpticalFlow::TimeOrientation<Test::Math::Versor<float> > >
      samples;
  gyroDataReader.getAngularValuesInInterval(gyroBlurParams.getTimeFirstSample(),
                                            gyroBlurParams.getTimeLastSample(),
                                            samples);

  // Show the versors.
  std::cout << "FirstSample: " << gyroBlurParams.getTimeFirstSample()
            << std::endl;
  std::cout << "LastSample: " << gyroBlurParams.getTimeLastSample()
            << std::endl;
  for (size_t i = 0; i < samples.size(); i++) {
    std::cout << i << " : " << samples[i].time << " : "
              << samples[i].orientation << std::endl;
  }

  // Blur kernel builder.
  Test::Deblurring::GyroBlurKernelBuilder kernelBuilder(gyroBlurParams,
                                                        samples);

  int gridStepX;
  int gridStepY;
  float maxXCoordsSpan;
  float maxYCoordsSpan;
  Test::Deblurring::calculateKernelGridProperties(
      kernelBuilder, gridStepX, gridStepY, maxXCoordsSpan, maxYCoordsSpan);

  // Print blur kernel properties.
  std::cout << "gridStepX= " << gridStepX << "  gridStepY= " << gridStepY
            << std::endl;
  std::cout << "maxXCoordsSpan: " << round(maxXCoordsSpan)
            << "  maxYCoordsSpan: " << round(maxYCoordsSpan) << std::endl;

  // TODO: Routine to set the image patches for processing.
  const float PATCH_SCALE_UP = 3.0;
  int patchWidth = round(PATCH_SCALE_UP * maxXCoordsSpan) + gridStepX;
  int patchHeight = round(PATCH_SCALE_UP * maxYCoordsSpan) + gridStepY;
  Test::Deblurring::optimizePatchSize(patchWidth);
  Test::Deblurring::optimizePatchSize(patchHeight);

  // Print blur kernel properties.
  std::cout << "Patch width: " << patchWidth << std::endl;
  std::cout << "Patch height: " << patchHeight << std::endl;

  cv::Mat kernelImage(patchHeight, patchWidth, CV_32FC1);

  int patchImageType = CV_8UC1;
  typedef uint8_t PatchDataT;

  cv::Mat patchImage(patchHeight, patchWidth, patchImageType);
  cv::Mat outputPatchImage(gridStepY, gridStepX, patchImageType);

  cv::Mat outputImage = initialImage.clone();
  outputImage = cv::Scalar(0);

  uint8_t* pInitialImageData = (uint8_t*)initialImage.data;
  uint8_t* pOutputImageData = (uint8_t*)outputImage.data;
  PatchDataT* mPatchDataBuffer =
      (PatchDataT*)patchImage.data;  // patchHeight, patchWidth

  // FFT regularizers
  Test::Deblurring::TmFftRegularizer testTmFftRegularizer(patchWidth,
                                                          patchHeight);
  testTmFftRegularizer.setRegularizationWeight(0.01);

  Test::Deblurring::GradientFftRegularizer testGradientFftRegularizer(
      patchWidth, patchHeight);
  testGradientFftRegularizer.setRegularizationWeight(0.00002);

  Test::Deblurring::LaplaceFftRegularizer testLaplaceFftRegularizer(
      patchWidth, patchHeight);
  testLaplaceFftRegularizer.setRegularizationWeight(0.001);

  Test::Deblurring::TvFftRegularizer testTvFftRegularizer(patchWidth,
                                                          patchHeight);
  testTvFftRegularizer.setRegularizationWeight(0.0000001);

  Test::Deblurring::HLTwoThirdsFftRegularizer testHLTwoThirdsFftRegularizer(
      patchWidth, patchHeight);
  testHLTwoThirdsFftRegularizer.setRegularizationWeight(0.00000005);

  Test::Deblurring::HLFftRegularizer testHLFftRegularizer(patchWidth,
                                                          patchHeight);
  testHLFftRegularizer.setRegularizationWeight(0.00000005);
  testHLFftRegularizer.setPowerRegularizationNorm(0.8);

  Test::Deblurring::FftRegularizer* testFftRegularizer =
      &testLaplaceFftRegularizer;

  // Regularizers
  Test::Deblurring::TmIterationRegularizer testTmIterationRegularizer(
      patchWidth, patchHeight);
  testTmIterationRegularizer.setRegularizationWeight(0.01);

  Test::Deblurring::LaplaceIterationRegularizer testLaplaceIterationRegularizer(
      patchWidth, patchHeight);
  testLaplaceIterationRegularizer.setRegularizationWeight(0.001);

  Test::Deblurring::GradientsIterationRegularizer
      testGradientsIterationRegularizer(patchWidth, patchHeight);
  testGradientsIterationRegularizer.setRegularizationWeight(0.00001);

  Test::Deblurring::TvIterationRegularizer testTvIterationRegularizer(
      patchWidth, patchHeight);
  testTvIterationRegularizer.setRegularizationWeight(0.0001);

  Test::Deblurring::HLIterationRegularizer testHlIterationRegularizer(
      patchWidth, patchHeight);
  testHlIterationRegularizer.setRegularizationWeight(0.00001);
  testHlIterationRegularizer.setPowerRegularizationNorm(0.8);

  Test::Deblurring::MaskLaplaceIterationRegularizer
      testMaskLaplaceIterationRegularizer(patchWidth, patchHeight);
  testMaskLaplaceIterationRegularizer.setRegularizationWeight(0.001);

  Test::Deblurring::IterationRegularizer* testRegularizer =
      &testLaplaceIterationRegularizer;

  // Iterative deblurrers
  Test::Deblurring::vanCittertDeblurrer testVanCittertDeblurrer(patchWidth,
                                                                patchHeight);
  testVanCittertDeblurrer.setMinimalNorm(5e-8);
  testVanCittertDeblurrer.setRegularizer(testRegularizer);
  testVanCittertDeblurrer.setUseBestIteration(true);

  Test::Deblurring::LrDeblurrer testLrDeblurrer(patchWidth, patchHeight);
  testLrDeblurrer.setMaxIterations(100);
  testLrDeblurrer.setMinimalNorm(5e-8);
  testLrDeblurrer.setRegularizer(testRegularizer);
  testLrDeblurrer.setUseBestIteration(true);

  Test::Deblurring::CgDeblurrer testCgDeblurrer(patchWidth, patchHeight);
  testCgDeblurrer.setMaxIterations(100);
  testCgDeblurrer.setMinimalNorm(5e-8);
  testCgDeblurrer.setRegularizer(testRegularizer);
  testCgDeblurrer.setUseBestIteration(true);

  // FFT deblurrers
  Test::Deblurring::InverseFftDeblurrer testInverseFftDeblurrer(patchWidth,
                                                                patchHeight);

  Test::Deblurring::CutOffFftDeblurrer testCutOffFftDeblurrer(patchWidth,
                                                              patchHeight);
  testCutOffFftDeblurrer.setCutOffFrequencyX(0.2 * patchWidth);
  testCutOffFftDeblurrer.setCutOffFrequencyY(0.2 * patchHeight);

  Test::Deblurring::LsFftDeblurrer testLsFftDeblurrer(patchWidth, patchHeight);
  testLsFftDeblurrer.setRegularizer(testFftRegularizer);

  Test::Deblurring::ImageDeblurrer& testImageDeblurrer = testLsFftDeblurrer;

  Test::Deblurring::IterativeDeblurrer* testIterativeDeblurrer =
      dynamic_cast<Test::Deblurring::IterativeDeblurrer*>(&testImageDeblurrer);
  testIterativeDeblurrer = NULL;

  std::vector<Test::Deblurring::SparseBlurKernel> patchBlurKernels;
  if (testIterativeDeblurrer != NULL) {
    patchBlurKernels.resize(patchWidth * patchHeight);
  }

  // ProcessImageGridPatches
  for (int gridCenterY = gridStepY / 2; gridCenterY < imageHeight;
       gridCenterY += gridStepY) {
    for (int gridCenterX = gridStepX / 2; gridCenterX < imageWidth;
         gridCenterX += gridStepX) {
      // Calculate blur kernel.
      Test::Deblurring::SparseBlurKernel currentBlurKernel;
      kernelBuilder.calculateAtPoint(gridCenterX, gridCenterY,
                                     currentBlurKernel);
      if (testIterativeDeblurrer != NULL) {
        int blurKernelIndex = 0;
        for (int patchRow = gridCenterY - patchHeight / 2;
             patchRow < gridCenterY + patchHeight / 2; patchRow++) {
          for (int patchCol = gridCenterX - patchWidth / 2;
               patchCol < gridCenterX + patchWidth / 2; patchCol++) {
            kernelBuilder.calculateAtPoint(patchCol, patchRow,
                                           patchBlurKernels[blurKernelIndex++]);
          }
        }
      }

      // Process multichannel image
      for (int channel = 1; channel <= imageChannels; channel++) {
        Test::Deblurring::getPatchFromMultichannelImage(
            gridCenterX - patchWidth / 2, gridCenterY - patchHeight / 2,
            pInitialImageData, imageWidth, imageHeight, imageWidth,
            imageChannels, mPatchDataBuffer, patchWidth, patchHeight,
            patchWidth, channel);

        if (testIterativeDeblurrer != NULL) {
          (*testIterativeDeblurrer)(
              mPatchDataBuffer, patchWidth, &patchBlurKernels[0],
              patchBlurKernels.size(), mPatchDataBuffer, patchWidth);
        } else {
          testImageDeblurrer(mPatchDataBuffer, patchWidth, currentBlurKernel,
                             mPatchDataBuffer, patchWidth);
        }

        PatchDataT* pOutputCropPatch =
            (PatchDataT*)mPatchDataBuffer + (patchWidth - gridStepX + 1) / 2 +
            patchWidth * (patchHeight - gridStepY) / 2;
        Test::Deblurring::putPatchInMultichannelImage(
            gridCenterX - gridStepX / 2, gridCenterY - gridStepY / 2,
            pOutputCropPatch, gridStepX, gridStepY, patchWidth, channel,
            pOutputImageData, imageWidth, imageHeight, imageWidth,
            imageChannels);
      }

      // Draw kernel as separate image.
      extractSparseKernelToImage(
          currentBlurKernel, (Test::Deblurring::point_value_t*)kernelImage.data,
          kernelImage.cols, kernelImage.rows, kernelImage.cols);

      if (testIterativeDeblurrer != NULL) {
        cv::imwrite("current_step.bmp", outputImage);
      }

      cv::namedWindow("Kernel", cv::WINDOW_NORMAL);
      cv::imshow("Kernel", kernelImage);

      cv::namedWindow("Patch", cv::WINDOW_NORMAL);
      cv::imshow("Patch", patchImage);

      cv::namedWindow("Output Image", cv::WINDOW_NORMAL);
      cv::imshow("Output Image", outputImage);

      cv::waitKey(10);

    }  // for (int gridCenterX = ...
  }  // for (int gridCenterY = ...

  cv::imwrite("deblurred_" + imageFilename, outputImage);
  cv::waitKey(0);

  std::cout << "Done." << std::endl;

  return EXIT_SUCCESS;
}
