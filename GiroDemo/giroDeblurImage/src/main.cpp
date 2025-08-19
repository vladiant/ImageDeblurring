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
#include "GradientsIterationRegularizer.h"
#include "GyroBlurKernelBuilder.h"
#include "GyroBlurParams.h"
#include "GyroDataReader.h"
#include "HLIterationRegularizer.h"
#include "LaplaceIterationRegularizer.h"
#include "LrDeblurrer.h"
#include "MaskLaplaceIterationRegularizer.h"
#include "SparseBlurKernel.h"
#include "TmIterationRegularizer.h"
#include "TvIterationRegularizer.h"
#include "vanCittertDeblurrer.h"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "\nProgram to deblur image using\nthe gyro data from the "
                 "parsegyro data file.\n"
              << std::endl;
    std::cout << "Usage: " << argv[0] << "  image_file  parsegyro_data_file";
    return 0;
  }

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
  Test::Time exposureDuration = Test::timeFromSeconds<float>(0.04999);

  // Time for readout in seconds
  const Test::Time readoutTime = Test::timeFromSeconds<float>(0.03283);

  // *** End of sample calibration data *** //

  // Reading gyro data from txt file
  const int numberGyroSamples = 200;
  const std::string gyroDataFilename(argv[2]);
  const std::string imageFilename(argv[1]);

  Test::Deblurring::GyroDataReader gyroDataReader(numberGyroSamples);

  std::cout << "Reading data from image file: " << imageFilename << std::endl;

  gyroDataReader.readDataFromParsedTextFile(
      const_cast<std::string&>(gyroDataFilename));

  std::cout << "Correcting angular velocities. " << std::endl;

  gyroDataReader.correctAngularVelocities(gyroSpaceToCameraSpaceQuaternion,
                                          isZAxisInvertedInGyroSpace);

  std::cout << "Reading capture timestamp. " << std::endl;

  Test::Time captureTimeStamp = gyroDataReader.getCaptureTimestamp();

  std::cout << "Capture timestamp: " << captureTimeStamp << std::endl;

  // Load image and print sampling points.
  cv::Mat initialImage = cv::imread(imageFilename);
  cv::namedWindow(imageFilename, cv::WINDOW_AUTOSIZE);

  int imageWidth = initialImage.cols;
  int imageHeight = initialImage.rows;
  int imagePpln = initialImage.cols;

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

  cv::Mat grayImage(imageHeight, imageWidth, CV_8UC1);
  cv::cvtColor(initialImage, grayImage, cv::COLOR_BGR2GRAY, 1);
  cv::Mat outputGrayImage(imageHeight, imageWidth, CV_8UC1);

  cv::namedWindow("Gray Image", cv::WINDOW_NORMAL);
  cv::imshow("Gray Image", grayImage);

  uint8_t* pInitialImageData = (uint8_t*)grayImage.data;
  uint8_t* pOutputImageData = (uint8_t*)outputGrayImage.data;

  // Regularizers
  Test::Deblurring::TmIterationRegularizer testTmIterationRegularizer(
      initialImage.cols, initialImage.rows);
  testTmIterationRegularizer.setRegularizationWeight(0.01);

  Test::Deblurring::LaplaceIterationRegularizer testLaplaceIterationRegularizer(
      initialImage.cols, initialImage.rows);
  testLaplaceIterationRegularizer.setRegularizationWeight(0.001);

  Test::Deblurring::GradientsIterationRegularizer
      testGradientsIterationRegularizer(initialImage.cols, initialImage.rows);
  testGradientsIterationRegularizer.setRegularizationWeight(0.0001);

  Test::Deblurring::TvIterationRegularizer testTvIterationRegularizer(
      initialImage.cols, initialImage.rows);
  testTvIterationRegularizer.setRegularizationWeight(0.0001);

  Test::Deblurring::HLIterationRegularizer testHlIterationRegularizer(
      initialImage.cols, initialImage.rows);
  testHlIterationRegularizer.setRegularizationWeight(0.00005);
  testHlIterationRegularizer.setPowerRegularizationNorm(0.8);

  Test::Deblurring::MaskLaplaceIterationRegularizer
      testMaskLaplaceIterationRegularizer(initialImage.cols, initialImage.rows);
  testMaskLaplaceIterationRegularizer.setRegularizationWeight(0.001);

  Test::Deblurring::IterationRegularizer* testRegularizer =
      &testLaplaceIterationRegularizer;

  // Iterative deblurrers
  Test::Deblurring::vanCittertDeblurrer testVanCittertDeblurrer(
      initialImage.cols, initialImage.rows);
  testVanCittertDeblurrer.setMinimalNorm(5e-7);
  testVanCittertDeblurrer.setRegularizer(testRegularizer);

  Test::Deblurring::LrDeblurrer testLrDeblurrer(initialImage.cols,
                                                initialImage.rows);
  testLrDeblurrer.setMaxIterations(100);
  testLrDeblurrer.setMinimalNorm(5e-7);
  testLrDeblurrer.setRegularizer(testRegularizer);

  Test::Deblurring::CgDeblurrer testCgDeblurrer(initialImage.cols,
                                                initialImage.rows);
  testCgDeblurrer.setMaxIterations(100);
  testCgDeblurrer.setMinimalNorm(5e-7);
  testCgDeblurrer.setRegularizer(testRegularizer);

  Test::Deblurring::IterativeDeblurrer& testImageDeblurrer =
      testVanCittertDeblurrer;

  testImageDeblurrer(pInitialImageData, imageWidth, kernelBuilder,
                     pOutputImageData, imageWidth);

  cv::namedWindow("Deblurred Image", cv::WINDOW_NORMAL);
  cv::imshow("Deblurred Image", grayImage);
  cv::imwrite("deblurred.bmp", outputGrayImage);
  cv::waitKey(0);

  std::cout << "Done." << std::endl;

  return EXIT_SUCCESS;
}
