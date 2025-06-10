//============================================================================
// Name        :
// Author      : Vladislav Antonov
// Version     :
// Copyright   :
// Description : Program to perform integration of gyro data and to calculate
//             : the kernel by reading the output of parsegyro program
//             : and using the video stabilization libraries
//             : Test class approach to gyro calculation
//             : and block Fourier deblurring
// Created on  : April 23, 2012
//============================================================================

// gyro data reading from file
#include <stdint.h>
#include <string.h>

#include <fstream>
#include <iostream>

// OpenCV libraries
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "DiscreteAngularVelocitiesIntegrator.h"
#include "GyroDataCorrection.h"
#include "Versor.h"

// gyro kernel class
#include "GyroBlurKernel.h"

// block deblurring procedures
#include "BlockDeblur.h"
#include "ParseGyroData.h"

using namespace std;

using Test::Time;
using Test::timeFromSeconds;
using Test::timeToSeconds;
using Test::Math::Vector;
using Test::Math::Versor;
using Test::OpticalFlow::DiscreteAngularVelocitiesIntegrator;
using Test::OpticalFlow::GyroDataCorrection;
using Test::OpticalFlow::TimeOrientation;

// this procedure corrects the entered gyro data
void CalcAngularVelocity(
    GyroDataCorrection<float> &gyroCorrector,
    GyroDataCorrection<float>::GyroSample_t gyroSample,
    DiscreteAngularVelocitiesIntegrator<Versor<float>, float>
        &angularVelocityIntegrator) {
  if (gyroCorrector.startup(gyroSample)) {
    // GyroDataCorrection<float>::GyroSample_t gyroSampleSaved = gyroSample;
    while (gyroCorrector.sweepForInsertOrCorrect(gyroSample)) {
      GyroDataCorrection<float>::GyroSample_t gyroSample2 = gyroCorrector.get();
      // recursive call of this function
      CalcAngularVelocity(gyroCorrector, gyroSample2,
                          angularVelocityIntegrator);
    }

    // corrected angular velocity added to container
    angularVelocityIntegrator.addAngularVelocity(gyroSample);
  }
}

int main(int argc, char *argv[]) {
  // container for the gyro data - for 200 samples
  DiscreteAngularVelocitiesIntegrator<Versor<float>, float>
      angularVelocityIntegrator(200);

  // from calibration data
  Versor<float> gyroSpaceToCameraSpaceQuaternion =
      Versor<float>(-0.0470602, 0.698666, 0.71093, -0.0650423);
  Time frameTimestampDelayMinusGyroTimestampDelay =
      -3.78381e+06 - timeFromSeconds<float>(0.0019);
  bool isZAxisInvertedInGyroSpace = 1;

  // Field of view for intrinsic camera matrix K
  const float FOVx = 2657.2 / 2952;  // experimentally measured
  const float FOVy = 2637.0 / 1944;

  Time capture_timestamp = 0;  // capture end timestamp
  Time time_delay = frameTimestampDelayMinusGyroTimestampDelay;  // delay time
  Time time_exposure = timeFromSeconds<float>(0.04999);  // exposure time

  // time for readout in seconds
  const Time time_readout = timeFromSeconds<float>(0.03283);

  // time for rolling shutter delay per row
  float rollingShutterDuration;

  // set number of samples for further treatment
  int N;

  /*
   * Reading the command line parameters
   */

  if (argc < 2) {
    cout << "\nProgram to draw the blur kernels using\nthe gyro data via the "
            "parsegyro algorithm.\n"
         << endl;
    cout << "Usage: " << argv[0] << "  image_file";
    cout << "  [exposure_time  delay_time]\n" << endl;
    return 0;
  }

  cout << "\nExtracting gyrodata from " << argv[1] << endl;

  if (argc > 2) {
    float expos_time = atof(argv[2]);
    if ((expos_time <= 0) || (expos_time == HUGE_VAL)) {
      cout << "\nEntered exposure time defaulted to "
           << timeToSeconds<float>(time_exposure) << " s" << endl;
    } else {
      time_exposure = timeFromSeconds<float>(expos_time);
    }
  }

  if (argc > 3) {
    float dela_time = atof(argv[3]);
    if ((dela_time <= 0) || (dela_time == HUGE_VAL)) {
      cout << "\nEntered delay time defaulted to "
           << timeToSeconds<float>(time_delay) << " s" << endl;
    } else {
      time_delay = timeFromSeconds<float>(dela_time);
    }
  }

  cout << "\nExposure time:   " << timeToSeconds<float>(time_exposure) << endl;
  cout << "Delay time   :   " << timeToSeconds<float>(time_delay) << endl;

  ParseGyroData GyroData = ParseGyroData(argv[1]);

  if (GyroData.Status() != 0) {
    if (GyroData.Status() == 1)
      cout << '\n' << argv[1] << " couldn't be opened.\n";
    if (GyroData.Status() == 2)
      cout << '\n' << argv[1] << " contains no gyro data.\n";

    std::cout << "GyroData.Status() " << GyroData.Status() << "\n";
    return 0;
  }

  N = GyroData.TotalSamples();
  capture_timestamp = GyroData.CaptureTimestamp();

  // samples to be used for averaging
  const int gyroSamplesBeforehand = 20;

  // declare the gyro data correction class - first value is tolerance in ns,
  // second - N samples for averaging; this means the first N samples will be
  // skipped
  Time gyroPeriodVariation = 1000000;
  GyroDataCorrection<float> gyroCorrector(gyroPeriodVariation,
                                          gyroSamplesBeforehand);

  /*
   * Data correction and calculation of rotation versors
   */

  // put the values to container and correct them
  for (int j = 0; j < N; j++) {
    GyroDataCorrection<float>::GyroSample_t gyroSample(
        GyroData.TimeStamp(j),
        Vector<float, 3>(
            GyroData.Wx(j), GyroData.Wy(j),
            (isZAxisInvertedInGyroSpace) ? -GyroData.Wz(j) : GyroData.Wz(j)));

    // correct the displacement between the gyro and image sensor
    gyroSpaceToCameraSpaceQuaternion.rotateVector(gyroSample.velocity);

    // correct the gyro data
    CalcAngularVelocity(gyroCorrector, gyroSample, angularVelocityIntegrator);

    // uncomment this to add the samples without correction and comment the
    // upper line angularVelocityIntegrator.addAngularVelocity(gyroSample);
  }

  // image processing
  cv::Mat imgi = cv::imread(argv[1], cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
  // cv::Mat imgi=cv::imread(argv[1],cv::IMREAD_ANYDEPTH |
  // cv::IMREAD_GRAYSCALE); IplImage
  // *imgk=cv::Mat(cv::Size(imgi.cols, imgi.rows),IPL_DEPTH_8U,1); // separate
  // image, black and white kernels cv::Mat imgk=cvCloneImage(imgi);
  cv::Mat imgk = cv::Mat(cv::Size(imgi.cols, imgi.rows),
                         CV_8UC3);  // separate image, color kernels
  if (imgi.channels() > 1) {
    imgi.copyTo(imgk);
  } else {
    cv::cvtColor(imgi, imgk, cv::COLOR_GRAY2BGR);
  }

  int imw = imgi.cols, imh = imgi.rows;  // image width, image height
  float u0 = imw / 2.0, v0 = imh / 2.0;  // principal point in pixels

  // Rolling shutter shift
  rollingShutterDuration = time_readout / imh;
  cout << "# Rolling shutter time shift: "
       << timeToSeconds<float>(rollingShutterDuration) << "\n\n";

  // Deblur parameters
  GyroDeblurParams BlurParam;

  // Deblur parameters initialization
  BlurParam.FOVx = FOVx;
  BlurParam.FOVy = FOVy;
  BlurParam.ImageHeight = imh;
  BlurParam.ImageWidth = imw;
  BlurParam.OptCentX = u0;
  BlurParam.OptCentY = v0;
  BlurParam.TimeCapture = capture_timestamp;
  BlurParam.TimeDelay = time_delay;
  BlurParam.TimeExposure = time_exposure;
  BlurParam.TimeReadout = time_readout;

  GyroBlurKernel BlurProc = GyroBlurKernel(BlurParam, angularVelocityIntegrator,
                                           N - gyroSamplesBeforehand);

  // loops to draw the kernels
  for (int r0y = 50; r0y < imgk.rows; r0y += 100)
    for (int r0x = 50; r0x < imgk.cols; r0x += 100) {
      // draw the kernel for this point
      cv::Mat kernel = BlurProc.Kernel(r0x, r0y);

      // set the big kernel image for normal kernel
      for (int row = 0; row < kernel.rows; row++)
        for (int col = 0; col < kernel.cols; col++)
          if ((int(r0y + row - kernel.rows / 2) < imgk.rows) &&
              (int(r0y + row - kernel.rows / 2) >= 0) &&
              (int(r0x + col - kernel.cols / 2) < imgk.cols) &&
              (int(r0x + col - kernel.cols / 2) >= 0))
            if (((float *)(kernel.data + row * (kernel.step)))[col] > 0) {
              if (imgk.channels() > 1) {
                // red channel
                ((uchar *)(imgk.data +
                           int(r0y + row - kernel.rows / 2) * (imgk.step)))
                    [int(r0x + col - kernel.cols / 2) * imgk.channels() + 2] =
                        ((float *)(kernel.data + row * (kernel.step)))[col] *
                        255;
                ((uchar *)(imgk.data +
                           int(r0y + row - kernel.rows / 2) * (imgk.step)))
                    [int(r0x + col - kernel.cols / 2) * imgk.channels() + 1] =
                        0;  // green
                ((uchar *)(imgk.data +
                           int(r0y + row - kernel.rows / 2) * (imgk.step)))
                    [int(r0x + col - kernel.cols / 2) * imgk.channels() + 0] =
                        0;  // blue
              } else {
                // black and white
                ((uchar *)(imgk.data +
                           int(r0y + row - kernel.rows / 2) *
                               (imgk.step)))[int(r0x + col - kernel.cols / 2)] =
                    ((float *)(kernel.data + row * (kernel.step)))[col] * 255;
              }
            }

      if (imgk.channels() > 1) {
        ((uchar *)(imgk.data +
                   int(r0y) * (imgk.step)))[int(r0x) * imgk.channels() + 2] =
            0;  // red
        ((uchar *)(imgk.data +
                   int(r0y) * (imgk.step)))[int(r0x) * imgk.channels() + 1] =
            255;  // green
        ((uchar *)(imgk.data +
                   int(r0y) * (imgk.step)))[int(r0x) * imgk.channels() + 0] =
            255;  // blue
      } else {
        // black and white
        ((uchar *)(imgk.data + int(r0y) * (imgk.step)))[int(r0x)] = 255;
      }
    }

  // image window control
  int IMG_WIN;

  if ((imw > 1080) || (imh > 720))
    IMG_WIN = 0;
  else
    IMG_WIN = cv::WINDOW_AUTOSIZE;

  // calculate size of the block for the block deblurring
  int xk1, yk1;
  BlurProc.MaxBlockSize(xk1, yk1);  // max deblurring area
  cout << "Block size (x,y):       " << xk1 << "  " << yk1 << endl;

  int xk, yk;
  BlurProc.MaxKernelSpan(xk, yk);  // max kernel size
  cout << "Max kernel size (x,y):  " << xk << "  " << yk << endl;

  // Temp images for regularization
  cv::Mat img = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
  imgi.convertTo(img, CV_32F);
  img = img / 255;
  cv::Mat imgd1 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
  imgi.convertTo(imgd1, CV_32F);
  imgd1 = imgd1 / 255;
  cv::Mat imgd2 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
  imgi.convertTo(imgd2, CV_32F);
  imgd2 = imgd1 / 255;

  // Images for the first derivatives
  cv::Mat imgw1 = imgd1.clone();
  cv::Mat imgw2 = imgd2.clone();

  float betha = 1;  // regularization weight

  for (int it1 = 0; it1 < 9; it1++, betha *= 2) {
    // regularization weights
    cv::Sobel(imgd1, imgw1, -1, 1, 0);

    for (int row = 0; row < imgw1.rows; row++)
      for (int col = 0; col < imgw1.cols; col++) {
        float a1 = ((float *)(imgw1.data + (row)*imgw1.step))[col];
        float a2 = WeightCalc(a1, betha);
        ((float *)(imgw1.data + (row)*imgw1.step))[col] = a2;
      }

    cv::Sobel(imgd1, imgw2, -1, 0, 1);

    for (int row = 0; row < imgw2.rows; row++)
      for (int col = 0; col < imgw2.cols; col++) {
        float a1 = ((float *)(imgw2.data + (row)*imgw2.step))[col];
        float a2 = WeightCalc(a1, betha);
        ((float *)(imgw2.data + (row)*imgw2.step))[col] = a2;
      }

    // block deblurring procedure
    BlockDeblur(img, imgd2, imgw1, imgw2, betha, BlurProc, xk1, yk1, xk, yk);

    imgd1.copyTo(imgd2);

    char c = cv::waitKey(10);
    if (c == 27) break;

    cout << betha << endl;
  }

  // displays image
  cv::namedWindow("Initial", IMG_WIN);
  cv::namedWindow("Kernels", 0);
  if (IMG_WIN == 0) cv::resizeWindow("Initial", 1080, 720);
  cv::imshow("Initial", imgi);
  cv::imshow("Kernels", imgk);

  // write the kernel in output file
  char *ps_suffix = new char[strlen("_k.jpg") + 1];
  strcpy(ps_suffix, "_k.jpg");
  char ps_data_file[strlen(argv[1]) + strlen(ps_suffix)];
  strcpy(ps_data_file, argv[1]);
  char *pch = strchr(ps_data_file, '.');
  strncpy(pch, ps_suffix, strlen(ps_suffix) + 1);
  cv::imwrite(ps_data_file, imgk);
  cout << "Kernels saved as:  " << ps_data_file << '\n' << endl;
  delete[] ps_suffix;

  // write the kernel in output file
  char *db_suffix = new char[strlen("_debl.jpg") + 1];
  strcpy(db_suffix, "_debl.jpg");
  char db_data_file[strlen(argv[1]) + strlen(db_suffix)];
  strcpy(db_data_file, argv[1]);
  pch = strchr(db_data_file, '.');
  strncpy(pch, db_suffix, strlen(db_suffix) + 1);
  imgd2 = imgd2 * 255;
  cv::imwrite(db_data_file, imgd2);
  cout << "Deblurred image saved as:  " << db_data_file << '\n' << endl;
  delete[] db_suffix;

  cv::waitKey(0);

  cv::destroyWindow("Initial");
  cv::destroyWindow("Kernels");

  return 0;
}
