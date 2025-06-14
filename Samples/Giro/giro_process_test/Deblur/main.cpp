//============================================================================
// Name        :
// Author      : Vladislav Antonov
// Version     :
// Copyright   :
// Description : Program to perform integration of gyro data and to calculate
//             : the kernel by reading the output of parsegyro program
//             : and using the video stabilization libraries
//             : Test blind deblurring
// Created on  : April 25, 2012
//============================================================================

// blind deconvolution kernel estimation
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

// gyro data reading from file
#include "BlindDeblur.h"
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
  cv::Mat imgi =
      cv::imread(argv[1], cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);

  // initial image resized
  cv::Mat imgs =
      cv::Mat(cv::Size(imgi.cols / 2, imgi.rows / 2), imgi.depth(), 1);
  cv::resize(imgi, imgs, cv::Size(imgi.cols / 2, imgi.rows / 2));

  cv::Mat img = cv::Mat(cv::Size(imgs.cols, imgs.rows), CV_32FC1);
  imgs.convertTo(img, CV_32F);
  img = img / 255;
  cv::Mat imgd1 = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);
  img.copyTo(imgd1);
  cv::Mat imgd2 = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);
  img.copyTo(imgd2);

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

  // image window control
  int IMG_WIN;

  if ((imw > 1080) || (imh > 720))
    IMG_WIN = 0;
  else
    IMG_WIN = cv::WINDOW_AUTOSIZE;

  // components for blind deblurring
  int Nbd = 3;
  // arrays fro blind deblurring - x, r, p, A.p
  double Xbd[Nbd], Rbd[Nbd], Pbd[Nbd], APbd[Nbd];

  // calculate size of the block for the block deblurring
  int xk1, yk1;
  BlurProc.MaxBlockSize(xk1, yk1);  // max deblurring area
  cout << "Block size (x,y):       " << xk1 << "  " << yk1 << endl;

  int xk, yk;
  BlurProc.MaxKernelSpan(xk, yk);  // max kernel size
  cout << "Max kernel size (x,y):  " << xk << "  " << yk << endl;

  // initial initialization of arrays
  double resid = 0;
  for (int i = 0; i < Nbd; i++)
    Rbd[i] = dRdx(img, BlurProc, xk1, yk1, xk, yk, i);

  for (int i = 0; i < Nbd; i++) resid += Rbd[i] * Rbd[i];

  for (int i = 0; i < Nbd; i++) Pbd[i] = Rbd[i];
  for (int i = 0; i < Nbd; i++) Xbd[i] = IndexedBlurParam(BlurParam, i);

  for (int i = 0; i < Nbd; i++) cout << Xbd[i] << "  ";
  cout << resid << endl;

  // for (int i=0; i<7; i++) cout << IndexedBlurParam(BlurParam, i) << "  ";
  // cout << endl;

  // Deblur parameters for optimization
  GyroDeblurParams BlurParamCG = BlurParam;
  GyroBlurKernel BlurProcCG = BlurProc;
  double ak, bk;

  // CG loop
  int iter = 0;
  do {
    // change the gyro blur data
    for (int i = 0; i < Nbd; i++) IndexedBlurParamSet(BlurParamCG, i, Pbd[i]);
    BlurProcCG.SetBlurParameters(BlurParamCG);

    // calculate size of the block for the block deblurring
    BlurProcCG.MaxBlockSize(xk1, yk1);  // max deblurring area
    cout << "Block size (x,y):       " << xk1 << "  " << yk1 << endl;

    BlurProcCG.MaxKernelSpan(xk, yk);  // max kernel size
    cout << "Max kernel size (x,y):  " << xk << "  " << yk << endl;

    // A.p
    for (int i = 0; i < Nbd; i++)
      APbd[i] = dRdx(img, BlurProcCG, xk1, yk1, xk, yk, i);

    // alpha_k
    ak = 0;
    double a1 = 0;
    for (int i = 0; i < Nbd; i++) {
      a1 += Rbd[i] * Rbd[i];
      ak += Pbd[i] * APbd[i];
    }
    ak = a1 / ak;

    // x = x + ak*pk
    for (int i = 0; i < Nbd; i++) Xbd[i] += ak * Pbd[i];

    // betha_k
    double b1 = 0;
    for (int i = 0; i < Nbd; i++) {
      b1 += -ak * APbd[i] * Rbd[i];
      // r = r-ak*A.p
      Rbd[i] -= ak * APbd[i];
    }
    bk = b1 / a1;

    for (int i = 0; i < Nbd; i++) resid += Rbd[i] * Rbd[i];

    // p = r + bk*p
    for (int i = 0; i < Nbd; i++) Pbd[i] = Rbd[i] + bk * Pbd[i];

    for (int i = 0; i < Nbd; i++) cout << Xbd[i] << "  ";
    cout << resid << endl;

    iter++;

  } while (iter < 100);

  // cout << Residual(img, BlurProc, xk1, yk1, xk, yk) << endl;

  // cout << time_exposure << "  " << Time(time_exposure*1.001) << endl;

  // for (int i=0; i<7; i++) cout << i << "  " << dRdx(img, BlurProc, xk1, yk1,
  // xk, yk,i) << endl;

  // displays image
  cv::namedWindow("Initial", IMG_WIN);
  if (IMG_WIN == 0) cv::resizeWindow("Initial", 1080, 720);
  cv::imshow("Initial", imgi);
  cv::namedWindow("Deblurred", IMG_WIN);
  cv::imshow("Deblurred", imgd1);
  cv::namedWindow("ReBlurred", IMG_WIN);
  cv::imshow("ReBlurred", imgd2);

  // write the kernel in output file
  char *db_suffix = new char[strlen("_debl.jpg") + 1];
  strcpy(db_suffix, "_debl.jpg");
  char db_data_file[strlen(argv[1]) + strlen(db_suffix)];
  strcpy(db_data_file, argv[1]);
  char *pch = strchr(db_data_file, '.');
  strncpy(pch, db_suffix, strlen(db_suffix) + 1);
  imgd1 = imgd1 * 255;
  cv::imwrite(db_data_file, imgd1);
  cout << "Deblurred image saved as:  " << db_data_file << '\n' << endl;
  delete[] db_suffix;

  cv::waitKey(0);

  cv::destroyWindow("Initial");
  cv::destroyWindow("Deblurred");
  cv::destroyWindow("ReBlurred");
  return 0;
}
