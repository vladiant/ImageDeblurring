//============================================================================
// Name        :
// Author      : Vladislav Antonov
// Version     :
// Copyright   :
// Description : Program to perform integration of gyro data and to calculate
//             : the kernel by reading the output of parsegyro program
//             : and using the video stabilization libraries
//             : Test class approach to gyro calculation
// Created on  : April 21, 2012
//============================================================================

// gyro kernel class
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
#include "GyroBlurKernel.h"
#include "GyroDataCorrection.h"
#include "Versor.h"

using namespace std;

using Test::Time;
using Test::timeFromSeconds;
using Test::timeToSeconds;
using Test::Math::Vector;
using Test::Math::Versor;
using Test::OpticalFlow::DiscreteAngularVelocitiesIntegrator;
using Test::OpticalFlow::GyroDataCorrection;
using Test::OpticalFlow::TimeOrientation;

// Field of view for intrinsic camera matrix K
const float FOVx = 2657.2 / 2952;  // experimentally measured
const float FOVy = 2637.0 / 1944;

// from calibration data
Versor<float> gyroSpaceToCameraSpaceQuaternion =
    Versor<float>(-0.0470602, 0.698666, 0.71093, -0.0650423);
Time frameTimestampDelayMinusGyroTimestampDelay = -3.78381e+06;
bool isZAxisInvertedInGyroSpace = 1;

// container for the gyro data - for 200 samples
DiscreteAngularVelocitiesIntegrator<Versor<float>, float>
    angularVelocityIntegrator(200);

// this procedure corrects the entered gyro data
void CalcAngularVelocity(GyroDataCorrection<float> &gyroCorrector,
                         GyroDataCorrection<float>::GyroSample_t gyroSample) {
  if (gyroCorrector.startup(gyroSample)) {
    // GyroDataCorrection<float>::GyroSample_t gyroSampleSaved = gyroSample;
    while (gyroCorrector.sweepForInsertOrCorrect(gyroSample)) {
      GyroDataCorrection<float>::GyroSample_t gyroSample2 = gyroCorrector.get();
      // recursive call of this function
      CalcAngularVelocity(gyroCorrector, gyroSample2);
    }

    // corrected angular velocity added to container
    angularVelocityIntegrator.addAngularVelocity(gyroSample);
  }
}

int main(int argc, char *argv[]) {
  // samples to be used for averaging
  const int gyroSamplesBeforehand = 20;

  // set number of samples for further treatment
  int N;

  Time capture_timestamp = 0;  // capture end timestamp
  Time time_delay = frameTimestampDelayMinusGyroTimestampDelay;  // delay time
  Time time_exposure = timeFromSeconds<float>(0.05999);  // exposure time

  // time for readout in seconds
  const Time time_readout = timeFromSeconds<float>(0.03283);

  // time for rolling shutter delay per row
  float rollingShutterDuration;

  /*
   * Reading the command line parameters
   */

  if (argc < 3) {
    cout << "\nProgram to draw the blur kernels using\nthe gyro data from the "
            "parsegyro data file.\n"
         << endl;
    cout << "Usage: " << argv[0] << "  image_file  parsegyro_data_file";
    cout << "  [exposure_time  delay_time]\n" << endl;
    return 0;
  }

  cout << "\nExtracting gyrodata from " << argv[2] << endl;

  if (argc > 3) {
    float expos_time = atof(argv[3]);
    if ((expos_time <= 0) || (expos_time == HUGE_VAL)) {
      cout << "\nEntered exposure time defaulted to "
           << timeToSeconds<float>(time_exposure) << " s" << endl;
    } else {
      time_exposure = timeFromSeconds<float>(expos_time);
    }
  }

  if (argc > 4) {
    float dela_time = atof(argv[4]);
    if ((dela_time <= 0) || (dela_time == HUGE_VAL)) {
      cout << "\nEntered delay time defaulted to "
           << timeToSeconds<float>(time_delay) << " s" << endl;
    } else {
      time_delay = timeFromSeconds<float>(dela_time);
    }
  }

  cout << "\nExposure time:   " << timeToSeconds<float>(time_exposure) << endl;
  cout << "Delay time   :   " << timeToSeconds<float>(time_delay) << endl;

  /*
   * File reading procedure starts here
   */

  // string constants to be searched in data file
  const char CAPTURE_MARK[] = "capture timestamp";
  const char GYRO_SAMPLES_MARK[] = "total gyro samples";

  // number of rows in data file
  int dfile_rows;

  // temporal variable for lines in text file
  string line;

  dfile_rows = 0;

  ifstream datafile(argv[2]);
  if (datafile.is_open()) {
    // check flags for availability of the data
    bool flag1 = false, flag2 = false;

    while (datafile.good()) {
      char *pcs;  // pointer to string position
      dfile_rows++;
      getline(datafile, line);

      // find the capture timestamp
      pcs = strstr(&line[0], CAPTURE_MARK);
      if (pcs != NULL) {
        char *pEnd;
        pcs += strlen(CAPTURE_MARK);  //
        capture_timestamp = strtol(pcs, &pEnd, 10);
        flag1 = true;
      }

      // find the gyro samples number
      pcs = strstr(&line[0], GYRO_SAMPLES_MARK);
      if (pcs != NULL) {
        pcs += strlen(GYRO_SAMPLES_MARK);  //
        N = atoi(pcs);
        flag2 = true;
      }

      // cout << dfile_rows << "  " << line << endl;
    }
    datafile.close();

    if (!(flag1)) {
      cout << '\n'
           << CAPTURE_MARK << " not found in " << argv[2] << " ." << endl;
      return 0;
    }

    if (!(flag2)) {
      cout << '\n'
           << GYRO_SAMPLES_MARK << " not found in " << argv[2] << " ." << endl;
      return 0;
    }

  } else {
    cout << '\n' << argv[2] << " couldn't be opened.\n";
    return 0;
  }

  /*
   * File reading procedure ends here
   */

  // define the time when the capture starts
  // Time
  // time_capture_start=capture_timestamp-time_exposure-time_readout;

  // timestamp, angular velocities
  float *wx = new float[N];       // angular velocity x [rad/s]
  float *wy = new float[N];       // angular velocity y [rad/s]
  float *wz = new float[N];       // angular velocity z [rad/s]
  Time *timestamp = new Time[N];  // timestamp [ns]

  // declare the gyro data correction class - first value is tolerance in ns,
  // second - N samples for averaging; this means the first N samples will be
  // skipped
  Time gyroPeriodVariation = 1000000;
  GyroDataCorrection<float> gyroCorrector(gyroPeriodVariation,
                                          gyroSamplesBeforehand);

  // read the values from file
  ifstream datafile1(argv[2]);
  for (int j = 0; j < N; j++) {
    datafile1 >> timestamp[j] >> wx[j] >> wy[j] >> wz[j];
    // cout << j << "  " << timestamp[j] << "  " << wx[j] << "  " << wy[j] << "
    // " << wz[j] << endl;
    if (isZAxisInvertedInGyroSpace) {
      wz[j] *= -1.0;
    }
  }
  datafile1.close();

  /*
   * Data correction and calculation of rotation versors
   */

  // put the values to container and correct them
  for (int j = 0; j < N; j++) {
    GyroDataCorrection<float>::GyroSample_t gyroSample(
        timestamp[j], Vector<float, 3>(wx[j], wy[j], wz[j]));

    // correct the displacement between the gyro and image sensor
    gyroSpaceToCameraSpaceQuaternion.rotateVector(gyroSample.velocity);

    // correct the gyro data
    CalcAngularVelocity(gyroCorrector, gyroSample);

    // uncomment this to add the samples without correction and comment the
    // upper line angularVelocityIntegrator.addAngularVelocity(gyroSample);
  }

  // image processing
  cv::Mat imgi = cv::imread(argv[1], cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
  // IplImage *imgi=cv::imread(argv[1],cv::IMREAD_ANYDEPTH |
  // cv::IMREAD_GRAYSCALE); IplImage
  // *imgk=cv::Mat(cvGetSize(imgi),IPL_DEPTH_8U,1); // separate image,
  // black and white kernels IplImage *imgk=cvCloneImage(imgi);
  cv::Mat imgk = cv::Mat(cv::Size(imgi.cols, imgi.rows),
                         CV_8UC3);  // separate image, color kernels
  if (imgi.channels() > 1) {
    imgi.copyTo(imgk);
  } else {
    cv::cvtColor(imgi, imgk, cv::COLOR_GRAY2BGR);
  }

  int imw = imgi.cols, imh = imgi.rows;  // image width, image height
  float u0 = imw / 2.0, v0 = imh / 2.0;  // principal point in pixels
  // float x0b, y0b;                        // base coordinates (zero point) for
  // kernel estimation

  // Rolling shutter shift
  rollingShutterDuration = time_readout / imh;
  cout << "# Rolling shutter time shift: "
       << timeToSeconds<float>(rollingShutterDuration) << "\n\n";

  GyroDeblurParams BlurParam;

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

  int xk, yk;
  BlurProc.MaxKernelSpan(xk, yk);
  cout << "Maximal size of kernel (x,y):" << endl;
  cout << xk << "  " << yk << endl;

  int xk1, yk1;
  BlurProc.MaxBlockSize(xk1, yk1);
  cout << "Maximal size of block (x,y):" << endl;
  cout << xk1 << "  " << yk1 << endl;

  // loops to draw the kernels
  for (int r0y = 50; r0y < imgk.rows; r0y += 100)
    for (int r0x = 50; r0x < imgk.cols; r0x += 100) {
      /*
                              // draw the kernel for this point
                              IplImage *kernel = BlurProc.Kernel(r0x,r0y);
      */

      // draw sparse kernel for this point
      cv::SparseMatIterator it;
      cv::SparseMat skernel = BlurProc.SparseKernel(r0x, r0y);
      int dims[] = {skernel.size()[0], skernel.size()[1]};

      /*
                              //set the big kernel image for normal kernel
                              for (int row=0; row<kernel.rows;row++)
                                      for (int col=0; col<kernel.cols;col++)
                                              if
         ((int(r0y+row-kernel.rows/2)<imgk.rows)&&(int(r0y+row-kernel.rows/2)>=0)&&(int(r0x+col-kernel.cols/2)<imgk.cols)&&(int(r0x+col-kernel.cols/2)>=0))
                                                      if
         (((float*)(kernel.data + row*(kernel.step)))[col]>0)
                                                              //((uchar*)(imgk.data
         +
         int(r0x+row)*(imgk.step)))[int(r0y+col)]=((float*)(kernel.data
         + row*(kernel.step)))[col]*255;
                                                              {
                                                                      if
         (imgk.channels()>1)
                                                                      {
                                                                              //red
         channel
                                                                              ((uchar*)(imgk.data
         +
         int(r0y+row-kernel.rows/2)*(imgk.step)))[int(r0x+col-kernel.cols/2)*imgk.channels()+2]
                                                                                                                                                            =((float*)(kernel.data + row*(kernel.step)))[col]*255;
                                                                              ((uchar*)(imgk.data
         +
         int(r0y+row-kernel.rows/2)*(imgk.step)))[int(r0x+col-kernel.cols/2)*imgk.channels()+1]=0;
         //green
                                                                              ((uchar*)(imgk.data
         +
         int(r0y+row-kernel.rows/2)*(imgk.step)))[int(r0x+col-kernel.cols/2)*imgk.channels()+0]=0;
         //blue
                                                                      }
                                                                      else
                                                                      {
                                                                              //black
         and white
                                                                              ((uchar*)(imgk.data
         +
         int(r0y+row-kernel.rows/2)*(imgk.step)))[int(r0x+col-kernel.cols/2)]
                                                                                                                                              =((float*)(kernel.data + row*(kernel.step)))[col]*255;
                                                                      }
                                                              }
      */

      // set the big kernel image from sparse matrix
      for (it = skernel.begin(); it != skernel.end(); ++it) {
        cv::SparseMat::Node *node = it.node();
        int *idx = node->idx;
        float val = it.value<float>();
        if ((int(r0y + idx[1] - dims[1] / 2) < imgk.rows) &&
            (int(r0y + idx[1] - dims[1] / 2) >= 0) &&
            (int(r0x + idx[0] - dims[0] / 2) < imgk.cols) &&
            (int(r0x + idx[0] - dims[0] / 2) >= 0))
          if (val > 0) {
            if (imgk.channels() > 1) {
              // red channel
              ((uchar *)(imgk.data +
                         int(r0y + idx[1] - dims[1] / 2) * (imgk.step)))
                  [int(r0x + idx[0] - dims[0] / 2) * imgk.channels() + 2] =
                      val * 255;
              ((uchar *)(imgk.data +
                         int(r0y + idx[1] - dims[1] / 2) * (imgk.step)))
                  [int(r0x + idx[0] - dims[0] / 2) * imgk.channels() + 1] =
                      0;  // green
              ((uchar *)(imgk.data +
                         int(r0y + idx[1] - dims[1] / 2) * (imgk.step)))
                  [int(r0x + idx[0] - dims[0] / 2) * imgk.channels() + 0] =
                      0;  // blue
            } else {
              // black and white
              ((uchar *)(imgk.data +
                         int(r0y + idx[1] - dims[1] / 2) *
                             (imgk.step)))[int(r0x + idx[0] - dims[0] / 2)] =
                  val * 255;
            }
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
      /*
                              //show the normal kernel
                              cv::namedWindow("Kernel", 0);
                              cvScale(kernel,kernel,255);
                              cv::imshow("Kernel", kernel);
                              cv::waitKey(2);
      */
    }

  // image window control
  int IMG_WIN;

  if ((imw > 1080) || (imh > 720))
    IMG_WIN = 0;
  else
    IMG_WIN = cv::WINDOW_AUTOSIZE;

  // displays image
  cv::namedWindow("Initial", IMG_WIN);
  cv::namedWindow("Kernels", 0);
  if (IMG_WIN == 0) cv::resizeWindow("Initial", 1080, 720);
  cv::imshow("Initial", imgi);
  cv::imshow("Kernels", imgk);

  // write the kernel in output file
  const char ps_suffix[] = "_k.jpg";
  char ps_data_file[strlen(argv[1]) + strlen(ps_suffix)];
  strcpy(ps_data_file, argv[1]);
  char *pch = strchr(ps_data_file, '.');
  strncpy(pch, ps_suffix, strlen(ps_suffix) + 1);
  cv::imwrite(ps_data_file, imgk);
  cout << "Kernels saved as:  " << ps_data_file << '\n' << endl;

  cv::waitKey(0);

  cv::destroyWindow("Initial");
  cv::destroyWindow("Kernels");

  // release the memory
  delete[] timestamp;
  delete[] wx;
  delete[] wy;
  delete[] wz;

  return 0;
}
