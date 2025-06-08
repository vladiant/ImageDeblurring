//============================================================================
// Name        :
// Author      : Vladislav Antonov
// Version     :
// Copyright   :
// Description : Program to perform integration of gyro data and to calculate
//             : the kernel by using sparse matrices, algorithm from the
//             : parsegyro program and using the video stabilization libraries
// Created on  : Mar 28, 2012
//============================================================================

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

// parse_gyro_data segment
struct DataSample {
  Time timestamp;
  float x;
  float y;
  float z;
  int extra;
};
typedef struct DataSample DataSample;

struct MovementDataHeader {
  char md_identifier[16];
  Time md_capture_timestamp;
  unsigned int total_gyro_samples;
  unsigned int last_sample_index;
  Time capture_command_timestamp;
};

typedef struct MovementDataHeader MovementDataHeader;

unsigned int segment_len = 16384;

DataSample *gyroSamples;
MovementDataHeader *gyroDataHeader;
// parse_gyro_data segment ends here

int main(int argc, char *argv[]) {
  // samples to be used for averaging
  const int SAMPL_AVER = 20;

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

  /*
   * File reading procedure starts here
   */

  // parse_gyro_data segment
  FILE *pFile;
  unsigned int total_samples, last_sample;
  char *buf;
  int j = 0;  // position for data transfer

  pFile = fopen(argv[1], "r");

  if (pFile == NULL) {
    cout << '\n' << argv[1] << " couldn't be opened.\n";
    return 0;
  }

  buf = new char[segment_len];
  gyroDataHeader = (MovementDataHeader *)malloc(sizeof(MovementDataHeader));

  fseek(pFile, 0, SEEK_END);
  long size = ftell(pFile);
  size -= segment_len;
  fseek(pFile, size, SEEK_SET);
  fread(buf, sizeof(char), segment_len, pFile);

  memcpy(gyroDataHeader, buf, sizeof(MovementDataHeader));

  total_samples = gyroDataHeader->total_gyro_samples;

  // set number of samples for further treatment
  N = total_samples;

  // set the capture end timestamp
  capture_timestamp = gyroDataHeader->md_capture_timestamp;

  last_sample = gyroDataHeader->last_sample_index;

  gyroSamples = (DataSample *)malloc(sizeof(DataSample) * total_samples);

  memcpy(gyroSamples, buf + sizeof(MovementDataHeader),
         sizeof(DataSample) * total_samples);

  // print the gyro data parameters
  cout << "\n# Capture timestamp: "
       << gyroDataHeader->md_capture_timestamp -
              gyroSamples[last_sample].timestamp
       << "\n";
  cout << "# User press timestamp: "
       << gyroDataHeader->capture_command_timestamp -
              gyroSamples[last_sample].timestamp
       << "\n";
  cout << "# Total gyro samples: " << gyroDataHeader->total_gyro_samples
       << "\n\n";

  // data transfer completed, release memory
  delete[] buf;
  free(gyroSamples);
  free(gyroDataHeader);
  fclose(pFile);

  /*
   * File reading procedure ends here
   */

  // define the time when the capture starts
  Time time_capture_start = capture_timestamp - time_exposure - time_readout;

  // timestamp, angular velocities
  float *wx = new float[N];       // angular velocity x [rad/s]
  float *wy = new float[N];       // angular velocity y [rad/s]
  float *wz = new float[N];       // angular velocity z [rad/s]
  Time *timestamp = new Time[N];  // timestamp [ns]

  // declare the gyro data correction class - first value is tolerance in ns,
  // second - N samples for averaging; this means the first N samples will be
  // skipped
  GyroDataCorrection<float> gyroCorrector(1000000, SAMPL_AVER);

  // read the values from file
  for (unsigned int i = last_sample; i < total_samples; i++, j++) {
    timestamp[j] = gyroSamples[i].timestamp;
    wx[j] = gyroSamples[i].x;
    wy[j] = gyroSamples[i].y;
    wz[j] = gyroSamples[i].z;
    if (isZAxisInvertedInGyroSpace) {
      wz[j] *= -1.0;
    }
  }

  for (unsigned int i = 0; i < last_sample; i++, j++) {
    timestamp[j] = gyroSamples[i].timestamp;
    wx[j] = gyroSamples[i].x;
    wy[j] = gyroSamples[i].y;
    wz[j] = gyroSamples[i].z;
    if (isZAxisInvertedInGyroSpace) {
      wz[j] *= -1.0;
    }
  }

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
  /*
          //correct the displacement between the gyro and image sensor
          for (int j=0;j<N-SAMPL_AVER-1; j++)
          {
                  std::pair<long int, Versor<float> >
     CurrentGyroSample =
     angularVelocityIntegrator.orientations().orientations()[j];
                  angularVelocityIntegrator.orientations().orientations()[j].second=gyroSpacetoCameraQuaternion*CurrentGyroSample.second;
                  //cout << CurrentGyroSample.first << "  " <<
     angularVelocityIntegrator.orientations().orientations()[j].second << endl;
          }
  */

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
  float x0b, y0b;  // base coordinates (zero point) for kernel estimation

  // Rolling shutter shift
  rollingShutterDuration = time_readout / imh;
  cout << "# Rolling shutter time shift: "
       << timeToSeconds<float>(rollingShutterDuration) << "\n\n";

  // search for the time interval of exposure
  int j_i, j_f, j_0;
  bool flag1 = 1;  // initial point, final point, zero point, control flags

  // first timestamp
  TimeOrientation<Versor<float> > FirstGyroSample =
      angularVelocityIntegrator.orientations().orientations()[0];
  // Time time_zero=FirstGyroSample.first;	//gyro capture time

  // zero point
  for (int j = 0; j < N - SAMPL_AVER - 1; j++) {
    TimeOrientation<Versor<float> > TEST =
        angularVelocityIntegrator.orientations().orientations()[j];

    if ((TEST.time > time_capture_start + time_delay) && (flag1)) {
      j_0 = j - 1;
      break;
    }
  }

  // loops to draw the kernels
  for (int r0y = 50; r0y < imgk.rows; r0y += 100)
    for (int r0x = 50; r0x < imgk.cols; r0x += 100) {
      bool flag1 = 1, flag2 = 1;  // control flags
      float x1, y1;               // current coordinates

      // search for the exposure interval
      for (int j = 0; j < N - SAMPL_AVER - 1; j++) {
        TimeOrientation<Versor<float> > CurrentSample =
            angularVelocityIntegrator.orientations().orientations()[j];

        if ((CurrentSample.time >
             time_capture_start + rollingShutterDuration * r0y + time_delay) &&
            (flag1)) {
          j_i = j - 1;
          flag1 = 0;
        }

        if ((CurrentSample.time > time_capture_start + time_exposure +
                                      rollingShutterDuration * r0y +
                                      time_delay) &&
            (flag2)) {
          j_f = j;
          flag2 = 0;
          break;
        }
      }

      // number of samples for interpolation
      int bsamples = j_f - j_i + 1;

      // cout << "\nFirst sample: " << j_i << endl;
      // cout << "Last sample : " << j_f << '\n' << endl;

      // non homogenious coordinates
      float x0 = (r0x - u0) / (FOVx * imw), y0 = (r0y - v0) / (FOVy * imh);

      // zero point calculation
      float x0b0, x0b1, y0b0, y0b1;
      {
        Versor<float> CurrentVersor = angularVelocityIntegrator.orientations()
                                          .orientations()[j_0]
                                          .orientation;
        Vector<float, 3> VectorVersor = Vector<float, 3>(x0, y0, 1.0);
        CurrentVersor.rotateVector(VectorVersor);
        x1 = VectorVersor.x() / VectorVersor.z();
        y1 = VectorVersor.y() / VectorVersor.z();
        x0b0 = x1 * (FOVx * imw) + u0 - r0x;
        y0b0 = y1 * (FOVy * imh) + v0 - r0y;
        Time time_j0 =
            angularVelocityIntegrator.orientations().orientations()[j_0].time;

        CurrentVersor = angularVelocityIntegrator.orientations()
                            .orientations()[j_0 + 1]
                            .orientation;
        VectorVersor = Vector<float, 3>(x0, y0, 1.0);
        CurrentVersor.rotateVector(VectorVersor);
        x1 = VectorVersor.x() / VectorVersor.z();
        y1 = VectorVersor.y() / VectorVersor.z();
        x0b1 = x1 * (FOVx * imw) + u0 - r0x;
        y0b1 = y1 * (FOVy * imh) + v0 - r0y;
        Time time_j01 = angularVelocityIntegrator.orientations()
                            .orientations()[j_0 + 1]
                            .time;

        // set the initial and final points
        x0b = (x0b1 - x0b0) * 1.0 *
                  (time_capture_start + time_delay - time_j0) /
                  (time_j01 - time_j0) +
              x0b0;
        y0b = (y0b1 - y0b0) * 1.0 *
                  (time_capture_start + time_delay - time_j0) /
                  (time_j01 - time_j0) +
              y0b0;
      }

      // (x,y) coordinates of points, which correspond to blurring
      // alongside with angles an time
      Time *timeb = new Time[bsamples];
      float *xb = new float[bsamples];
      float *yb = new float[bsamples];
      float xbi, ybi, xbf, ybf, xbmax = 0, ybmax = 0;
      int *bstep = new int[bsamples - 1];  // steps for interpolation
      int bstepsmax = 0;

      for (int j = j_i; j < j_f + 1; j++) {
        // Thetabx[j-j_i]=Thetax[j];
        Versor<float> CurrentVersor = angularVelocityIntegrator.orientations()
                                          .orientations()[j]
                                          .orientation;
        Vector<float, 3> VectorVersor = Vector<float, 3>(x0, y0, 1.0);
        CurrentVersor.rotateVector(VectorVersor);
        x1 = VectorVersor.x() / VectorVersor.z();
        y1 = VectorVersor.y() / VectorVersor.z();
        xb[j - j_i] = x1 * (FOVx * imw) + u0 - r0x;
        yb[j - j_i] = y1 * (FOVy * imh) + v0 - r0y;
        timeb[j - j_i] =
            angularVelocityIntegrator.orientations().orientations()[j].time;
      }

      // set the initial and final points
      xbi = (xb[1] - xb[0]) * 1.0 *
                (time_capture_start + time_delay +
                 rollingShutterDuration * r0y - timeb[0]) /
                (timeb[1] - timeb[0]) +
            xb[0];
      ybi = (yb[1] - yb[0]) * 1.0 *
                (time_capture_start + time_delay +
                 rollingShutterDuration * r0y - timeb[0]) /
                (timeb[1] - timeb[0]) +
            yb[0];

      xbf = (xb[bsamples - 1] - xb[bsamples - 2]) * 1.0 *
                (time_capture_start + time_exposure + time_delay +
                 rollingShutterDuration * r0y - timeb[bsamples - 2]) /
                (timeb[bsamples - 1] - timeb[bsamples - 2]) +
            xb[bsamples - 2];
      ybf = (yb[bsamples - 1] - yb[bsamples - 2]) * 1.0 *
                (time_capture_start + time_exposure + time_delay +
                 rollingShutterDuration * r0y - timeb[bsamples - 2]) /
                (timeb[bsamples - 1] - timeb[bsamples - 2]) +
            yb[bsamples - 2];

      // set the number of steps decrease for the first and the last line
      float bstepsi =
          sqrt(((xb[1] - xbi) * (xb[1] - xbi) + (yb[1] - ybi) * (yb[1] - ybi)) /
               ((xb[1] - xb[0]) * (xb[1] - xb[0]) +
                (yb[1] - yb[0]) * (yb[1] - yb[0])));
      float bstepsf =
          sqrt(((xbf - xb[bsamples - 2]) * (xbf - xb[bsamples - 2]) +
                (ybf - yb[bsamples - 2]) * (ybf - yb[bsamples - 2])) /
               ((xb[bsamples - 1] - xb[bsamples - 2]) *
                    (xb[bsamples - 1] - xb[bsamples - 2]) +
                (yb[bsamples - 1] - yb[bsamples - 2]) *
                    (yb[bsamples - 1] - yb[bsamples - 2])));

      // calculate number of steps
      for (int j = 0; j < bsamples - 1; j++) {
        bstep[j] = int(2.0 * sqrt((xb[j + 1] - xb[j]) * (xb[j + 1] - xb[j]) +
                                  (yb[j + 1] - yb[j]) * (yb[j + 1] - yb[j])) +
                       0.5);
        if (bstepsmax < bstep[j]) bstepsmax = bstep[j];
      }

      // same number of steps for all
      for (int j = 0; j < bsamples - 1; j++) bstep[j] = bstepsmax;
      // correct the number f steps for the first and last line
      bstep[0] *= bstepsi;
      bstep[bsamples - 2] *= bstepsf;

      // the final set of points
      xb[0] = xbi - x0b;
      yb[0] = ybi - y0b;
      xb[bsamples - 1] = xbf - x0b;
      yb[bsamples - 1] = ybf - y0b;
      for (int j = 1; j < bsamples - 1; j++) {
        xb[j] -= x0b;
        yb[j] -= y0b;
        if (abs(xb[j]) > abs(xbmax)) xbmax = xb[j];
        if (abs(yb[j]) > abs(ybmax)) ybmax = yb[j];
      }
      if (abs(xb[0]) > abs(xbmax)) xbmax = xb[0];
      if (abs(yb[0]) > abs(ybmax)) ybmax = yb[0];
      if (abs(xb[bsamples - 1]) > abs(xbmax)) xbmax = xb[bsamples - 1];
      if (abs(yb[bsamples - 1]) > abs(ybmax)) ybmax = yb[bsamples - 1];

      // draw the kernel for this point
      cv::Mat kernel = cv::Mat(cv::Size(2 * int(abs(xbmax) + 0.5) + 11,
                                        2 * int(abs(ybmax) + 0.5) + 11),
                               CV_32FC1);
      kernel = 0;

      // sparse kernel
      int dims[] = {2 * int(abs(xbmax) + 0.5) + 10,
                    2 * int(abs(ybmax) + 0.5) + 10};
      cv::SparseMat skernel = cv::SparseMat(2, dims, CV_32FC1);

      // set the kernel values
      for (int j = 0; j < bsamples - 1; j++) {
        float step = 1.0 / bstepsmax;
        for (int jj = 0; jj < bstep[j]; jj++) {
          float xd =
              ((xb[j + 1] - xb[j]) * jj / bstep[j] + xb[j]) + dims[0] / 2;
          float yd =
              ((yb[j + 1] - yb[j]) * jj / bstep[j] + yb[j]) + dims[1] / 2;
          // sparse kernel
          int idx[] = {int(xd), int(yd)};
          skernel.ref<float>(idx) +=
              (int(xd) - xd + 1) * (int(yd) - yd + 1) * step;
          idx[0] = int(xd);
          idx[1] = int(yd + 1);
          skernel.ref<float>(idx) += (int(xd) - xd + 1) * (yd - int(yd)) * step;
          idx[0] = int(xd + 1);
          idx[1] = int(yd);
          skernel.ref<float>(idx) += (xd - int(xd)) * (int(yd) - yd + 1) * step;
          idx[0] = int(xd + 1);
          idx[1] = int(yd + 1);
          skernel.ref<float>(idx) += (xd - int(xd)) * (yd - int(yd)) * step;
        }
      }

      float kernelsum = 0;
      // sparse kernel sum
      cv::SparseMatIterator it;
      for (it = skernel.begin(); it != skernel.end(); ++it) {
        float val = it.value<float>();
        if (kernelsum < val) kernelsum = val;
        // kernelsum+=val;
      }

      // normalize sparse kernel
      for (it = skernel.begin(); it != skernel.end(); ++it) {
        it.value<float>() /= kernelsum;
      }

      // set the kernel from the sparse kernel
      for (it = skernel.begin(); it != skernel.end(); ++it) {
        cv::SparseMat::Node *node = it.node();
        int *idx = node->idx;
        float val = it.value<float>();
        ((float *)((kernel).data + (kernel).step * (idx[1])))[(idx[0])] = val;
      }

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

      // center of the kernel
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

      delete[] timeb;
      delete[] xb;
      delete[] yb;
      delete[] bstep;

      cv::namedWindow("Kernel", 0);
      kernel = kernel * 255;
      cv::imshow("Kernel", kernel);
      cv::waitKey(2);
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
