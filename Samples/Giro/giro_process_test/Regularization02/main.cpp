//============================================================================
// Name        :
// Author      : Vladislav Antonov
// Version     :
// Copyright   :
// Description : Program used to demonstrate the motion deblurring for images
//             : using a gyro data calculated kernel with RL approximation
//             : Boundaries condition: replicate
//             : Gyro data read from cin (can be given from file)
//             : Rolling shutter correction added
//             : Regularization added for noise suppression: empirical GCV
// Created on  : Mar 29, 2012
//============================================================================

#include <iostream>
#include <string>

// OpenCV libraries
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#ifndef CLOCKS_PER_SEC
#define CLOCKS_PER_SEC
#endif

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

// kernel calculation steps
const int kernel_stepx = 1;
const int kernel_stepy = 1;

bool flag = true;  // false if kernel loaded

float filt;        // Filter value
double sca = 1.0;  // scaling factor

// time for readout in seconds
const Time time_readout = timeFromSeconds<float>(0.03283);

// exposure time in seconds - to be read from EXIF Data!!!
Time time_exposure = timeFromSeconds<float>(0.05999);

// Field of view for intrinsic camera matrix K
const float FOVx = 2657.2 / 2952;  // experimentally measured
const float FOVy = 2637.0 / 1944;

// samples to be used for averaging
const int SAMPL_AVER = 20;

// from calibration data
Versor<float> gyroSpaceToCameraSpaceQuaternion =
    Versor<float>(-0.0470602, 0.698666, 0.71093, -0.0650423);
Time frameTimestampDelayMinusGyroTimestampDelay = -3.78381e+06;
bool isZAxisInvertedInGyroSpace = 1;

// initial, loaded, merged (float), blue/gray, green, red (float),  blue/gray,
// green, red (initial) images
cv::Mat imgi, img, imgc, imgc1, imgc2, imgc3, imgc1i, imgc2i, imgc3i;

// images to be blurred
cv::Mat imgcb;
// images to be deblurred
cv::Mat imgcd, imgc1d, imgc2d, imgc3d;
// images to be shown - first blurred, then deblurred
cv::Mat img1b, img3b, img1d, img3d;

cv::Mat kernel, imgb,
    krnl;  // kernel, image for blurring, one for all, kernel image

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

/*
 * This function gives axes to the elements
 * of the packed Fourier transformed matrix.
 * It was found in internet address
 * http://mildew.ee.engr.uky.edu/~weisu/OpenCV/OpenCV_archives_16852-19727.htm
 * and proposed by Vadim Pisarevsky
 */
void FGet2D(const cv::Mat &Y, int k, int i, float *re, float *im) {
  int x, y;                       // pixel coordinates of Re Y(i,k).
  float *Yptr = (float *)Y.data;  // pointer to Re Y(i,k)
  int stride = Y.step / sizeof(float);

  if (k == 0 || k * 2 == Y.cols) {
    x = (k == 0 ? 0 : Y.cols - 1);
    if (i == 0 || i * 2 == Y.rows) {
      y = i == 0 ? 0 : Y.rows - 1;
      *re = Yptr[y * stride + x];
      *im = 0;
    } else if (i * 2 < Y.rows) {
      y = i * 2 - 1;
      *re = Yptr[y * stride + x];
      *im = Yptr[(y + 1) * stride + x];
    } else {
      y = (Y.rows - i) * 2 - 1;
      *re = Yptr[y * stride + x];
      *im = Yptr[(y + 1) * stride + x];
    }
  } else if (k * 2 < Y.cols) {
    x = k * 2 - 1;
    y = i;
    *re = Yptr[y * stride + x];
    *im = Yptr[y * stride + x + 1];
  } else {
    x = (Y.cols - k) * 2 - 1;
    y = i;
    *re = Yptr[y * stride + x];
    *im = Yptr[y * stride + x + 1];
  }
}

/*
 * This function calculates
 * standard noise deviation
 * from Fourier transformed
 * image
 */

float NoiseDev(const cv::Mat &imga) {
  cv::Mat imgd(cv::Size(imga.cols, imga.rows), CV_32FC1);
  float re, im, shar, sum;

  shar = 0.8;
  sum = 0.0;
  imga.convertTo(imgd, CV_32F, 1.0 / 255.0);
  cv::dft(imgd, imgd);
  for (int row = int(shar * imgd.rows / 2); row < (imgd.rows / 2) - 1; row++) {
    for (int col = int(shar * imgd.cols / 2); col < (imgd.cols / 2) - 1;
         col++) {
      FGet2D(imgd, col, row, &re, &im);
      sum += sqrt(re * re + im * im);
    }
  }
  sum /= ((imgd.rows / 2) - int(shar * imgd.rows / 2)) *
         ((imgd.cols / 2) - int(shar * imgd.cols / 2)) *
         sqrt((imgd.rows) * (imgd.cols));
  return (sum);
}

/*
 * Function which introduces
 * blurring kernel on image
 * with reflective boundary conditions
 * kernel is spatially dependent
 */

void BlurrPBCsv(cv::Mat &imga, cv::Mat &imgc1, int Ns,
                DiscreteAngularVelocitiesIntegrator<Versor<float>, float>
                    &angularVelocityIntegrator,
                Time capture_timestamp, Time time_exposure, Time time_readout,
                Time time_delay, int fl = 0) {
  float s2, s3;
  int i, j;

  int imw = imga.cols, imh = imga.rows;  // image width, image height
  float u0 = imw / 2.0, v0 = imh / 2.0;  // principal point in pixels

  // Rolling shutter shift
  float rollingShutterDuration = time_readout / imh;

  // define the time when the capture starts
  Time time_capture_start = capture_timestamp - time_exposure - time_readout;

  // search for the time interval of exposure
  int j_i, j_f, j_0;
  bool flag1 = 1;  // initial point, final point, zero point, control flags

  // first timestamp
  TimeOrientation<Versor<float>> FirstGyroSample =
      angularVelocityIntegrator.orientations().orientations()[0];

  // zero point
  for (int j = 0; j < Ns; j++) {
    TimeOrientation<Versor<float>> TEST =
        angularVelocityIntegrator.orientations().orientations()[j];

    if ((TEST.time > time_capture_start + time_delay) && (flag1)) {
      j_0 = j - 1;
      break;
    }
  }

  cv::SparseMat skernel;     // sparse matrix for the kernel
  cv::SparseMatIterator it;  // iterator
  int dims[2];               // dimensions

  cv::Mat imgc = imgc1.clone();

  // cv::Mat& kernel1=cvCloneImage(imga);
  // cvSetZero(kernel1);

  for (int row = 0; row < imga.rows; row++) {
    for (int col = 0; col < imga.cols; col++) {
      s2 = 0;

      /*
       * Kernel calculation
       */

      // if (((row%kernel_stepy)==0)&&((col%kernel_stepx)==0))
      {
        bool flag1 = 1, flag2 = 1;  // control flags
        float x1, y1;               // current coordinates
        // search for the exposure interval
        for (int j = 0; j < Ns; j++) {
          TimeOrientation<Versor<float>> CurrentSample =
              angularVelocityIntegrator.orientations().orientations()[j];

          if ((CurrentSample.time > time_capture_start +
                                        rollingShutterDuration * row +
                                        time_delay) &&
              (flag1)) {
            j_i = j - 1;
            flag1 = 0;
          }

          if ((CurrentSample.time > time_capture_start + time_exposure +
                                        rollingShutterDuration * row +
                                        time_delay) &&
              (flag2)) {
            j_f = j;
            flag2 = 0;
            break;
          }
        }

        // number of samples for interpolation
        int bsamples = j_f - j_i + 1;

        // non homogenious coordinates
        float x0 = (row - u0) / (FOVx * imw), y0 = (col - v0) / (FOVy * imh);

        // zero point calculation
        float x0b, y0b;  // base coordinates (zero point) for kernel estimation
        float x0b0, x0b1, y0b0, y0b1;
        {
          Versor<float> CurrentVersor = angularVelocityIntegrator.orientations()
                                            .orientations()[j_0]
                                            .orientation;
          Vector<float, 3> VectorVersor = Vector<float, 3>(x0, y0, 1.0);
          CurrentVersor.rotateVector(VectorVersor);
          x1 = VectorVersor.x() / VectorVersor.z();
          y1 = VectorVersor.y() / VectorVersor.z();
          x0b0 = x1 * (FOVx * imw) + u0 - col;
          y0b0 = y1 * (FOVy * imh) + v0 - row;
          Time time_j0 =
              angularVelocityIntegrator.orientations().orientations()[j_0].time;

          CurrentVersor = angularVelocityIntegrator.orientations()
                              .orientations()[j_0 + 1]
                              .orientation;
          VectorVersor = Vector<float, 3>(x0, y0, 1.0);
          CurrentVersor.rotateVector(VectorVersor);
          x1 = VectorVersor.x() / VectorVersor.z();
          y1 = VectorVersor.y() / VectorVersor.z();
          x0b1 = x1 * (FOVx * imw) + u0 - col;
          y0b1 = y1 * (FOVy * imh) + v0 - row;
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
        Time timeb[bsamples];
        float xb[bsamples];
        float yb[bsamples];
        float xbi, ybi, xbf, ybf, xbmax = 0, ybmax = 0;
        int bstep[bsamples - 1];  // steps for interpolation
        int bstepsmax = 0;

        for (int j = j_i; j < j_f + 1; j++) {
          Versor<float> CurrentVersor = angularVelocityIntegrator.orientations()
                                            .orientations()[j]
                                            .orientation;
          Vector<float, 3> VectorVersor = Vector<float, 3>(x0, y0, 1.0);
          CurrentVersor.rotateVector(VectorVersor);
          x1 = VectorVersor.x() / VectorVersor.z();
          y1 = VectorVersor.y() / VectorVersor.z();
          xb[j - j_i] = x1 * (FOVx * imw) + u0 - col;
          yb[j - j_i] = y1 * (FOVy * imh) + v0 - row;
          timeb[j - j_i] =
              angularVelocityIntegrator.orientations().orientations()[j].time;
        }

        // set the initial and final points
        xbi = (xb[1] - xb[0]) * 1.0 *
                  (time_capture_start + time_delay +
                   rollingShutterDuration * row - timeb[0]) /
                  (timeb[1] - timeb[0]) +
              xb[0];
        ybi = (yb[1] - yb[0]) * 1.0 *
                  (time_capture_start + time_delay +
                   rollingShutterDuration * row - timeb[0]) /
                  (timeb[1] - timeb[0]) +
              yb[0];

        xbf = (xb[bsamples - 1] - xb[bsamples - 2]) * 1.0 *
                  (time_capture_start + time_exposure + time_delay +
                   rollingShutterDuration * row - timeb[bsamples - 2]) /
                  (timeb[bsamples - 1] - timeb[bsamples - 2]) +
              xb[bsamples - 2];
        ybf = (yb[bsamples - 1] - yb[bsamples - 2]) * 1.0 *
                  (time_capture_start + time_exposure + time_delay +
                   rollingShutterDuration * row - timeb[bsamples - 2]) /
                  (timeb[bsamples - 1] - timeb[bsamples - 2]) +
              yb[bsamples - 2];

        // set the number of steps decrease for the first and the last line
        float bstepsi = sqrt(
            ((xb[1] - xbi) * (xb[1] - xbi) + (yb[1] - ybi) * (yb[1] - ybi)) /
            ((xb[1] - xb[0]) * (xb[1] - xb[0]) +
             (yb[1] - yb[0]) * (yb[1] - yb[0])));
        float bstepsf =
            sqrt(((xbf - xb[bsamples - 2]) * (xbf - xb[bsamples - 2]) +
                  (ybf - yb[bsamples - 2]) * (ybf - yb[bsamples - 2])) /
                 ((xb[bsamples - 1] - xb[bsamples - 2]) *
                      (xb[bsamples - 1] - xb[bsamples - 2]) +
                  (yb[bsamples - 1] - yb[bsamples - 2]) *
                      (yb[bsamples - 1] - yb[bsamples - 2])));

        // cout << xb[bsamples-2] << "  " << xbf << "  " << xb[bsamples-1] <<
        // endl; cout << yb[bsamples-2] << "  " << ybf << "  " << yb[bsamples-1]
        // << endl;

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

        // cout << bstepsi << "  " << bstepsf << endl;

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

        // for (int j=0;j<bsamples; j++) cout << xb[j] << "  " << yb[j] << endl;

        // sparse kernel
        dims[0] = 2 * int(abs(xbmax) + 0.5) + 11;
        dims[1] = 2 * int(abs(ybmax) + 0.5) + 11;
        skernel = cv::SparseMat(2, dims, CV_32FC1);

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
            skernel.ref<float>(idx) +=
                (int(xd) - xd + 1) * (yd - int(yd)) * step;
            idx[0] = int(xd + 1);
            idx[1] = int(yd);
            skernel.ref<float>(idx) +=
                (xd - int(xd)) * (int(yd) - yd + 1) * step;
            idx[0] = int(xd + 1);
            idx[1] = int(yd + 1);
            skernel.ref<float>(idx) += (xd - int(xd)) * (yd - int(yd)) * step;
          }
        }

        float kernelsum = 0;
        // sparse kernel sum
        for (it = skernel.begin(); it != skernel.end(); ++it) {
          kernelsum += it.value<float>();
        }

        // normalize sparse kernel
        for (it = skernel.begin(); it != skernel.end(); ++it) {
          it.value<float>() /= kernelsum;
        }

        /*
         * Kernel calculation ends
         */
      }

      /*
       //set the big kernel image from sparse matrix
       if ((row%10==0)&&(col%50==0))
              for(CvSparseNode *node = cvInitSparseMatIterator( skernel, &it );
       node != 0; node = cvGetNextSparseNode( &it ))
                      {
                              int* idx = CV_NODE_IDX(skernel,node);
                              float val = *(float*)CV_NODE_VAL( skernel, node );
                              if
       ((int(row+idx[1]-dims[1]/2)<kernel1.rows)&&(int(row+idx[1]-dims[1]/2)>=0)&&(int(col+idx[0]-dims[0]/2)<kernel1.cols)&&(int(col+idx[0]-dims[0]/2)>=0))
                                      if (val>0)
                                              {
                                                              //black and white
                                                              ((float*)(kernel1.data
       +
       int(row+idx[1]-dims[1]/2)*(kernel1.step)))[int(col+idx[0]-dims[0]/2)]=val*10.0;
                                                              //cout <<
       int(row+idx[1]-dims[1]/2) << "  " << int(col+idx[0]-dims[0]/2) << "  " <<
       val*255.0 << endl;
                                              }
                      }
      */

      // apply the kernel
      for (it = skernel.begin(); it != skernel.end(); ++it) {
        cv::SparseMat::Node *node = it.node();
        int *idx = node->idx;
        float val = it.value<float>();

        int row1, col1;

        // here the kernel is transposed
        if (fl == 0) {
          row1 = idx[1];
          col1 = idx[0];
        } else {
          row1 = dims[1] - 1 - idx[1];
          col1 = dims[0] - 1 - idx[0];
        }

        // cout << row << "  " << col << "  " << row1 << "  " << col1 << endl;

        // replicate boundary conditions
        if ((row - row1 + dims[1] / 2) >= 0) {
          if ((row - row1 + dims[1] / 2) < (imga.rows)) {
            i = row - row1 + dims[1] / 2;
          } else {
            i = imga.rows - 1;
          }

        } else {
          i = 0;
        }

        if ((col - col1 + dims[0] / 2) >= 0) {
          if ((col - col1 + dims[0] / 2) < (imga.cols)) {
            j = col - col1 + dims[0] / 2;
          } else {
            j = imga.cols - 1;
          }

        } else {
          j = 0;
        }

        s3 = ((float *)(imga.data + i * imga.step))[j] * val;
        s2 += s3;
      }

      ((float *)(imgc.data + row * imgc.step))[col] = s2;
    }
  }

  /*
  cv::namedWindow("Kernel1", 0);
  cv::imshow("Kernel1", kernel1);
        while(1)  //wait till ESC is pressed
        {
    char c = cv::waitKey(2);
    if(c == 27) break;
    }
  */

  imgc.copyTo(imgc1);
}

/*
 * RL deconvolution
 */

void RLDFilter(cv::Mat &imgb, cv::Mat &imgx, int Ns,
               DiscreteAngularVelocitiesIntegrator<Versor<float>, float>
                   &angularVelocityIntegrator,
               Time capture_timestamp, Time time_exposure, Time time_readout,
               Time time_delay) {
  int it;              // iteration counter
  double norm;         // Regularization stop criteria
  float stdev, alpha;  // Standard noise deviation, regularization constant

  clock_t time_start, time_end;  // marks for start and end time

  cv::namedWindow("Processing...", 0);

  stdev = NoiseDev(imgb) * 1500 / 3.12;
  alpha = -(6.67533 * stdev * stdev + 0.00089085 * stdev + 2e-6);

  // initial approximation of the restored image
  imgb.copyTo(imgx);

  // temporal images
  cv::Mat img1 = imgb.clone();
  cv::Mat img2 = imgb.clone();

  // reset iteration counter
  it = 0;

  do {
    // time starts
    time_start = clock();

    // D.x - blurring x with D
    BlurrPBCsv(imgx, img1, Ns, angularVelocityIntegrator, capture_timestamp,
               time_exposure, time_readout, time_delay);
    // g/D.x - dividing blurred image on blurred x
    cv::divide(imgb, img1, img2);
    // Dt.(g/D.x) - blur again with transposed operator
    BlurrPBCsv(img2, img1, Ns, angularVelocityIntegrator, capture_timestamp,
               time_exposure, time_readout, time_delay, 1);

    // L.x
    cv::Laplacian(imgx, img2, -1, 5);
    // 1-alpha.L.x
    cv::addWeighted(imgx, 0.0, img2, 1.0 * alpha, 1.0, img2);
    // 1/(1-alpha.L.f)
    cv::divide(1.0, img2, img2);
    //[Dt.(g/D.x)]/(1-alpha.L.x)
    cv::multiply(img1, img2, img1);

    // complete iteration
    cv::multiply(img1, imgx, imgx);

    // stopping coefficient
    float s1 = 0, s2 = 0;
    for (int row = 0; row < imgx.rows; row++)
      for (int col = 0; col < imgx.cols; col++) {
        float a1 = ((float *)(imgx.data + row * imgx.step))[col];
        float a2 = ((float *)(img1.data + row * img1.step))[col];
        s1 += (1.0 - a2) * (1.0 - a2) * a1 * a1;
        s2 += a1 * a1;
      }
    norm = (s2 != 0) ? s1 / s2 : 0;

    // time end
    time_end = clock();

    cv::imshow("Processing...", imgx);

    char c = cv::waitKey(10);
    if (c == 27) break;

    it++;
    cout << it << "  " << norm << "  "
         << (time_end - time_start) * 1.0 / CLOCKS_PER_SEC << endl;
  } while ((norm > 1e-8));

  cv::destroyWindow("Processing...");
}

void Process(int pos, int Ns,
             DiscreteAngularVelocitiesIntegrator<Versor<float>, float>
                 &angularVelocityIntegrator,
             Time capture_timestamp, Time time_exposure, Time time_readout,
             Time time_delay);  // initial declaration of blurr function

int main(int argc, char *argv[]) {
  // time for capture start in seconds
  Time time_capture_start;

  if (argc < 2) {
    cout << "\nUsage: " << argv[0] << " <jpeg file> \n";
    return 0;
  }

  Time capture_timestamp = 0;  // capture end timestamp
  Time time_delay = frameTimestampDelayMinusGyroTimestampDelay;  // delay time

  // set number of samples for further treatment
  int N;

  // read the values without parser
  cin >> N;
  cin >> capture_timestamp;

  // timestamp, angular velocities
  float *wx = new float[N];       // angular velocity x [rad/s]
  float *wy = new float[N];       // angular velocity y [rad/s]
  float *wz = new float[N];       // angular velocity z [rad/s]
  Time *timestamp = new Time[N];  // timestamp [n

  // declare the gyro data correction class - first value is tolerance in ns,
  // second - N samples for averaging; this means the first N samples will be
  // skipped
  GyroDataCorrection<float> gyroCorrector(1000000, SAMPL_AVER);

  for (int j = 0; j < N; j++) {
    // read the values without parser
    cin >> timestamp[j] >> wx[j] >> wy[j] >> wz[j];
    if (isZAxisInvertedInGyroSpace) {
      wz[j] *= -1.0;
    }
  }

  // define the time when the capture starts
  time_capture_start = capture_timestamp - time_exposure - time_readout;

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

  // release the memory - samples are not needed anymore
  delete[] timestamp;
  delete[] wx;
  delete[] wy;
  delete[] wz;

  // creates initial image
  if (!(img = cv::imread(argv[1], cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR))
           .empty()) {
    imgi = cv::Mat(cv::Size(img.cols, img.rows), img.depth(), img.channels());
    kernel = img.clone();
    img.copyTo(imgi);

    // kernel=cvCloneImage(img);
    kernel = cv::Mat(cv::Size(img.cols, img.rows),
                     CV_8UC3);  // separate image, color kernels
    if (img.channels() > 1) {
      img.copyTo(kernel);
    } else {
      cv::cvtColor(img, kernel, cv::COLOR_GRAY2BGR);
    }

    img.copyTo(imgi);

    switch (imgi.depth()) {
      case CV_8U:
        sca = 255;
        break;

      case CV_16U:
        sca = 65535;
        break;

      case CV_32S:
        sca = 4294967295;
        break;

      default:  // unknown depth, program should go on
        sca = 1.0;
        break;
    }
  } else {
    cout << '\n' << argv[1] << " couldn't be opened.\n";
    return 0;
  }

  int imw = imgi.cols, imh = imgi.rows;  // image width, image height
  float u0 = imw / 2.0, v0 = imh / 2.0;  // principal point in pixels
  float x0b, y0b;  // base coordinates (zero point) for kernel estimation

  // image window control
  int IMG_WIN;

  if ((imw > 1080) || (imh > 720))
    IMG_WIN = 0;
  else
    IMG_WIN = cv::WINDOW_AUTOSIZE;

  if (imgi.channels() != 1) {
    cv::namedWindow("BlurredColor", IMG_WIN);
    cv::namedWindow("DeblurredColor", IMG_WIN);
    if (IMG_WIN == 0) {
      cv::resizeWindow("BlurredColor", 1080, 720);
      cv::resizeWindow("DeblurredColor", 1080, 720);
    }
  } else {
    cv::namedWindow("BlurredGray", IMG_WIN);
    cv::namedWindow("DeblurredGray", IMG_WIN);
    if (IMG_WIN == 0) {
      cv::resizeWindow("BlurredGray", 1080, 720);
      cv::resizeWindow("DeblurredGray", 1080, 720);
    }
  }

  // Rolling shutter shift
  float rollingShutterDuration = time_readout / imh;
  cout << "# Rolling shutter time shift: "
       << timeToSeconds<float>(rollingShutterDuration) << "\n\n";

  // search for the time interval of exposure
  int j_i, j_f, j_0;
  bool flag1 = 1;  // initial point, final point, zero point, control flags

  // first timestamp
  TimeOrientation<Versor<float>> FirstGyroSample =
      angularVelocityIntegrator.orientations().orientations()[0];
  // Time time_zero=FirstGyroSample.first;	//gyro capture time

  // zero point
  for (int j = 0; j < N - SAMPL_AVER - 1; j++) {
    TimeOrientation<Versor<float>> TEST =
        angularVelocityIntegrator.orientations().orientations()[j];

    if ((TEST.time > time_capture_start + time_delay) && (flag1)) {
      j_0 = j - 1;
      break;
    }
  }

  // loops to draw the kernels
  for (int r0y = 50; r0y < kernel.rows; r0y += 100)
    for (int r0x = 50; r0x < kernel.cols; r0x += 100) {
      bool flag1 = 1, flag2 = 1;  // control flags
      float x1, y1;               // current coordinates

      // search for the exposure interval
      for (int j = 0; j < N - SAMPL_AVER - 1; j++) {
        TimeOrientation<Versor<float>> CurrentSample =
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

      // sparse kernel
      int dims[] = {2 * int(abs(xbmax) + 0.5) + 11,
                    2 * int(abs(ybmax) + 0.5) + 11};
      cv::SparseMat skernel = cv::SparseMat(2, dims, CV_32F);

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
      // set the big kernel image from sparse matrix
      for (it = skernel.begin(); it != skernel.end(); ++it) {
        cv::SparseMat::Node *node = it.node();
        int *idx = node->idx;
        float val = it.value<float>();
        if ((int(r0y + idx[1] - dims[1] / 2) < kernel.rows) &&
            (int(r0y + idx[1] - dims[1] / 2) >= 0) &&
            (int(r0x + idx[0] - dims[0] / 2) < kernel.cols) &&
            (int(r0x + idx[0] - dims[0] / 2) >= 0))
          if (val > 0) {
            if (kernel.channels() > 1) {
              // red channel
              ((uchar *)(kernel.data +
                         int(r0y + idx[1] - dims[1] / 2) * (kernel.step)))
                  [int(r0x + idx[0] - dims[0] / 2) * kernel.channels() + 2] =
                      val * 255;
              ((uchar *)(kernel.data +
                         int(r0y + idx[1] - dims[1] / 2) * (kernel.step)))
                  [int(r0x + idx[0] - dims[0] / 2) * kernel.channels() + 1] =
                      0;  // green
              ((uchar *)(kernel.data +
                         int(r0y + idx[1] - dims[1] / 2) * (kernel.step)))
                  [int(r0x + idx[0] - dims[0] / 2) * kernel.channels() + 0] =
                      0;  // blue
            } else {
              // black and white
              ((uchar *)(kernel.data +
                         int(r0y + idx[1] - dims[1] / 2) *
                             (kernel.step)))[int(r0x + idx[0] - dims[0] / 2)] =
                  val * 255;
            }
          }
      }

      // center of the kernel
      if (kernel.channels() > 1) {
        ((uchar *)(kernel.data +
                   int(r0y) *
                       (kernel.step)))[int(r0x) * kernel.channels() + 2] = 0;
        ((uchar *)(kernel.data +
                   int(r0y) *
                       (kernel.step)))[int(r0x) * kernel.channels() + 1] = 255;
        ((uchar *)(kernel.data +
                   int(r0y) *
                       (kernel.step)))[int(r0x) * kernel.channels() + 0] = 255;
      } else
        ((uchar *)(kernel.data + int(r0y) * (kernel.step)))[int(r0x)] = 255;

      delete[] timeb;
      delete[] xb;
      delete[] yb;
      delete[] bstep;
    }

  // this is the kernel image for blurring
  imgb = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);

  // the initial splitting of channels
  imgc1 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
  // creation of channel images for deblurring
  imgc1d = cv::Mat(cv::Size(imgb.cols, imgb.rows), CV_32FC1);

  if (imgi.channels() != 1) {
    imgc1i = cv::Mat(cv::Size(imgi.cols, imgi.rows), imgi.depth(), 1);
    imgc2i = cv::Mat(cv::Size(imgi.cols, imgi.rows), imgi.depth(), 1);
    imgc3i = cv::Mat(cv::Size(imgi.cols, imgi.rows), imgi.depth(), 1);
    imgc2 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
    imgc3 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
    cv::split(imgi, std::vector{imgc1i, imgc2i, imgc3i});
    imgc1i.convertTo(imgc1, CV_32F);
    imgc1 = imgc1 / sca;
    imgc2i.convertTo(imgc2, CV_32F);
    imgc2 = imgc2 / sca;
    imgc3i.convertTo(imgc3, CV_32F);
    imgc3 = imgc3 / sca;
    // image to present the cropped blurr
    img3b = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC3);
    // creation of channel images for deblurring
    imgc2d = cv::Mat(cv::Size(imgb.cols, imgb.rows), CV_32FC1);
    imgc3d = cv::Mat(cv::Size(imgb.cols, imgb.rows), CV_32FC1);
    // image to present the cropped deblurred image
    img3d = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC3);
  } else {
    // image to present the cropped blurr
    img1b = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
    // image to present the cropped deblurred
    img1d = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
    imgi.convertTo(imgc1, CV_32F);
    imgc1 = imgc1 / sca;
  }

  cv::namedWindow("Kernel", IMG_WIN);
  if (IMG_WIN == 0) cv::resizeWindow("Kernel", 1080, 720);

  Process(0, N - SAMPL_AVER - 1, angularVelocityIntegrator, capture_timestamp,
          time_exposure, time_readout, time_delay);

  while (1)  // wait till ESC is pressed
  {
    char c = cv::waitKey(0);
    if (c == 27) break;
  }

  cv::destroyWindow("Kernel");

  if (imgi.channels() != 1) {
    cv::destroyWindow("BlurredColor");
    cv::destroyWindow("DeblurredColor");
  } else {
    cv::destroyWindow("BlurredGray");
    cv::destroyWindow("DeblurredGray");
  }

  return (0);
}

void Process(int pos, int Ns,
             DiscreteAngularVelocitiesIntegrator<Versor<float>, float>
                 &angularVelocityIntegrator,
             Time capture_timestamp, Time time_exposure, Time time_readout,
             Time time_delay) {
  cv::imshow("Kernel", kernel);

  if (imgi.channels() != 1) {
    // merge and show
    cv::merge(std::vector{imgc1, imgc2, imgc3}, img3b);

    cv::imshow("BlurredColor", imgi);

    RLDFilter(imgc1, imgc1d, Ns, angularVelocityIntegrator, capture_timestamp,
              time_exposure, time_readout, time_delay);
    RLDFilter(imgc2, imgc2d, Ns, angularVelocityIntegrator, capture_timestamp,
              time_exposure, time_readout, time_delay);
    RLDFilter(imgc3, imgc3d, Ns, angularVelocityIntegrator, capture_timestamp,
              time_exposure, time_readout, time_delay);

    // merge and show
    cv::merge(std::vector{imgc1d, imgc2d, imgc3d}, img3d);

    cv::imshow("DeblurredColor", img3d);
    img3d = img3d * 255;
    cv::imwrite("Deblurred.tif", img3d);
  } else {
    cv::imshow("BlurredGray", imgc1);

    // deblur
    RLDFilter(imgc1, imgc1d, Ns, angularVelocityIntegrator, capture_timestamp,
              time_exposure, time_readout, time_delay);

    cv::imshow("DeblurredGray", imgc1d);

    imgc1d = imgc1d * 255;
    cv::imwrite("Deblurredbw.tif", imgc1d);
  }
}
