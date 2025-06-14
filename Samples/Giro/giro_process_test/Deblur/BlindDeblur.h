/*
 * BlindDeblur.h
 *
 *  Created on: Apr 26, 2012
 *      Author: vantonov
 */

#pragma once

// OpenCV libraries
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>

// gyro kernel class
#include "GyroBlurKernel.h"

// block deblurring procedures
#include "BlockDeblur.h"

using namespace std;

/*
 * This function calculates ||A.x-b|| residual
 * img=b is the blurred image
 * BlurProc are the blur kernel parameters
 * xw, yw - size of the block
 * xp, yp - frames of the block
 */

float Residual(cv::Mat &img, GyroBlurKernel &BlurProc, int xw, int yw, int xp,
               int yp) {
  float value;

  cv::Mat imgd1 = img.clone();
  cv::Mat imgd2 = img.clone();

  // block deblurring procedure
  BlockDeblurRegul(img, imgd1, BlurProc, xw, yw, xp, yp);
  // block blurring procedure
  BlockBlur(imgd1, imgd2, BlurProc, xw, yw, xp, yp);

  cv::subtract(img, imgd2, imgd2);

  value = cv::norm(imgd2, cv::NORM_L2);

  return (value);
}

/*
 * Function which returns the residual derivative
 * of blur kernel parameter
 *
 * index of kernel variables (as in GyroBlurKernel.h):
 * 0 - TimeDelay
 * 1 - FOVx
 * 2 - FOVy
 * 3 - OptCentX
 * 4 - OptCentY
 * 5 - TimeExposure
 * 6 - TimeReadout
 */

float dRdx(cv::Mat &img, GyroBlurKernel &BlurProc, int xw, int yw, int xp,
           int yp, int index) {
  float value;

  // size of the differential step
  float deps = 1e-3;

  // initial residual
  float r0 = Residual(img, BlurProc, xw, yw, xp, yp);

  // perturbed value
  GyroBlurKernel BlurProcPert = BlurProc;

  GyroDeblurParams ParamsPert = BlurProcPert.GetBlurParameters();

  switch (index) {
    case 0:
      ParamsPert.TimeDelay *= 1 + deps;
      break;
    case 1:
      ParamsPert.FOVx *= 1 + deps;
      break;
    case 2:
      ParamsPert.FOVy *= 1 + deps;
      break;
    case 3:
      ParamsPert.OptCentX *= 1 + deps;
      break;
    case 4:
      ParamsPert.OptCentY *= 1 + deps;
      break;
    case 5:
      ParamsPert.TimeExposure *= 1 + deps;
      break;
    case 6:
      ParamsPert.TimeReadout *= 1 + deps;
      break;
    default:
      break;
  }

  // set the new blur kernel
  BlurProcPert.SetBlurParameters(ParamsPert);

  // perturbed residual
  float r1 = Residual(img, BlurProcPert, xw, yw, xp, yp);

  value = (r1 - r0) / deps;

  return (value);
}

/*
 * Function to extract the indexed blur parameter
 */

double IndexedBlurParam(GyroDeblurParams Params, int index) {
  double value;

  switch (index) {
    case 0:
      value = Params.TimeDelay;
      break;
    case 1:
      value = Params.FOVx;
      break;
    case 2:
      value = Params.FOVy;
      break;
    case 3:
      value = Params.OptCentX;  /// 1e3;
      break;
    case 4:
      value = Params.OptCentY;  /// 1e3;
      break;
    case 5:
      value = Params.TimeExposure;  /// 1e7;
      break;
    case 6:
      value = Params.TimeReadout;  /// 1e7;
      break;
    default:
      break;
  }

  return (value);
}

/*
 * Function to set the indexed blur parameter
 */

void IndexedBlurParamSet(GyroDeblurParams &Params, int index, double value) {
  switch (index) {
    case 0:
      Params.TimeDelay = value;
      break;
    case 1:
      Params.FOVx = value;
      break;
    case 2:
      Params.FOVy = value;
      break;
    case 3:
      Params.OptCentX = value;  //*1e3;
      break;
    case 4:
      Params.OptCentY = value;  //*1e3;
      break;
    case 5:
      Params.TimeExposure = value;  //*1e7;
      break;
    case 6:
      Params.TimeReadout = value;  //*1e7;
      break;
    default:
      break;
  }
}
