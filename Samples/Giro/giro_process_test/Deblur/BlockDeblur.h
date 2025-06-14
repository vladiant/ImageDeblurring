/*
 * BlockDeblur.h
 *
 *  Created on: Apr 24, 2012
 *      Author: vantonov
 */

#pragma once

// OpenCV libraries
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>

// gyro kernel class
#include "GyroBlurKernel.h"

/*
 * Window function: Bartlett-Hanning window
 * Transforms directly the input image
 */

void DFTWindow(cv::Mat &imga) {
  for (int row = 0; row < imga.rows; row++) {
    float s1 = 0.62 - 0.48 * abs((1.0 * row / (imga.rows - 1)) - 0.5) +
               0.38 * cos(2 * CV_PI * ((1.0 * row / (imga.rows - 1)) - 0.5));
    for (int col = 0; col < imga.cols; col++) {
      float s2 = ((float *)(imga.data + row * imga.step))[col];
      s2 *= s1;
      s2 *= 0.62 - 0.48 * abs((1.0 * col / (imga.cols - 1)) - 0.5) +
            0.38 * cos(2 * CV_PI * ((1.0 * col / (imga.cols - 1)) - 0.5));
      ((float *)(imga.data + row * imga.step))[col] = s2;
    }
  }
}

/*
 * This function selects block from imga
 * with upper left position (x,y), size (xw,yw)
 * and frames (px, py). The block is written in imgb
 */

void BlockSel(cv::Mat &imga, cv::Mat &imgb, int x, int y, int x1, int y1,
              int px, int py) {
  int x2 = (x1 + 2 * px);
  int y2 = (y1 + 2 * py);
  int xmin = (2 * x - x2 + x1) / 2;
  int xmax = (2 * x + x2 + x1) / 2;
  int ymin = (2 * y - y2 + y1) / 2;
  int ymax = (2 * y + y2 + y1) / 2;

  for (int row = ymin; row < ymax; row++)
    for (int col = xmin; col < xmax; col++) {
      int i, j;

      // replicate boundary conditions
      if ((row) >= 0) {
        if ((row) < (imga.rows)) {
          i = row;
        } else {
          i = imga.rows - 1;
        }

      } else {
        i = 0;
      }

      if ((col) >= 0) {
        if ((col) < (imga.cols)) {
          j = col;
        } else {
          j = imga.cols - 1;
        }

      } else {
        j = 0;
      }

      float s2 = ((float *)(imga.data + i * imga.step))[j];
      ((float *)(imgb.data + (row - ymin) * imgb.step))[col - xmin] = s2;
    }
}

/*
 * This function puts back block image imga, selected by previous procedure
 * The block is with upper left position (x,y), size (xw,yw) and frames (px, py)
 */

void BlockPut(cv::Mat &imga, cv::Mat &imgb, int x, int y, int x1, int y1,
              int px, int py) {
  for (int row = y, row1 = py; row < std::min(y + y1, imga.rows); row++, row1++)
    for (int col = x, col1 = px; col < std::min(x + x1, imga.cols);
         col++, col1++) {
      float s2 = ((float *)(imgb.data + row1 * imgb.step))[col1];
      ((float *)(imga.data + (row)*imga.step))[col] = s2;
    }
}

/*
 * This procedure is used to return the inverse of
 * the Fourier transform matrix,
 * regularized as defined in paper:
 *
 * D. Krishnan, R. Fergus.
 * Fast Image Deconvolution using Hyper-Laplacian Priors
 * Neural Information Processing Systems 2009
 *
 * imgq, imgq1 - regularization operator, applied
 * on initial image imga1
 */

void FMatInv(cv::Mat &imgw, cv::Mat &imgq, cv::Mat &imgq1, float gamma) {
  float a1, a2, a3, a4, a5, a6, sum;  // temporal variables for matrix inversion
  int w, h, h2, w2;                   // image width and height, help variables

  w = imgw.cols;
  h = imgw.rows;

  w2 = ((w % 2 == 0) ? w - 2 : w - 1);
  h2 = ((h % 2 == 0) ? h - 2 : h - 1);

  // sets upper left
  ((float *)(imgw.data))[0] =
      ((float *)(imgw.data))[0] /
      (((float *)(imgw.data))[0] * ((float *)(imgw.data))[0] +
       gamma * ((float *)(imgq.data))[0] * ((float *)(imgq.data))[0] +
       gamma * ((float *)(imgq1.data))[0] * ((float *)(imgq1.data))[0]);

  // set first column
  for (int row = 1; row < h2; row += 2) {
    a1 = ((float *)(imgw.data + row * imgw.step))[0];
    a2 = ((float *)(imgw.data + (row + 1) * imgw.step))[0];

    a3 = ((float *)(imgq.data + row * imgq.step))[0];
    a4 = ((float *)(imgq.data + (row + 1) * imgq.step))[0];

    a5 = ((float *)(imgq1.data + row * imgq1.step))[0];
    a6 = ((float *)(imgq1.data + (row + 1) * imgq1.step))[0];

    sum = a1 * a1 + a2 * a2 + gamma * (a3 * a3 + a4 * a4 + a5 * a5 + a6 * a6);

    ((float *)(imgw.data + row * imgw.step))[0] = a1 / sum;
    ((float *)(imgw.data + (row + 1) * imgw.step))[0] = -a2 / sum;
  }

  // sets down left if needed
  if (h % 2 == 0) {
    ((float *)(imgw.data + (h - 1) * imgw.step))[0] =
        ((float *)(imgw.data + (h - 1) * imgw.step))[0] /
        (((float *)(imgw.data + (h - 1) * imgw.step))[0] *
             ((float *)(imgw.data + (h - 1) * imgw.step))[0] +
         gamma * ((float *)(imgq.data + (h - 1) * imgq.step))[0] *
             ((float *)(imgq.data + (h - 1) * imgq.step))[0] +
         gamma * ((float *)(imgq1.data + (h - 1) * imgq1.step))[0] *
             ((float *)(imgq1.data + (h - 1) * imgq1.step))[0]);
  }

  if (w % 2 == 0) {
    // sets upper right
    ((float *)(imgw.data))[w - 1] =
        ((float *)(imgw.data))[w - 1] /
        (((float *)(imgw.data))[w - 1] * ((float *)(imgw.data))[w - 1] +
         gamma * ((float *)(imgq.data))[w - 1] * ((float *)(imgq.data))[w - 1] +
         gamma * ((float *)(imgq1.data))[w - 1] *
             ((float *)(imgq1.data))[w - 1]);

    // set last column
    for (int row = 1; row < h2; row += 2) {
      a1 = ((float *)(imgw.data + row * imgw.step))[w - 1];
      a2 = ((float *)(imgw.data + (row + 1) * imgw.step))[w - 1];

      a3 = ((float *)(imgq.data + row * imgq.step))[w - 1];
      a4 = ((float *)(imgq.data + (row + 1) * imgq.step))[w - 1];

      a5 = ((float *)(imgq1.data + row * imgq1.step))[w - 1];
      a6 = ((float *)(imgq1.data + (row + 1) * imgq1.step))[w - 1];

      sum = a1 * a1 + a2 * a2 + gamma * (a3 * a3 + a4 * a4 + a5 * a5 + a6 * a6);

      ((float *)(imgw.data + row * imgw.step))[w - 1] = a1 / sum;
      ((float *)(imgw.data + (row + 1) * imgw.step))[w - 1] = -a2 / sum;
    }

    // sets down right
    if (h % 2 == 0) {
      ((float *)(imgw.data + (h - 1) * imgw.step))[w - 1] =
          ((float *)(imgw.data + (h - 1) * imgw.step))[w - 1] /
          (((float *)(imgw.data + (h - 1) * imgw.step))[w - 1] *
               ((float *)(imgw.data + (h - 1) * imgw.step))[w - 1] +
           gamma * ((float *)(imgq.data + (h - 1) * imgq.step))[w - 1] *
               ((float *)(imgq.data + (h - 1) * imgq.step))[w - 1] +
           gamma * ((float *)(imgq1.data + (h - 1) * imgq1.step))[w - 1] *
               ((float *)(imgq1.data + (h - 1) * imgq1.step))[w - 1]);
    }
  }

  for (int row = 0; row < h; row++) {
    for (int col = 1; col < w2; col += 2) {
      a1 = ((float *)(imgw.data + row * imgw.step))[col];
      a2 = ((float *)(imgw.data + row * imgw.step))[col + 1];

      a3 = ((float *)(imgq.data + row * imgq.step))[col];
      a4 = ((float *)(imgq.data + row * imgq.step))[col + 1];

      a5 = ((float *)(imgq1.data + row * imgq1.step))[col];
      a6 = ((float *)(imgq1.data + row * imgq1.step))[col + 1];

      sum = a1 * a1 + a2 * a2 + gamma * (a3 * a3 + a4 * a4 + a5 * a5 + a6 * a6);

      ((float *)(imgw.data + row * imgw.step))[col] = a1 / sum;
      ((float *)(imgw.data + row * imgw.step))[col + 1] = (-a2 / sum);
    }
  }
}

/*
 * This procedure does deconvolution
 * of two images via the DFT
 * with regularization
 * as defined in paper:
 *
 * D. Krishnan, R. Fergus.
 * Fast Image Deconvolution using Hyper-Laplacian Priors
 * Neural Information Processing Systems 2009
 *
 * imgq1, imgq2 - regularization operators
 * applied on initial image imga1
 * to obtain deconvolved image imgc
 * with convolution kernel imgb1
 *
 * imgv1, imgv2 - weight images,
 * obtained from the gradients of the
 * imga imga1
 *
 */

void FDFilter(cv::Mat &imga1, cv::Mat &imgb1, cv::Mat &imgc, cv::Mat &imgv1,
              cv::Mat &imgv2, cv::Mat &imgq1, cv::Mat &imgq2, float gamma) {
  cv::Mat imga = imga1.clone();
  cv::Mat imgb = imgb1.clone();

  // Forward Fourier Transform of initial and blurring image
  cv::dft(imga, imga);

  // inverts Fourier transformed blurred
  FMatInv(imgb, imgq1, imgq2, gamma);

  // blurring by multiplication
  imga = imgv1 + imga;
  imga = imgv2 + imga;

  cv::mulSpectrums(imga, imgb, imgb, 0);

  // Backward Fourier Transform
  cv::dft(imgb, imgc, cv::DFT_INVERSE | cv::DFT_SCALE);
}

/*
 * Function to calculate the weight from the
 * filtered image using given factor betha
 * based on:
 *
 * D. Krishnan, R. Fergus.
 * Fast Image Deconvolution using Hyper-Laplacian Priors
 * Neural Information Processing Systems 2009
 *
 */

float WeightCalc(float x, float betha) {
  float value = 0;  // return value
  float m = 8.0 / (27 * betha * betha * betha);
  float t1 = 0.25 * x * x;
  float t2 = 27.0 * x * x * x * x * m * m - 256.0 * m * m * m;
  if (t2 < 0) return (value);
  float t3, t4, t5, t6;
  if (m != 0) {
    t3 = 9.0 * x * x * m;
    t4 = exp(log(sqrt(3 * t2) + t3) / 3.0);
    t5 = t4 * 0.381571414;
    t6 = 3.494321859 * m / t4;
  } else {
    t3 = 0;
    t4 = 0;
    t5 = 0;
    t6 = 0;
  }

  float t7 = sqrt(t1 + t5 + t6);
  float t8 = (x != 0) ? x * x * x * 0.25 / t7 : 0;

  float det1 = 2 * t1 - t5 - t6 + t8;
  float det2 = det1 - 2 * t8;

  float r1, r2, r3, r4, r;

  float c1 = abs(x) / 2.0, c2 = abs(x);

  if (det1 >= 0) {
    r3 = 0.75 * x + 0.5 * (-t7 - sqrt(det1));
    r4 = 0.75 * x + 0.5 * (-t7 + sqrt(det1));
    r = std::max(r3, r4);

    if (det2 >= 0) {
      r1 = 0.75 * x + 0.5 * (t7 - sqrt(det2));
      r2 = 0.75 * x + 0.5 * (t7 + sqrt(det2));
      r = std::max(r, r1);
      r = std::max(r, r2);
    }

    if ((abs(r) >= c1) && (abs(r) <= c2))
      value = r;
    else
      value = 0;
  } else {
    value = 0;
  }

  return (value);
}

/*
 * This procedure sets the blur image
 * which is DFT transformed for blurring
 * using given kernel
 * Note that the blur image should be the same size
 * as the image to be blurred by DFT
 */

void BlurSet(cv::Mat &imga,
             cv::Mat &krnl)  // sets the blurr image via a kernel
{
  cv::Mat tmp;

  imga = 0;  // nullifies
  tmp = imga(cv::Rect(0, 0, 1 + krnl.cols / 2, 1 + krnl.rows / 2));
  tmp.copyTo(krnl(cv::Rect(krnl.cols / 2, krnl.rows / 2, 1 + krnl.cols / 2,
                           1 + krnl.rows / 2)));

  tmp = imga(
      cv::Rect(imga.cols - krnl.cols / 2, 0, krnl.cols / 2, 1 + krnl.rows / 2));
  tmp.copyTo(
      krnl(cv::Rect(0, krnl.rows / 2, krnl.cols / 2, 1 + krnl.rows / 2)));

  tmp = imga(cv::Rect(0, imga.rows - 1 - krnl.rows / 2, krnl.cols / 2 + 1,
                      krnl.rows / 2));
  tmp.copyTo(
      krnl(cv::Rect(krnl.cols / 2, 0, krnl.cols / 2 + 1, krnl.rows / 2)));

  tmp = imga(cv::Rect(imga.cols - 1 - krnl.cols / 2, imga.rows - krnl.rows / 2,
                      krnl.cols / 2, krnl.rows / 2));
  tmp.copyTo(krnl(cv::Rect(0, 0, krnl.cols / 2, krnl.rows / 2)));
}

/*
 * Block deblurring procedure, using functions
 * for selecting and putting back blocks
 */

void BlockDeblur(cv::Mat &imga, cv::Mat &imgb, cv::Mat &imge1, cv::Mat &imge2,
                 float beth, GyroBlurKernel &BlurData, int xw, int yw, int px1,
                 int py1) {
  // processed block, block with blur kernel, block for correction of the
  // windowing artifacts
  cv::Mat imbl, imbbl, imblw;
  // blocks for regularization operators - (x and y) weight and gradient based
  cv::Mat imblw1, imblw2, imbll1, imbll2;
  // current coordinates, size of treated block
  int x, y, x_2, y_2;

  // regularization weight
  float alpha = 1e-6;

  // image window control
  int IMG_WIN;

  if ((imga.cols > 1080) || (imga.rows > 720))
    IMG_WIN = 0;
  else
    IMG_WIN = cv::WINDOW_AUTOSIZE;
  cv::namedWindow("Processing...", IMG_WIN);

  // optimal blocks for the DFT
  // x_2=cvGetOptimalDFTSize(xw+2*px1);
  // y_2=cvGetOptimalDFTSize(yw+2*py1);

  x_2 = 1;
  do {
    x_2 *= 2;
  } while (x_2 < (xw + 2 * px1));
  y_2 = 1;
  do {
    y_2 *= 2;
  } while (y_2 < (yw + 2 * py1));

  int px = 0.5 * (x_2 - xw);
  int py = 0.5 * (y_2 - yw);

  // processed block image
  imbl = cv::Mat(cv::Size(x_2, y_2), CV_32FC1);

  // blurred block image
  imbbl = cv::Mat(cv::Size(x_2, y_2), CV_32FC1);

  // image for correction of the block artifacts
  imblw = cv::Mat(cv::Size(x_2, y_2), CV_32FC1);

  // weight block images
  imblw1 = cv::Mat(cv::Size(x_2, y_2), CV_32FC1);
  imblw2 = cv::Mat(cv::Size(x_2, y_2), CV_32FC1);

  // regularization images
  imbll1 = cv::Mat(cv::Size(x_2, y_2), CV_32FC1);
  imbll2 = cv::Mat(cv::Size(x_2, y_2), CV_32FC1);

  // initialization of regularization images
  imbll1 = 0;
  ((float *)(imbll1.data + (0) * imbll1.step))[0] = 1.0;
  cv::Sobel(imbll1, imbll1, -1, 1, 0);
  cv::dft(imbll1, imbll1);

  imbll2 = 0;
  ((float *)(imbll2.data + (0) * imbll2.step))[0] = 1.0;
  cv::Sobel(imbll2, imbll2, -1, 0, 1);
  cv::dft(imbll2, imbll2);

  for (x = 0; x < imga.cols; x += xw)
    for (y = 0; y < imga.rows; y += yw) {
      // kernel calculated at the center of the block
      cv::Mat kernel = BlurData.Kernel(x + xw / 2, y + yw / 2);

      // weight images
      BlockSel(imge1, imblw1, x, y, xw, yw, px, py);
      cv::dft(imblw1, imblw1);
      cv::mulSpectrums(imblw1, imbll1, imblw1, 0);
      imblw1 = imblw1 * alpha * beth;

      BlockSel(imge2, imblw2, x, y, xw, yw, px, py);
      cv::dft(imblw2, imblw2);
      cv::mulSpectrums(imblw2, imbll2, imblw2, 0);
      imblw2 = imblw2 * alpha * beth;

      // get the block to be processed
      BlockSel(imga, imbl, x, y, xw, yw, px, py);

      // set the kernel
      BlurSet(imbbl, kernel);
      cv::dft(imbbl, imbbl);

      // applying window function
      DFTWindow(imbl);

      // block deblurring
      FDFilter(imbl, imbbl, imbl, imblw1, imblw2, imbll1, imbll2, alpha * beth);

      // procedure for correction of block artifacts
      imblw = cv::Scalar(1.0);
      DFTWindow(imblw);
      FDFilter(imblw, imbbl, imblw, imblw1, imblw2, imbll1, imbll2,
               alpha * beth);
      cv::divide(imbl, imblw, imbl);

      // puts back the deblurred block
      BlockPut(imgb, imbl, x, y, xw, yw, px, py);

      cv::imshow("Processing...", imgb);
      cv::waitKey(2);
    }
  cv::destroyWindow("Processing...");
}

/*
 * This procedure is used to return the inverse of
 * the Fourier transform matrix,
 * regularized by a given operator
 *
 * imgq1 - regularization operator, applied
 * on initial image imga1
 */
void FMatInvReg(cv::Mat &imgw, cv::Mat &imgq, float gamma) {
  float a1, a2, a3, a4, sum;  // temporal variables for matrix inversion
  int w, h, h2, w2;           // image width and height, help variables

  w = imgw.cols;
  h = imgw.rows;

  w2 = ((w % 2 == 0) ? w - 2 : w - 1);
  h2 = ((h % 2 == 0) ? h - 2 : h - 1);

  // sets upper left
  ((float *)(imgw.data))[0] =
      ((float *)(imgw.data))[0] /
      (((float *)(imgw.data))[0] * ((float *)(imgw.data))[0] +
       gamma * ((float *)(imgq.data))[0] * ((float *)(imgq.data))[0]);

  // set first column
  for (int row = 1; row < h2; row += 2) {
    a1 = ((float *)(imgw.data + row * imgw.step))[0];
    a2 = ((float *)(imgw.data + (row + 1) * imgw.step))[0];
    a3 = ((float *)(imgq.data + row * imgq.step))[0];
    a4 = ((float *)(imgq.data + (row + 1) * imgq.step))[0];
    sum = a1 * a1 + a2 * a2 + gamma * (a3 * a3 + a4 * a4);
    ((float *)(imgw.data + row * imgw.step))[0] = a1 / sum;
    ((float *)(imgw.data + (row + 1) * imgw.step))[0] = -a2 / sum;
  }

  // sets down left if needed
  if (h % 2 == 0) {
    ((float *)(imgw.data + (h - 1) * imgw.step))[0] =
        ((float *)(imgw.data + (h - 1) * imgw.step))[0] /
        (((float *)(imgw.data + (h - 1) * imgw.step))[0] *
             ((float *)(imgw.data + (h - 1) * imgw.step))[0] +
         gamma * ((float *)(imgq.data + (h - 1) * imgq.step))[0] *
             ((float *)(imgq.data + (h - 1) * imgq.step))[0]);
  }

  if (w % 2 == 0) {
    // sets upper right
    ((float *)(imgw.data))[w - 1] =
        ((float *)(imgw.data))[w - 1] /
        (((float *)(imgw.data))[w - 1] * ((float *)(imgw.data))[w - 1] +
         gamma * ((float *)(imgq.data))[w - 1] * ((float *)(imgq.data))[w - 1]);

    // set last column
    for (int row = 1; row < h2; row += 2) {
      a1 = ((float *)(imgw.data + row * imgw.step))[w - 1];
      a2 = ((float *)(imgw.data + (row + 1) * imgw.step))[w - 1];
      a3 = ((float *)(imgq.data + row * imgq.step))[w - 1];
      a4 = ((float *)(imgq.data + (row + 1) * imgq.step))[w - 1];
      sum = a1 * a1 + a2 * a2 + gamma * (a3 * a3 + a4 * a4);
      ((float *)(imgw.data + row * imgw.step))[w - 1] = a1 / sum;
      ((float *)(imgw.data + (row + 1) * imgw.step))[w - 1] = -a2 / sum;
    }

    // sets down right
    if (h % 2 == 0) {
      ((float *)(imgw.data + (h - 1) * imgw.step))[w - 1] =
          ((float *)(imgw.data + (h - 1) * imgw.step))[w - 1] /
          (((float *)(imgw.data + (h - 1) * imgw.step))[w - 1] *
               ((float *)(imgw.data + (h - 1) * imgw.step))[w - 1] +
           gamma * ((float *)(imgq.data + (h - 1) * imgq.step))[w - 1] *
               ((float *)(imgq.data + (h - 1) * imgq.step))[w - 1]);
    }
  }

  for (int row = 0; row < h; row++) {
    for (int col = 1; col < w2; col += 2) {
      a1 = ((float *)(imgw.data + row * imgw.step))[col];
      a2 = ((float *)(imgw.data + row * imgw.step))[col + 1];
      a3 = ((float *)(imgq.data + row * imgq.step))[col];
      a4 = ((float *)(imgq.data + row * imgq.step))[col + 1];
      sum = a1 * a1 + a2 * a2 + gamma * (a3 * a3 + a4 * a4);
      ((float *)(imgw.data + row * imgw.step))[col] = a1 / sum;
      ((float *)(imgw.data + row * imgw.step))[col + 1] = (-a2 / sum);
    }
  }
}

/*
 * This procedure does deconvolution
 * of two images via the DFT
 * with regularization
 *
 * imgq1 - regularization operator, applied
 * on initial image imga1 to obtain
 * deconvolved image imgc via convolution
 * operator imgb1
 *
 */

void FDFilterReg(cv::Mat &imga1, cv::Mat &imgb, cv::Mat &imgc, cv::Mat &imgq,
                 float gamma) {
  cv::Mat imga = imga1.clone();

  // Forward Fourier Transform of initial image
  cv::dft(imga, imga);

  // blurring by multiplication
  cv::mulSpectrums(imga, imgb, imga, 0);

  // Backward Fourier Transform
  cv::dft(imga, imgc, cv::DFT_INVERSE | cv::DFT_SCALE);
}

/*
 * Block deblurring procedure, using functions
 * for selecting and putting back blocks
 * Regularized with laplacian only
 */

void BlockDeblurRegul(cv::Mat &imga, cv::Mat &imgb, GyroBlurKernel &BlurData,
                      int xw, int yw, int px1, int py1) {
  // processed block, block with blur kernel, block for correction of the
  // windowing artifacts
  cv::Mat imbl, imbbl, imblw;
  // blocks for regularization operators
  cv::Mat imbll;
  // current coordinates, size of treated block
  int x, y, x_2, y_2;

  // regularization weight
  float alpha = 1e-3;

  // image window control
  int IMG_WIN;

  if ((imga.cols > 1080) || (imga.rows > 720))
    IMG_WIN = 0;
  else
    IMG_WIN = cv::WINDOW_AUTOSIZE;
  cv::namedWindow("Processing Deblur ...", IMG_WIN);

  // optimal blocks for the DFT
  // x_2=cvGetOptimalDFTSize(xw+2*px1);
  // y_2=cvGetOptimalDFTSize(yw+2*py1);

  x_2 = 1;
  do {
    x_2 *= 2;
  } while (x_2 < (xw + 2 * px1));
  y_2 = 1;
  do {
    y_2 *= 2;
  } while (y_2 < (yw + 2 * py1));

  int px = 0.5 * (x_2 - xw);
  int py = 0.5 * (y_2 - yw);

  // processed block image
  imbl = cv::Mat(cv::Size(x_2, y_2), CV_32FC1);

  // blurred block image
  imbbl = cv::Mat(cv::Size(x_2, y_2), CV_32FC1);

  // image for correction of the block artifacts
  imblw = cv::Mat(cv::Size(x_2, y_2), CV_32FC1);

  // regularization image
  imbll = cv::Mat(cv::Size(x_2, y_2), CV_32FC1);

  // initialization of regularization image
  imbll = 0;
  ((float *)(imbll.data + (0) * imbll.step))[0] = 1.0;
  cv::Laplacian(imbll, imbll, -1);
  cv::dft(imbll, imbll);

  for (x = 0; x < imga.cols; x += xw)
    for (y = 0; y < imga.rows; y += yw) {
      // kernel calculated at the center of the block
      cv::Mat kernel = BlurData.Kernel(x + xw / 2, y + yw / 2);

      // get the block to be processed
      BlockSel(imga, imbl, x, y, xw, yw, px, py);

      // set the kernel
      BlurSet(imbbl, kernel);
      cv::dft(imbbl, imbbl);
      // regularized inversion Fourier transformed
      FMatInvReg(imbbl, imbll, alpha);

      // applying window function
      DFTWindow(imbl);

      // block deblurring
      FDFilterReg(imbl, imbbl, imbl, imbll, alpha);

      // procedure for correction of block artifacts
      imblw = cv::Scalar(1.0);
      DFTWindow(imblw);
      FDFilterReg(imblw, imbbl, imblw, imbll, alpha);
      cv::divide(imbl, imblw, imbl);

      // puts back the deblurred block
      BlockPut(imgb, imbl, x, y, xw, yw, px, py);

      cv::imshow("Processing Deblur ...", imgb);
      cv::waitKey(2);
    }

  cv::destroyWindow("Processing Deblur ...");
}

/*
 * Block blurring procedure, using functions
 * for selecting and putting back blocks
 */

void BlockBlur(cv::Mat &imga, cv::Mat &imgb, GyroBlurKernel &BlurData, int xw,
               int yw, int px1, int py1) {
  // processed block, block with blur kernel
  cv::Mat imbl, imbbl;
  // current coordinates, size of treated block
  int x, y, x_2, y_2;

  // image window control
  int IMG_WIN;

  if ((imga.cols > 1080) || (imga.rows > 720))
    IMG_WIN = 0;
  else
    IMG_WIN = cv::WINDOW_AUTOSIZE;
  cv::namedWindow("Processing Blur ...", IMG_WIN);

  // optimal blocks for the DFT
  // x_2=cvGetOptimalDFTSize(xw+2*px1);
  // y_2=cvGetOptimalDFTSize(yw+2*py1);

  x_2 = 1;
  do {
    x_2 *= 2;
  } while (x_2 < (xw + 2 * px1));
  y_2 = 1;
  do {
    y_2 *= 2;
  } while (y_2 < (yw + 2 * py1));

  int px = 0.5 * (x_2 - xw);
  int py = 0.5 * (y_2 - yw);

  // processed block image
  imbl = cv::Mat(cv::Size(x_2, y_2), CV_32FC1);

  // blurred block image
  imbbl = cv::Mat(cv::Size(x_2, y_2), CV_32FC1);

  for (x = 0; x < imga.cols; x += xw)
    for (y = 0; y < imga.rows; y += yw) {
      // kernel calculated at the center of the block
      cv::Mat kernel = BlurData.Kernel(x + xw / 2, y + yw / 2);

      // get the block to be processed
      BlockSel(imga, imbl, x, y, xw, yw, px, py);

      // set the kernel
      BlurSet(imbbl, kernel);
      cv::dft(imbbl, imbbl);

      // block blurring
      cv::dft(imbl, imbl);
      cv::mulSpectrums(imbl, imbbl, imbl, 0);
      cv::dft(imbl, imbl, cv::DFT_INVERSE | cv::DFT_SCALE);

      // puts back the deblurred block
      BlockPut(imgb, imbl, x, y, xw, yw, px, py);

      cv::imshow("Processing Blur ...", imgb);
      cv::waitKey(2);
    }

  cv::destroyWindow("Processing Blur ...");
}
