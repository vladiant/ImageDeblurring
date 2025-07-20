//============================================================================
// Name        :
// Author      : Vladislav Antonov
// Version     :
// Description : Block generation program for DFT transforms, based on:
//             : Michael Hirsch, Christian J. Schuler, Stefan Harmeling
//             : and Bernhard Scholkopf
//             : Fast Removal of Non-uniform Camera Shake
//             : Proc. IEEE International Conference on Computer Vision 2011
//             : testing simple blur
// Created on  : April 10, 2012
//============================================================================

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

const int amp = 15;

void Ximgset(cv::Mat &imga)  // generates the image
{
  imga = 0;
  // rectangle coordinates
  int i1 = round((imga.rows) / 5), i2 = round(3 * (imga.cols) / 5),
      j1 = round((imga.rows) / 5), j2 = round(3 * (imga.cols) / 5);

  // circle radius
  int r = round(max(imga.rows, imga.cols) / 5);

  // draws rectangle
  cv::rectangle(imga, cv::Point(i1, j1), cv::Point(i2, j2), cv::Scalar(0.5),
                -1);

  // draws circle
  cv::circle(imga,
             cv::Point(round(5 * (imga.cols) / 8), round(3 * (imga.rows) / 5)),
             r, cv::Scalar(1.0), -1);
}

/*
 * Window function: Bartlett-Hanning window
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
 * This function selects block
 * with upper left position (x,y)
 * with size (xw,yw) and frames (px, py)
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
 * This function puts back block
 * with upper left position (x,y)
 * with size (xw,yw) and frames (px, py)
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
 * This procedure sets the blurr image
 * which is DFT transformed for blurring
 * using given kernel
 * Note that the blurr image should be the same size
 * as the iage to be blurred by DFT
 */
void Blurrset(cv::Mat &imga,
              cv::Mat &krnl)  // sets the blurr image via a kernel
{
  // Get the size of the matrix.
  int rows = krnl.rows;
  int cols = krnl.cols;

  // Split the input matrix into quadrants.
  // Top-left to bottom-right and top-right to bottom-left
  int midRow = rows / 2;
  int midCol = cols / 2;

  // Top-left quadrant
  cv::Mat q0(krnl, cv::Rect(0, 0, midCol, midRow));
  // Bottom-right quadrant
  cv::Mat q1(krnl, cv::Rect(midCol, midRow, midCol + 1, midRow + 1));
  // Bottom-left quadrant
  cv::Mat q2(krnl, cv::Rect(0, midRow, midCol, midRow + 1));
  // Top-right quadrant
  cv::Mat q3(krnl, cv::Rect(midCol, 0, midCol + 1, midRow));

  // Swap quadrants
  q0.copyTo(
      imga(cv::Rect(imga.cols - midCol, imga.rows - midRow, midCol, midRow)));
  q1.copyTo(imga(cv::Rect(0, 0, midCol + 1, midRow + 1)));
  q2.copyTo(imga(cv::Rect(imga.cols - midCol, 0, midCol, midRow + 1)));
  q3.copyTo(imga(cv::Rect(0, imga.rows - midRow, midCol + 1, midRow)));
}

/*
 * This is procedure for setting of the move kernel
 * (x-x0,y-y0) defines the vector of movement
 * move was assumed with constant speed.
 */

void ksetMoveXY(cv::Mat &kernel, int x, int y, int x0 = 0, int y0 = 0) {
  cv::Scalar ksum;

  kernel = 0;
  cv::line(kernel, cv::Point(x0 + kernel.cols / 2, y0 + kernel.rows / 2),
           cv::Point(x + kernel.cols / 2, y + kernel.rows / 2), cv::Scalar(1.0),
           1, cv::LINE_AA);
  ksum = cv::sum(kernel);
  kernel = kernel / ksum.val[0];
}

/*
 * This procedure is used to return the inverse of
 * the Fourier transform matrix.
 * To solve the division by zero problem
 * and to apply Wiener filter
 * a variable of gamma is introduced:
 * it can be zero for pure inversion;
 * it can be equal to NSR for filtering;
 */
void FMatInv(cv::Mat &imgw, float gamma) {
  float a1, a2, sum;  // temporal variables for matrix inversion
  int w, h, h2, w2;   // image width and height, help variables

  w = imgw.cols;
  h = imgw.rows;

  w2 = ((w % 2 == 0) ? w - 2 : w - 1);
  h2 = ((h % 2 == 0) ? h - 2 : h - 1);

  // sets upper left
  ((float *)(imgw.data))[0] =
      ((float *)(imgw.data))[0] /
      (((float *)(imgw.data))[0] * ((float *)(imgw.data))[0] + gamma);

  // set first column
  for (int row = 1; row < h2; row += 2) {
    a1 = ((float *)(imgw.data + row * imgw.step))[0];
    a2 = ((float *)(imgw.data + (row + 1) * imgw.step))[0];
    sum = a1 * a1 + a2 * a2 + gamma;
    ((float *)(imgw.data + row * imgw.step))[0] = a1 / sum;
    ((float *)(imgw.data + (row + 1) * imgw.step))[0] = -a2 / sum;
  }

  // sets down left if needed
  if (h % 2 == 0) {
    ((float *)(imgw.data + (h - 1) * imgw.step))[0] =
        ((float *)(imgw.data + (h - 1) * imgw.step))[0] /
        (((float *)(imgw.data + (h - 1) * imgw.step))[0] *
             ((float *)(imgw.data + (h - 1) * imgw.step))[0] +
         gamma);
  }

  if (w % 2 == 0) {
    // sets upper right
    ((float *)(imgw.data))[w - 1] =
        ((float *)(imgw.data))[w - 1] /
        (((float *)(imgw.data))[w - 1] * ((float *)(imgw.data))[w - 1] + gamma);

    // set last column
    for (int row = 1; row < h2; row += 2) {
      a1 = ((float *)(imgw.data + row * imgw.step))[w - 1];
      a2 = ((float *)(imgw.data + (row + 1) * imgw.step))[w - 1];
      sum = a1 * a1 + a2 * a2 + gamma;
      ((float *)(imgw.data + row * imgw.step))[w - 1] = a1 / sum;
      ((float *)(imgw.data + (row + 1) * imgw.step))[w - 1] = -a2 / sum;
    }

    // sets down right
    if (h % 2 == 0) {
      ((float *)(imgw.data + (h - 1) * imgw.step))[w - 1] =
          ((float *)(imgw.data + (h - 1) * imgw.step))[w - 1] /
          (((float *)(imgw.data + (h - 1) * imgw.step))[w - 1] *
               ((float *)(imgw.data + (h - 1) * imgw.step))[w - 1] +
           gamma);
    }
  }

  for (int row = 0; row < h; row++) {
    for (int col = 1; col < w2; col += 2) {
      a1 = ((float *)(imgw.data + row * imgw.step))[col];
      a2 = ((float *)(imgw.data + row * imgw.step))[col + 1];
      sum = a1 * a1 + a2 * a2 + gamma;
      ((float *)(imgw.data + row * imgw.step))[col] = a1 / sum;
      ((float *)(imgw.data + row * imgw.step))[col + 1] = (-a2 / sum);
    }
  }
}

/*
 * This procedure does convolution
 * of two images via the DFT
 * Note: Periodic Border Conditions!
 */
void FCFilter(cv::Mat &imga1, cv::Mat &imgb1, cv::Mat &imgc1) {
  cv::Mat imga;
  cv::Mat imgb;
  cv::Mat imgc;

  // Forward Fourier Transform of initial and blurring image
  cv::dft(imga1, imga);
  cv::dft(imgb1, imgb);

  // blurring by multiplication
  cv::mulSpectrums(imga, imgb, imgc, 0);

  // Backward Fourier Transform
  cv::dft(imgc, imgc1, cv::DFT_INVERSE | cv::DFT_SCALE);
}

/*
 * This procedure does deconvolution
 * of two images via the DFT
 * Note: Periodic Border Conditions!
 */
void FDFilter(cv::Mat &imga1, cv::Mat &imgb1, cv::Mat &imgc,
              float gamma = 1e-7) {
  cv::Mat imga;
  cv::Mat imgb;
  cv::Mat imgd;

  // Forward Fourier Transform of initial and blurring image
  cv::dft(imga1, imga);
  cv::dft(imgb1, imgb);

  // inverts Fourier transformed blurred
  FMatInv(imgb, gamma);

  // blurring by multiplication
  cv::mulSpectrums(imga, imgb, imgd, 0);

  // Backward Fourier Transform
  cv::dft(imgd, imgc, cv::DFT_INVERSE | cv::DFT_SCALE);
}

/*
 * Block deblurring procedure, using functions
 * for selecting and puting back blocks
 */

void BlockDeblur(cv::Mat &imga, cv::Mat &imgb, int xb, int yb, float alpha) {
  cv::Mat imbl;
  int x, y, xw, yw, px, py, x_2, y_2;

  // block selection definition
  xw = 10;
  yw = 10;

  // optimal blocks for the DFT
  x_2 = 40;
  y_2 = 40;

  px = 0.5 * (x_2 - xw);
  py = 0.5 * (y_2 - yw);

  imbl = cv::Mat(cv::Size(x_2, y_2), CV_32FC1);

  // motion blur kernel
  cv::Mat kernel =
      cv::Mat(cv::Size(2 * int(abs(xb) + 0.5) + 3, 2 * int(abs(yb) + 0.5) + 3),
              CV_32FC1);
  kernel = 0;
  ksetMoveXY(kernel, xb, yb, 0, 0);

  cv::Mat imbbl = imbl.clone();
  Blurrset(imbbl, kernel);

  // correction coefficients image
  // it can vary from block to block
  cv::Mat imblw = imbl.clone();
  imblw = cv::Scalar(1.0);
  DFTWindow(imblw);
  FDFilter(imblw, imbbl, imblw, alpha);

  for (x = 0; x < imga.cols; x += xw)
    for (y = 0; y < imga.rows; y += yw) {
      // block selection
      BlockSel(imga, imbl, x, y, xw, yw, px, py);

      // applying window function
      DFTWindow(imbl);

      // deblurring
      FDFilter(imbl, imbbl, imbl, alpha);

      // correction
      cv::divide(imbl, imblw, imbl);

      // put back the deblurred block
      BlockPut(imgb, imbl, x, y, xw, yw, px, py);
    }
}

int main(int argc, char *argv[]) {
  cv::Mat img, imgi, imgb, imgd,
      imgdb;  // initials(2), blurred, block deblurred, simply deblurred
  cv::Mat kernel, imgk;  // kernel, DFT kernel
  int m = 320, n = 240;  // image dimensions
  int xb = 3, yb = 4;    // blur vector
  float alpha = 1e-6;    // regularization coefficient

  // creates initial image
  if ((argc == 2) && (!(imgi = cv::imread(argv[1], 1)).empty())) {
    img = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
    imgi.convertTo(img, CV_32F);
    img = img / 255.0;
  } else {
    img = cv::Mat(cv::Size(m, n), CV_32FC1);
    Ximgset(img);
  }

  imgb = img.clone();
  imgd = img.clone();
  imgk = img.clone();
  imgdb = img.clone();

  // direct blurring
  kernel =
      cv::Mat(cv::Size(2 * int(abs(xb) + 0.5) + 3, 2 * int(abs(yb) + 0.5) + 3),
              CV_32FC1);
  kernel = 0;
  ksetMoveXY(kernel, xb, yb, 0, 0);
  Blurrset(imgk, kernel);
  FCFilter(img, imgk, imgb);

  // add some noise
  // Gnoise(imgb, 0.0, 0.01);

  // direct deblurring
  FDFilter(imgb, imgk, imgd);

  // displays images
  cv::namedWindow("Initial", cv::WINDOW_AUTOSIZE);
  cv::imshow("Initial", img);
  cv::namedWindow("Blurred", cv::WINDOW_AUTOSIZE);
  cv::imshow("Blurred", imgb);
  cv::namedWindow("Deblurred", cv::WINDOW_AUTOSIZE);
  cv::imshow("Deblurred", imgd);

  BlockDeblur(imgb, imgdb, xb, yb, alpha);
  cout << "\nISNR  " << cv::norm(imgdb, imgd, cv::NORM_RELATIVE | cv::NORM_L2)
       << endl;

  // cvAbsDiff(imgd,imgdb,imgdb);

  cv::namedWindow("Block Deblurred", cv::WINDOW_AUTOSIZE);
  cv::imshow("Block Deblurred", imgdb);

  cv::waitKey(0);

  imgdb = imgdb * 255;
  cv::imwrite("Deblurred.tif", imgdb);

  cv::destroyWindow("Initial");
  cv::destroyWindow("Blocked");
  cv::destroyWindow("Deblurred");
  cv::destroyWindow("Block Deblurred");

  return 0;
}
