//============================================================================
// Name        :
// Author      : Vladislav Antonov
// Version     :
// Description : Block generation program for DFT transforms, based on:
//             : Michael Hirsch, Christian J. Schuler, Stefan Harmeling
//             : and Bernhard Scholkopf
//             : Fast Removal of Non-uniform Camera Shake
//             : Proc. IEEE International Conference on Computer Vision 2011
//             : sharp image update
// Created on  : April 10, 2012
//============================================================================

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
 * Function which introduces
 * blurring kernel on image
 * with periodic boundary conditions
 * kernel is spatially dependent
 */

void BlurrPBCsv(cv::Mat &imga, float rot[], cv::Mat &imgc, int tr = 0) {
  float s2, s3;
  int i, j;
  float xb, yb;
  int iter = 0;

  float u0 = imga.cols / 2.0, v0 = imga.rows / 2.0;
  float FOVx = u0 * 2.0, FOVy = u0 * 2.0;

  for (int row = 0; row < imga.rows; row++) {
    // cout << row << '\n';
    for (int col = 0; col < imga.cols; col++) {
      s2 = 0;

      // calculate unit vector of rotation
      float x0 = (col - u0) / (FOVx), y0 = (row - v0) / (FOVy);
      float urx = rot[0];
      float ury = rot[1];
      float urz = rot[2];
      float sur = sqrt(urx * urx + ury * ury + urz * urz);
      float c, s, cc;
      float Rt[9];

      if (sur != 0) {
        urx /= sur;
        ury /= sur;
        urz /= sur;
      } else {
        urx = 0;
        ury = 0;
        urz = 0;
      }
      c = cos(sur), s = sin(sur), cc = 1 - c;
      // inverse of the rotation matrix
      Rt[0] = urx * urx * cc + c;
      Rt[1] = urx * ury * cc - urz * s;
      Rt[2] = urx * urz * cc + ury * s;
      Rt[3] = ury * urx * cc + urz * s;
      Rt[4] = ury * ury * cc + c;
      Rt[5] = ury * urz * cc - urx * s;
      Rt[6] = urz * urx * cc - ury * s;
      Rt[7] = urz * ury * cc + urx * s;
      Rt[8] = urz * urz * cc + c;

      // here the kernel is changed
      float x1 = (Rt[0] * x0 + Rt[1] * y0 + Rt[2] * 1.0) /
                 (Rt[6] * x0 + Rt[7] * y0 + Rt[8] * 1.0);
      float y1 = (Rt[3] * x0 + Rt[4] * y0 + Rt[5] * 1.0) /
                 (Rt[6] * x0 + Rt[7] * y0 + Rt[8] * 1.0);
      xb = x1 * (FOVx) + u0 - col;
      yb = y1 * (FOVy) + v0 - row;

      // sparse kernel
      int dims[] = {2 * int(abs(xb) + 0.5) + 3, 2 * int(abs(yb) + 0.5) + 3};
      cv::SparseMat skernel = cv::SparseMat(2, dims, CV_32F);

      float bstep = int(2.0 * sqrt(xb * xb + yb * yb) + 0.5);
      float step = 1.0 / bstep;
      for (int jj = 0; jj < bstep; jj++) {
        float xd = (xb * jj / bstep) + dims[0] / 2;
        float yd = (yb * jj / bstep) + dims[1] / 2;
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

      // sparse kernel sum
      float kernelsum = 0;
      for (auto it = skernel.begin(); it != skernel.end(); ++it) {
        kernelsum += it.value<float>();
      }

      // normalize sparse kernel
      for (auto it = skernel.begin(); it != skernel.end(); ++it) {
        it.value<float>() /= kernelsum;
      }

      iter++;

      for (auto it = skernel.begin(); it != skernel.end(); ++it) {
        cv::SparseMat::Node *node = it.node();
        int *idx = node->idx;
        float val = it.value<float>();

        int row1, col1;

        // here the kernel is transposed
        if (tr == 0) {
          row1 = idx[1];
          col1 = idx[0];
        } else {
          row1 = dims[1] - 1 - idx[1];
          col1 = dims[0] - 1 - idx[0];
        }

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
       gamma * ((float *)(imgq.data))[0] * ((float *)(imgq.data))[0]);

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
 *
 */

void FDFilter(cv::Mat &imga1, cv::Mat &imgb1, cv::Mat &imgc, cv::Mat &imgq1,
              cv::Mat &imgq2, float gamma) {
  cv::Mat imga = imga1.clone();
  cv::Mat imgb = imgb1.clone();
  cv::Mat imgq = imgq1.clone();
  cv::Mat imgqb = imgq2.clone();

  // Forward Fourier Transform of initial and blurring image
  cv::dft(imga, imga);
  cv::dft(imgb, imgb);
  cv::dft(imgq, imgq);
  cv::dft(imgqb, imgqb);

  // inverts Fourier transformed blurred
  FMatInv(imgb, imgq, imgqb, gamma);

  // blurring by multiplication
  cv::mulSpectrums(imga, imgb, imgb, 0);

  // Backward Fourier Transform
  cv::dft(imgb, imgc, cv::DFT_INVERSE | cv::DFT_SCALE);
}

/*
 * This procedure sets the blurr image
 * which is DFT transformed for blurring
 * using given kernel
 * Note that the blurr image should be the same size
 * as the image to be blurred by DFT
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
 * Block deblurring procedure, using functions
 * for selecting and puting back blocks
 */

void BlockDeblur(cv::Mat &imga, cv::Mat &imgb, float rot[]) {
  cv::Mat imbl, imbbl, imblw;
  cv::Mat kernel;  // kernel for blurring
  int x, y, xw, yw, px, py, x_2, y_2;

  float u0 = imga.cols / 2.0, v0 = imga.rows / 2.0;
  float FOVx = u0 * 2.0, FOVy = u0 * 2.0;

  int blocksize = 13;
  xw = 1.0 * (blocksize);
  yw = 1.0 * (blocksize);

  for (x = 0; x < imga.cols; x += xw)
    for (y = 0; y < imga.rows; y += yw) {
      /*
       * Kernel calculation starts here
       */

      // calculate unit vector of rotation
      float x0 = (x - u0) / (FOVx), y0 = (y - v0) / (FOVy);
      float urx = rot[0];
      float ury = rot[1];
      float urz = rot[2];
      float sur = sqrt(urx * urx + ury * ury + urz * urz);
      float c, s, cc;
      float Rt[9];

      if (sur != 0) {
        urx /= sur;
        ury /= sur;
        urz /= sur;
      } else {
        urx = 0;
        ury = 0;
        urz = 0;
      }
      c = cos(sur), s = sin(sur), cc = 1 - c;
      // inverse of the rotation matrix
      Rt[0] = urx * urx * cc + c;
      Rt[1] = urx * ury * cc - urz * s;
      Rt[2] = urx * urz * cc + ury * s;
      Rt[3] = ury * urx * cc + urz * s;
      Rt[4] = ury * ury * cc + c;
      Rt[5] = ury * urz * cc - urx * s;
      Rt[6] = urz * urx * cc - ury * s;
      Rt[7] = urz * ury * cc + urx * s;
      Rt[8] = urz * urz * cc + c;

      // here the kernel is changed
      float x1 = (Rt[0] * x0 + Rt[1] * y0 + Rt[2] * 1.0) /
                 (Rt[6] * x0 + Rt[7] * y0 + Rt[8] * 1.0);
      float y1 = (Rt[3] * x0 + Rt[4] * y0 + Rt[5] * 1.0) /
                 (Rt[6] * x0 + Rt[7] * y0 + Rt[8] * 1.0);
      float xb = x1 * (FOVx) + u0 - x;
      float yb = y1 * (FOVy) + v0 - y;

      // motion blur kernel creation and definition
      kernel = cv::Mat(
          cv::Size(2 * int(abs(xb) + 0.5) + 3, 2 * int(abs(yb) + 0.5) + 3),
          CV_32FC1);
      kernel = 0;

      float bstep = int(2.0 * sqrt(xb * xb + yb * yb) + 0.5);
      float step = 1.0 / bstep;
      for (int jj = 0; jj < bstep; jj++) {
        float xd = (xb * jj / bstep) + kernel.cols / 2;
        float yd = (yb * jj / bstep) + kernel.rows / 2;
        ((float *)(kernel.data + int(yd) * (kernel.step)))[int(xd)] +=
            (int(xd) - xd + 1) * (int(yd) - yd + 1) * step;
        ((float *)(kernel.data + int(yd + 1) * (kernel.step)))[int(xd)] +=
            (int(xd) - xd + 1) * (yd - int(yd)) * step;
        ((float *)(kernel.data + int(yd) * (kernel.step)))[int(xd + 1)] +=
            (xd - int(xd)) * (int(yd) - yd + 1) * step;
        ((float *)(kernel.data + int(yd + 1) * (kernel.step)))[int(xd + 1)] +=
            (xd - int(xd)) * (yd - int(yd)) * step;
      }

      // normalize
      float kernelsum = 0;
      for (int row = 0; row < kernel.rows; row++)
        for (int col = 0; col < kernel.cols; col++)
          kernelsum += ((float *)(kernel.data + row * (kernel.step)))[col];

      for (int row = 0; row < kernel.rows; row++)
        for (int col = 0; col < kernel.cols; col++)
          ((float *)(kernel.data + row * (kernel.step)))[col] /= kernelsum;

      /*
       * Kernel calculation ends here
       */

      // block selection definition
      px = 2.0 * (2 * int(abs(xb) + 0.5) + 3);
      py = 2.0 * (2 * int(abs(yb) + 0.5) + 3);

      // optimal blocks for the DFT
      x_2 = cv::getOptimalDFTSize(xw + 2 * px);
      y_2 = cv::getOptimalDFTSize(yw + 2 * py);

      px = 0.5 * (x_2 - xw);
      py = 0.5 * (y_2 - yw);

      imbl = cv::Mat(cv::Size(x_2, y_2), CV_32FC1);
      imblw = cv::Mat(cv::Size(x_2, y_2), CV_32FC1);
      imbbl = cv::Mat(cv::Size(x_2, y_2), CV_32FC1);

      // regularization images
      cv::Mat imbll1 = cv::Mat(cv::Size(x_2, y_2), CV_32FC1);
      cv::Mat imbll2 = cv::Mat(cv::Size(x_2, y_2), CV_32FC1);
      imbll1 = 0;
      ((float *)(imbll1.data + (0) * imbll1.step))[0] = 1.0;
      imbll2 = 0;
      ((float *)(imbll2.data + (0) * imbll2.step))[0] = 1.0;
      cv::Sobel(imbll1, imbll1, -1, 1, 0);
      cv::Sobel(imbll2, imbll2, -1, 0, 1);

      // visualization of the blocks and segments
      BlockSel(imga, imbl, x, y, xw, yw, px, py);

      // ksetMoveXYb(kernel,xb,yb,0,0);
      Blurrset(imbbl, kernel);

      // applying window function
      DFTWindow(imbl);

      // block deblurring
      FDFilter(imbl, imbbl, imbl, imbll1, imbll2, 1e-5);

      // procedure for correction of block artefacts
      imblw = cv::Scalar(1.0);
      DFTWindow(imblw);
      FDFilter(imblw, imbbl, imblw, imbll1, imbll2, 1e-5);
      cv::divide(imbl, imblw, imbl);

      // put back the deblurred block
      BlockPut(imgb, imbl, x, y, xw, yw, px, py);
    }
}

int main(int argc, char *argv[]) {
  cv::Mat img, imgi, imgii, img1, img2, img3, img4,
      img5;              // initial, blurred, kernel, deblurred and noise image
  int m = 640, n = 480;  // image dimensions and radius of blurring
  // define real motion blurr
  float rot[3] = {1.0 * CV_PI / 180, 0.2 * CV_PI / 180, 0.0 * CV_PI / 180};

  // creates initial image
  if ((argc == 2) && (!(imgi = cv::imread(argv[1], 1)).empty())) {
    img = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
    if (imgi.channels() != 1) {
      imgii = cv::Mat(cv::Size(imgi.cols, imgi.rows), imgi.depth(), 1);
      cv::cvtColor(imgi, imgii, cv::COLOR_RGB2GRAY);
      imgii.convertTo(img, CV_32F);
      img = img / 255.0;
    } else {
      imgi.convertTo(img, CV_32F);
      img = img / 255.0;
    }
  } else {
    img = cv::Mat(cv::Size(m, n), CV_32FC1);
    Ximgset(img);
  }

  // creates probe image to remove block artifacts
  img1 = img.clone();
  img1 = cv::Scalar(1.0);

  // direct blurring
  img3 = img.clone();
  BlurrPBCsv(img, rot, img3);

  // probe deblurred images
  img2 = img1.clone();

  // image for reconstruction
  img4 = img.clone();
  img4 = 0;

  // kernel in space
  img5 = img.clone();
  img5 = 0;

  {
    cv::Mat imbl, imbbl, imblw;
    cv::Mat kernel;  // kernel for blurring
    int x, y, xw, yw, px, py, x_2, y_2;

    float u0 = img.cols / 2.0, v0 = img.rows / 2.0;
    float FOVx = u0 * 2.0, FOVy = u0 * 2.0;

    int blocksize = 9;
    xw = 1.0 * (blocksize);
    yw = 1.0 * (blocksize);

    for (x = 0; x < img3.cols; x += xw)
      for (y = 0; y < img3.rows; y += yw) {
        /*
         * Kernel calculation starts here
         */

        // calculate unit vector of rotation
        float x0 = (x - u0) / (FOVx), y0 = (y - v0) / (FOVy);
        float urx = rot[0];
        float ury = rot[1];
        float urz = rot[2];
        float sur = sqrt(urx * urx + ury * ury + urz * urz);
        float c, s, cc;
        float Rt[9];

        if (sur != 0) {
          urx /= sur;
          ury /= sur;
          urz /= sur;
        } else {
          urx = 0;
          ury = 0;
          urz = 0;
        }
        c = cos(sur), s = sin(sur), cc = 1 - c;
        // inverse of the rotation matrix
        Rt[0] = urx * urx * cc + c;
        Rt[1] = urx * ury * cc - urz * s;
        Rt[2] = urx * urz * cc + ury * s;
        Rt[3] = ury * urx * cc + urz * s;
        Rt[4] = ury * ury * cc + c;
        Rt[5] = ury * urz * cc - urx * s;
        Rt[6] = urz * urx * cc - ury * s;
        Rt[7] = urz * ury * cc + urx * s;
        Rt[8] = urz * urz * cc + c;

        // here the kernel is changed
        float x1 = (Rt[0] * x0 + Rt[1] * y0 + Rt[2] * 1.0) /
                   (Rt[6] * x0 + Rt[7] * y0 + Rt[8] * 1.0);
        float y1 = (Rt[3] * x0 + Rt[4] * y0 + Rt[5] * 1.0) /
                   (Rt[6] * x0 + Rt[7] * y0 + Rt[8] * 1.0);
        float xb = x1 * (FOVx) + u0 - x;
        float yb = y1 * (FOVy) + v0 - y;

        // motion blur kernel creation and definition
        kernel = cv::Mat(
            cv::Size(2 * int(abs(xb) + 0.5) + 3, 2 * int(abs(yb) + 0.5) + 3),
            CV_32FC1);
        kernel = 0;

        float bstep = int(2.0 * sqrt(xb * xb + yb * yb) + 0.5);
        float step = 1.0 / bstep;
        for (int jj = 0; jj < bstep; jj++) {
          float xd = (xb * jj / bstep) + kernel.cols / 2;
          float yd = (yb * jj / bstep) + kernel.rows / 2;
          ((float *)(kernel.data + int(yd) * (kernel.step)))[int(xd)] +=
              (int(xd) - xd + 1) * (int(yd) - yd + 1) * step;
          ((float *)(kernel.data + int(yd + 1) * (kernel.step)))[int(xd)] +=
              (int(xd) - xd + 1) * (yd - int(yd)) * step;
          ((float *)(kernel.data + int(yd) * (kernel.step)))[int(xd + 1)] +=
              (xd - int(xd)) * (int(yd) - yd + 1) * step;
          ((float *)(kernel.data + int(yd + 1) * (kernel.step)))[int(xd + 1)] +=
              (xd - int(xd)) * (yd - int(yd)) * step;
        }

        // normalize
        float kernelsum = 0;
        for (int row = 0; row < kernel.rows; row++)
          for (int col = 0; col < kernel.cols; col++)
            kernelsum += ((float *)(kernel.data + row * (kernel.step)))[col];

        for (int row = 0; row < kernel.rows; row++)
          for (int col = 0; col < kernel.cols; col++)
            ((float *)(kernel.data + row * (kernel.step)))[col] /= kernelsum;

        /*
         * Kernel calculation ends here
         */

        // block selection definition
        int ksize = max(2 * int(abs(xb) + 0.5) + 3, 2 * int(abs(yb) + 0.5) + 3);
        px = 2.0 * (2 * int(abs(xb) + 0.5) + 3);
        py = 2.0 * (2 * int(abs(yb) + 0.5) + 3);

        // optimal blocks for the DFT
        x_2 = cv::getOptimalDFTSize(xw + 2 * px);
        y_2 = cv::getOptimalDFTSize(yw + 2 * py);

        px = 0.5 * (x_2 - xw);
        py = 0.5 * (y_2 - yw);

        imbl = cv::Mat(cv::Size(x_2, y_2), img3.depth(), 1);
        imblw = cv::Mat(cv::Size(x_2, y_2), img3.depth(), 1);
        imbbl = cv::Mat(cv::Size(x_2, y_2), img3.depth(), 1);

        // regularization images
        cv::Mat imbll1 = cv::Mat(cv::Size(x_2, y_2), CV_32FC1);
        cv::Mat imbll2 = cv::Mat(cv::Size(x_2, y_2), CV_32FC1);
        imbll1 = 0;
        ((float *)(imbll1.data + (0) * imbll1.step))[0] = 1.0;
        imbll2 = 0;
        ((float *)(imbll2.data + (0) * imbll2.step))[0] = 1.0;
        cv::Sobel(imbll1, imbll1, -1, 1, 0);
        cv::Sobel(imbll2, imbll2, -1, 0, 1);

        // visualization of the blocks and segments
        BlockSel(img3, imbl, x, y, xw, yw, px, py);

        // ksetMoveXYb(kernel,xb,yb,0,0);
        Blurrset(imbbl, kernel);

        // applying window function
        DFTWindow(imbl);

        // block deblurring
        FDFilter(imbl, imbbl, imbl, imbll1, imbll2, 1e-5);

        // procedure for correction of block artefacts
        imblw = cv::Scalar(1.0);

        DFTWindow(imblw);
        FDFilter(imblw, imbbl, imblw, imbll1, imbll2, 1e-5);
        cv::divide(imbl, imblw, imbl);

        // prepare the weight mask
        BlockPut(img2, imblw, x, y, xw, yw, px, py);

        // return deblurred block
        BlockPut(img4, imbl, x, y, xw, yw, px, py);

        // cvScale(kernel,kernel,255.0);
        // cv::imshow("kernel", kernel);
        // cv::waitKey(2);
      }
  }

  // displays image
  cv::namedWindow("Initial", cv::WINDOW_AUTOSIZE);
  cv::imshow("Initial", img);
  cv::namedWindow("Probe", cv::WINDOW_AUTOSIZE);
  cv::imshow("Probe", img2);
  cv::namedWindow("Blurred", cv::WINDOW_AUTOSIZE);
  cv::imshow("Blurred", img3);
  cv::namedWindow("Deblurred", cv::WINDOW_AUTOSIZE);
  cv::imshow("Deblurred", img4);

  cv::waitKey(0);

  cv::destroyWindow("Initial");
  cv::destroyWindow("Blurred");
  cv::destroyWindow("Probe");
  cv::destroyWindow("Deblurred");

  return 0;
}
