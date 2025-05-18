/*
 * Fourier DeBlurring ver 1.0
 *
 * This library is to be used with the DFT approach
 * to deblurr and denoise images.
 *
 * Implementation is only for float numbers,
 * so the images must be rescaled
 *
 * Also it work only with one image plane,
 * so color images must be treated plane by plane
 *
 * Read the description of the functions
 * for more information.
 *
 * Created by Vladislav Antonov
 * October 2011
 */

#include "fdb.hpp"

void FCFilter(cv::Mat& imga1, cv::Mat& imgb1, cv::Mat& imgc1) {
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

void FMatInv(cv::Mat& imgw, float gamma) {
  float a1, a2, sum;  // temporal variables for matrix inversion
  int w, h, h2, w2;   // image width and height, help variables

  w = imgw.cols;
  h = imgw.rows;

  w2 = ((w % 2 == 0) ? w - 2 : w - 1);
  h2 = ((h % 2 == 0) ? h - 2 : h - 1);

  // sets upper left
  ((float*)(imgw.data))[0] =
      ((float*)(imgw.data))[0] /
      (((float*)(imgw.data))[0] * ((float*)(imgw.data))[0] + gamma);

  // set first column
  for (int row = 1; row < h2; row += 2) {
    a1 = ((float*)(imgw.data + row * imgw.step))[0];
    a2 = ((float*)(imgw.data + (row + 1) * imgw.step))[0];
    sum = a1 * a1 + a2 * a2 + gamma;
    ((float*)(imgw.data + row * imgw.step))[0] = a1 / sum;
    ((float*)(imgw.data + (row + 1) * imgw.step))[0] = -a2 / sum;
  }

  // sets down left if needed
  if (h % 2 == 0) {
    ((float*)(imgw.data + (h - 1) * imgw.step))[0] =
        ((float*)(imgw.data + (h - 1) * imgw.step))[0] /
        (((float*)(imgw.data + (h - 1) * imgw.step))[0] *
             ((float*)(imgw.data + (h - 1) * imgw.step))[0] +
         gamma);
  }

  if (w % 2 == 0) {
    // sets upper right
    ((float*)(imgw.data))[w - 1] =
        ((float*)(imgw.data))[w - 1] /
        (((float*)(imgw.data))[w - 1] * ((float*)(imgw.data))[w - 1] + gamma);

    // set last column
    for (int row = 1; row < h2; row += 2) {
      a1 = ((float*)(imgw.data + row * imgw.step))[w - 1];
      a2 = ((float*)(imgw.data + (row + 1) * imgw.step))[w - 1];
      sum = a1 * a1 + a2 * a2 + gamma;
      ((float*)(imgw.data + row * imgw.step))[w - 1] = a1 / sum;
      ((float*)(imgw.data + (row + 1) * imgw.step))[w - 1] = -a2 / sum;
    }

    // sets down right
    if (h % 2 == 0) {
      ((float*)(imgw.data + (h - 1) * imgw.step))[w - 1] =
          ((float*)(imgw.data + (h - 1) * imgw.step))[w - 1] /
          (((float*)(imgw.data + (h - 1) * imgw.step))[w - 1] *
               ((float*)(imgw.data + (h - 1) * imgw.step))[w - 1] +
           gamma);
    }
  }

  for (int row = 0; row < h; row++) {
    for (int col = 1; col < w2; col += 2) {
      a1 = ((float*)(imgw.data + row * imgw.step))[col];
      a2 = ((float*)(imgw.data + row * imgw.step))[col + 1];
      sum = a1 * a1 + a2 * a2 + gamma;
      ((float*)(imgw.data + row * imgw.step))[col] = a1 / sum;
      ((float*)(imgw.data + row * imgw.step))[col + 1] = (-a2 / sum);
    }
  }
}

void FDFilter(cv::Mat& imga1, cv::Mat& imgb1, cv::Mat& imgc, float gamma) {
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

void Blurrset(cv::Mat& imga,
              cv::Mat& krnl)  // sets the blurr image via a kernel
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

// cv::copyMakeBorder
void MirrorBorder(cv::Mat& src, cv::Mat& ibmat, int px, int py) {
  cv::Mat tmp, tmp1;
  cv::Mat ibmat0 = src.clone();
  // sets matrix to zero
  ibmat = 0;

  // put image in the center
  ibmat0.copyTo(ibmat(cv::Rect(px, py, src.cols, src.rows)));

  // left
  ibmat0(cv::Rect(0, 0, px, ibmat0.rows)).copyTo(tmp1);
  cv::flip(tmp1, tmp1, 1);  // flip horizontal
  tmp1.copyTo(ibmat(cv::Rect(0, py, px, ibmat0.rows)));

  // right
  ibmat0(cv::Rect((ibmat0.cols) - 1 - (px), 0, px + 1, ibmat0.rows))
      .copyTo(tmp1);
  cv::flip(tmp1, tmp1, 1);  // flip horizontal
  tmp1.copyTo(
      ibmat(cv::Rect((ibmat.cols) - 1 - (px), py, px + 1, ibmat0.rows)));

  // up
  ibmat0(cv::Rect(0, 0, ibmat0.cols, py)).copyTo(tmp1);
  cv::flip(tmp1, tmp1, 0);  // flip vertical
  tmp1.copyTo(ibmat(cv::Rect(px, 0, ibmat0.cols, py)));

  // down
  ibmat0(cv::Rect(0, (ibmat0.rows) - 1 - (py), ibmat0.cols, py + 1))
      .copyTo(tmp1);
  cv::flip(tmp1, tmp1, 0);  // flip horizontal
  tmp1.copyTo(
      ibmat(cv::Rect(px, (ibmat.rows) - 1 - (py), ibmat0.cols, py + 1)));

  // ul edge
  ibmat0(cv::Rect(0, 0, px, py)).copyTo(tmp1);
  cv::flip(tmp1, tmp1, -1);  // flip double
  tmp1.copyTo(ibmat(cv::Rect(0, 0, px, py)));

  // ur edge
  ibmat0(cv::Rect((ibmat0.cols) - 1 - (px), 0, px + 1, py)).copyTo(tmp1);
  cv::flip(tmp1, tmp1, -1);  // flip double
  tmp1.copyTo(ibmat(cv::Rect((ibmat.cols) - 1 - (px), 0, px + 1, py)));

  // dl edge
  ibmat0(cv::Rect(0, (ibmat0.rows) - 1 - (py), px, py + 1)).copyTo(tmp1);
  cv::flip(tmp1, tmp1, -1);  // flip double
  tmp1.copyTo(ibmat(cv::Rect(0, (ibmat.rows) - 1 - (py), px, py + 1)));

  // dr edge
  ibmat0(cv::Rect((ibmat0.cols) - 1 - (px), (ibmat0.rows) - 1 - (py), px + 1,
                  py + 1))
      .copyTo(tmp1);
  cv::flip(tmp1, tmp1, -1);  // flip double
  tmp1.copyTo(ibmat(cv::Rect((ibmat.cols) - 1 - (px), (ibmat.rows) - 1 - (py),
                             px + 1, py + 1)));
}

void FGet2D(const cv::Mat& Y, int k, int i, float* re, float* im) {
  int x, y;                      // pixel coordinates of Re Y(i,k).
  float* Yptr = (float*)Y.data;  // pointer to Re Y(i,k)
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

void FSet2D(cv::Mat& Y, int k, int i, float* re, float* im) {
  int x, y;                      // pixel coordinates of Re Y(i,k).
  float* Yptr = (float*)Y.data;  // pointer to Re Y(i,k)
  int stride = Y.step / sizeof(float);

  if (k == 0 || k * 2 == Y.cols) {
    x = k == 0 ? 0 : Y.cols - 1;
    if (i == 0 || i * 2 == Y.rows) {
      y = i == 0 ? 0 : Y.rows - 1;
      Yptr[y * stride + x] = *re;
    } else if (i * 2 < Y.rows) {
      y = i * 2 - 1;
      Yptr[y * stride + x] = *re;
      Yptr[(y + 1) * stride + x] = *im;
    } else {
      y = (Y.rows - i) * 2 - 1;
      Yptr[y * stride + x] = *re;
      Yptr[(y + 1) * stride + x] = *im;
    }
  } else if (k * 2 < Y.cols) {
    x = k * 2 - 1;
    y = i;
    Yptr[y * stride + x] = *re;
    Yptr[y * stride + x + 1] = *im;
  } else {
    x = (Y.cols - k) * 2 - 1;
    y = i;
    Yptr[y * stride + x] = *re;
    Yptr[y * stride + x + 1] = *im;
  }
}

void EdgeTaper(cv::Mat& imga, int px, int py) {
  cv::Mat rw, cl, rw0, cl0, rw1, cl1, rw2, cl2, rw3, cl3, rw4, cl4;

  for (int col = 0; col < px; col++) {
    imga(cv::Rect(col, 0, 1, imga.rows - 1)).copyTo(cl1);
    imga(cv::Rect(imga.cols - 1 - col, 0, 1, imga.rows - 1)).copyTo(cl2);
    imga(cv::Rect(col, 0, 1, imga.rows - 1)).copyTo(cl);
    cl3 = cl.clone();
    cl4 = cl.clone();
    cv::addWeighted(cl1, 1.0 * col / (px - 1), cl2, 0.0, 0.0, cl3);
    cv::addWeighted(cl2, 1.0 * col / (px - 1), cl1, 0.0, 0.0, cl4);
    cl3.copyTo(imga(cv::Rect(col, 0, 1, imga.rows - 1)));
    cl4.copyTo(imga(cv::Rect(imga.cols - 1 - col, 0, 1, imga.rows - 1)));
  }

  for (int row = 0; row < py; row++) {
    imga(cv::Rect(0, row, imga.cols - 1, 1)).copyTo(cl1);
    imga(cv::Rect(0, imga.rows - 1 - row, imga.cols - 1, 1)).copyTo(cl2);
    imga(cv::Rect(0, row, imga.cols - 1, 1)).copyTo(cl);
    cl3 = cl.clone();
    cl4 = cl.clone();
    cv::addWeighted(cl1, 1.0 * row / (px - 1), cl2, 0.0, 0.0, cl3);
    cv::addWeighted(cl2, 1.0 * row / (px - 1), cl1, 0.0, 0.0, cl4);
    cl3.copyTo(imga(cv::Rect(0, row, imga.cols - 1, 1)));
    cl4.copyTo(imga(cv::Rect(0, imga.rows - 1 - row, imga.cols - 1, 1)));
  }
}
