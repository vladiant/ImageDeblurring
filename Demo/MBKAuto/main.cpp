/*
 * This is a demo program
 * used to demonstrate the
 * motion blur for images
 * using a given kernel
 * and automatic noise filtering
 *
 * Created by Vladislav Antonov
 * February 2012
 */

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

#include "blurrf.hpp"
#include "fdb.hpp"
#include "noise.hpp"

using namespace std;

const int MAX_R = 20;      // maximal length of motion blurr
const float regul = 2500;  // regularization coefficient

// initial, loaded, merged (float), blue/gray, green, red (float),  blue/gray,
// green, red (initial) images
cv::Mat imgi, img, imgc, imgc1, imgc2, imgc3, imgc1i, imgc2i, imgc3i;

// images to be blurred
cv::Mat imgcb, imgc1b, imgc2b, imgc3b;
// images to be deblurred
cv::Mat imgcd, imgc1d, imgc2d, imgc3d;
// images to be shown - first blurred, then deblurred
cv::Mat img1b, img3b, img1d, img3d;

cv::Mat kernel, imgb,
    krnl;  // kernel, image for blurring, one for all, kernel image

bool flag = true;  // false if kernel loaded

int pos_Theta = 10, pos_r = 10;  // Angle and radius of motion blurr (initial)
int pos_f = 6;                   // Initial position of the filter
float xb1 = -10.5, yb1 = -20.5;  // Define vector of motion blurr
float filt;                      // Filter value
double sca = 1.0;                // scaling factor
int ksize;                       // max kernel size
float stdev;                     // Fourier standard deviation

cv::Mat tmp, tmp1;  // temporal matrix

// this function generates a simple test image
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
  imga.convertTo(imgd, CV_32F);
  imgd = imgd / 255.0;
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
 * This procedure does deconvolution
 * of two images via the DFT
 * with regularization
 * Note: Periodic Border Conditions!
 */

void FDFilter(cv::Mat &imga1, cv::Mat &imgb1, cv::Mat &imgc, cv::Mat &imgq,
              float gamma) {
  cv::Mat imga = imga1.clone();
  cv::Mat imgb = imgb1.clone();
  imga1.convertTo(imga, CV_32F);
  imgb1.convertTo(imgb, CV_32F);

  // Forward Fourier Transform of initial and blurring image
  cv::dft(imga, imga);
  cv::dft(imgb, imgb);

  // inverts Fourier transformed blurred
  FMatInv(imgb, gamma);

  // blurring by multiplication
  cv::mulSpectrums(imga, imgb, imgb, 0);

  // Backward Fourier Transform
  cv::dft(imgb, imgc, cv::DFT_INVERSE | cv::DFT_SCALE);
}

/*
 * Gradient operator
 */

void GradNorm(cv::Mat &imga, cv::Mat &imgb) {
  float s1, s2;
  int x1, x2, y1, y2;

  for (int row = 0; row < imga.rows; row++) {
    for (int col = 0; col < imga.cols; col++) {
      if (row == (imga.rows - 1)) {
        y1 = row;

      } else {
        y1 = row + 1;
      }

      if (col == (imga.cols - 1)) {
        x2 = col;
      } else {
        x2 = col + 1;
      }

      if (row == 0) {
        y2 = 0;

      } else {
        y2 = row - 1;
      }

      if (col == 0) {
        x1 = col;
      } else {
        x1 = col - 1;
      }

      s1 = ((float *)(imga.data + row * imga.step))[x2] -
           ((float *)(imga.data + row * imga.step))[x1];
      s2 = ((float *)(imga.data + y2 * imga.step))[col] -
           ((float *)(imga.data + y1 * imga.step))[col];
      ((float *)(imgb.data + row * imgb.step))[col] = abs(s1) + abs(s2);
    }
  }
}

/*
 * Hyper Laplacian operator
 */

void HLNorm(cv::Mat &imga, cv::Mat &imgb) {
  float s1, s2, s3 = 0;
  int x1, x2, y1, y2;

  for (int row = 0; row < imga.rows; row++) {
    for (int col = 0; col < imga.cols; col++) {
      if (row == (imga.rows - 1)) {
        y1 = row;

      } else {
        y1 = row + 1;
      }

      if (col == (imga.cols - 1)) {
        x2 = col;
      } else {
        x2 = col + 1;
      }

      if (row == 0) {
        y2 = 0;

      } else {
        y2 = row - 1;
      }

      if (col == 0) {
        x1 = col;
      } else {
        x1 = col - 1;
      }

      s1 = ((float *)(imga.data + row * imga.step))[x2] -
           ((float *)(imga.data + row * imga.step))[x1];
      s2 = ((float *)(imga.data + y2 * imga.step))[col] -
           ((float *)(imga.data + y1 * imga.step))[col];
      ((float *)(imgb.data + row * imgb.step))[col] =
          sqrt(pow(abs(s1), 0.8) + pow(abs(s2), 0.8));
    }
  }
}

/*
 * Total variation operator
 */

void TotVar(cv::Mat &imga, cv::Mat &imgb) {
  float s1, s2;
  int x1, x2, y1, y2;

  for (int row = 0; row < imga.rows; row++) {
    for (int col = 0; col < imga.cols; col++) {
      if (row == (imga.rows - 1)) {
        y1 = row;

      } else {
        y1 = row + 1;
      }

      if (col == (imga.cols - 1)) {
        x2 = col;
      } else {
        x2 = col + 1;
      }

      if (row == 0) {
        y2 = 0;

      } else {
        y2 = row - 1;
      }

      if (col == 0) {
        x1 = col;
      } else {
        x1 = col - 1;
      }

      s1 = ((float *)(imga.data + row * imga.step))[x2] -
           ((float *)(imga.data + row * imga.step))[x1];
      s2 = ((float *)(imga.data + y2 * imga.step))[col] -
           ((float *)(imga.data + y1 * imga.step))[col];
      ((float *)(imgb.data + row * imgb.step))[col] = sqrt(s1 * s1 + s2 * s2);
    }
  }
}

void Process(int pos, void *);  // initial declaration of blurr function

int main(int argc, char *argv[]) {
  int i;

  // creates initial image
  if ((argc == 3) &&
      (!(img = cv::imread(argv[1], cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR))
            .empty()) &&
      (!(krnl = cv::imread(argv[2], cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE))
            .empty())) {
    imgi = cv::Mat(cv::Size(img.cols, img.rows), img.depth(), img.channels());
    kernel = cv::Mat(cv::Size(krnl.cols, krnl.rows), CV_32FC1);
    img.copyTo(imgi);
    flag = false;

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
    }

    krnl.convertTo(kernel, CV_32F);
    switch (krnl.depth()) {
      case CV_8U:
        kernel = kernel / 255;
        break;

      case CV_16U:
        kernel = kernel / 65535;
        break;

      case CV_32S:
        kernel = kernel / 4294967295;
        break;

      default:  // unknown depth, program should go on
        kernel = kernel;
    }

    // kernel normalization
    float sm = 0;
    for (int row = 0; row < kernel.rows; row++)
      for (int col = 0; col < kernel.cols; col++) {
        sm += ((float *)(kernel.data + row * kernel.step))[col];
      }

    kernel = kernel / sm;

    ksize = max(kernel.cols, kernel.rows);

  } else {
    sca = 1.0;
    int m = 640, n = 480;  // VGA size
    imgi = cv::Mat(cv::Size(m, n), CV_32FC1);
    Ximgset(imgi);

    // cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    // imgi = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);
    // img.convertTo(imgi, CV_32F);
    // imgi = imgi / 255;

    // createst simple kernel for motion
    ksize = MAX_R;
    kernel = cv::Mat(cv::Size(2 * ksize + 1, 2 * ksize + 1), CV_32FC1);
  }

  char *nmR, *nmG, *nmB;  // names for the red, green and blue channel file

  // displays image
  if (flag) {
    cv::namedWindow("Initial", cv::WINDOW_AUTOSIZE);
    cv::imshow("Initial", imgi);
  }

  if (imgi.channels() != 1) {
    cv::namedWindow("BlurredColor", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("DeblurredColor", cv::WINDOW_AUTOSIZE);
  } else {
    cv::namedWindow("BlurredGray", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("DeblurredGray", cv::WINDOW_AUTOSIZE);
  }

  // this is the kernel image for blurring
  imgb =
      cv::Mat(cv::Size(imgi.cols + 2 * ksize, imgi.rows + 2 * ksize), CV_32FC1);

  // the initial splitting of channels
  imgc1 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
  // creation of channel images for blurring
  imgc1b = cv::Mat(cv::Size(imgb.cols, imgb.rows), CV_32FC1);
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
    // creation of channel images for blurring
    imgc2b = cv::Mat(cv::Size(imgb.cols, imgb.rows), CV_32FC1);
    imgc3b = cv::Mat(cv::Size(imgb.cols, imgb.rows), CV_32FC1);
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

  cv::namedWindow("Kernel", cv::WINDOW_NORMAL);

  if (flag) {
    cv::namedWindow("Motion Blurr", cv::WINDOW_NORMAL);
    cv::resizeWindow("Motion Blurr", 600, 1);
    cv::createTrackbar("Radius, pix", "Motion Blurr", &pos_r, MAX_R, Process);
    cv::createTrackbar("Angle, deg ", "Motion Blurr", &pos_Theta, 360, Process);
  }

  Process(0, nullptr);

  while (1)  // wait till ESC is pressed
  {
    char c = cv::waitKey(0);
    if (c == 27) break;
  }

  if (flag) {
    cv::destroyWindow("Initial");
    cv::destroyWindow("Motion Blurr");
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

void Process(int pos, void *) {
  if (flag) {
    // motion blurr kernel preparation
    xb1 = pos_r * cos(pos_Theta * CV_PI / 180);
    yb1 = pos_r * sin(pos_Theta * CV_PI / 180);
    // motion blurr
    if ((xb1 != 0) || (yb1 != 0)) {
      ksetMoveXYb(kernel, (xb1), (yb1));
    } else {
      kernel = 0;
      ((float *)(kernel.data + int(kernel.cols / 2) *
                                   (kernel.step)))[int(kernel.rows / 2)] = 1.0;
    }
  }

  // creates Fourier smearing image
  imgb = 0;
  Blurrset(imgb, kernel);

  // impose border conditions
  cv::copyMakeBorder(imgc1, imgc1b, ksize, ksize, ksize, ksize,
                     cv::BORDER_REPLICATE);
  EdgeTaper(imgc1b, ksize, ksize);

  if (imgi.channels() != 1) {
    cv::copyMakeBorder(imgc2, imgc2b, ksize, ksize, ksize, ksize,
                       cv::BORDER_REPLICATE);
    EdgeTaper(imgc2b, ksize, ksize);
    cv::copyMakeBorder(imgc3, imgc3b, ksize, ksize, ksize, ksize,
                       cv::BORDER_REPLICATE);
    EdgeTaper(imgc3b, ksize, ksize);
  }

  if (imgi.channels() != 1) {
    if (flag) {
      // blurring - each channel separately
      FCFilter(imgc1b, imgb, imgc1b);
      FCFilter(imgc2b, imgb, imgc2b);
      FCFilter(imgc3b, imgb, imgc3b);
    }

    // merge and show
    imgcb = cv::Mat(cv::Size(imgb.cols, imgb.rows), CV_32FC3);
    cv::merge(std::vector{imgc1b, imgc2b, imgc3b}, imgcb);
    // crop the borders
    // imgcb(cv::Rect(kernel.cols, kernel.rows,
    // imgi.cols, imgi.rows)).copyTo(tmp);
    imgcb(cv::Rect(ksize, ksize, imgi.cols, imgi.rows)).copyTo(tmp);
    tmp.copyTo(img3b);
    cv::imshow("BlurredColor", img3b);

    // deblurring - each channel separately
    if (flag) {
      FDFilter(imgc1b, imgb, imgc1d);
      FDFilter(imgc2b, imgb, imgc2d);
      FDFilter(imgc3b, imgb, imgc3d);
    } else {
      imgcd = cv::Mat(cv::Size(imgb.cols, imgb.rows), CV_32FC3);
      cv::Mat imgcl(cv::Size(imgb.cols, imgb.rows), CV_32FC1);

      // |x| norm GCV
      stdev = NoiseDev(imgc1b);
      stdev *= regul / 3.12;
      filt = sqrt(14.709 * stdev * stdev + 0.241801 * stdev + 1e-6);
      FDFilter(imgc1b, imgb, imgc1d, imgc1b, filt * filt);
      stdev = NoiseDev(imgc2b);
      stdev *= regul / 3.12;
      filt = sqrt(14.709 * stdev * stdev + 0.241801 * stdev + 1e-6);
      FDFilter(imgc2b, imgb, imgc2d, imgc2b, filt * filt);
      stdev = NoiseDev(imgc3b);
      stdev *= regul / 3.12;
      filt = sqrt(14.709 * stdev * stdev + 0.241801 * stdev + 1e-6);
      FDFilter(imgc3b, imgb, imgc3d, imgc3b, filt * filt);
      cv::merge(std::vector{imgc1d, imgc2d, imgc3d}, imgcd);
      imgcd(cv::Rect(ksize, ksize, imgi.cols, imgi.rows)).copyTo(tmp1);
      tmp1.copyTo(img3d);
      img3d = img3d / 255;
      cv::imwrite("Deblurred_1.png", img3d);

      // |grad x| norm GCV
      stdev = NoiseDev(imgc1b);
      stdev *= regul / 3.12;
      GradNorm(imgc1b, imgcl);
      filt = sqrt(14.8053 * stdev * stdev + 0.0383666 * stdev + 7.87368e-5);
      FDFilter(imgc1b, imgb, imgc1d, imgcl, filt * filt);
      stdev = NoiseDev(imgc2b);
      stdev *= regul / 3.12;
      GradNorm(imgc2b, imgcl);
      filt = sqrt(14.8053 * stdev * stdev + 0.0383666 * stdev + 7.87368e-5);
      FDFilter(imgc2b, imgb, imgc2d, imgcl, filt * filt);
      stdev = NoiseDev(imgc3b);
      stdev *= regul / 3.12;
      GradNorm(imgc3b, imgcl);
      filt = sqrt(14.8053 * stdev * stdev + 0.0383666 * stdev + 7.87368e-5);
      FDFilter(imgc3b, imgb, imgc3d, imgcl, filt * filt);
      cv::merge(std::vector{imgc1d, imgc2d, imgc3d}, imgcd);
      imgcd(cv::Rect(ksize, ksize, imgi.cols, imgi.rows)).copyTo(tmp1);
      tmp1.copyTo(img3d);
      img3d = img3d * 255;
      cv::imwrite("Deblurred_2.png", img3d);

      // Laplace norm GCV
      stdev = NoiseDev(imgc1b);
      stdev *= regul / 3.12;
      cv::Laplacian(imgc1b, imgcl, -1);
      filt = sqrt(6.67533 * stdev * stdev + 0.00089085 * stdev + 2e-6);
      FDFilter(imgc1b, imgb, imgc1d, imgcl, filt * filt);
      stdev = NoiseDev(imgc2b);
      stdev *= regul / 3.12;
      cv::Laplacian(imgc2b, imgcl, -1);
      filt = sqrt(6.67533 * stdev * stdev + 0.00089085 * stdev + 2e-6);
      FDFilter(imgc2b, imgb, imgc2d, imgcl, filt * filt);
      stdev = NoiseDev(imgc3b);
      stdev *= regul / 3.12;
      cv::Laplacian(imgc3b, imgcl, -1);
      filt = sqrt(6.67533 * stdev * stdev + 0.00089085 * stdev + 2e-6);
      FDFilter(imgc3b, imgb, imgc3d, imgcl, filt * filt);
      cv::merge(std::vector{imgc1d, imgc2d, imgc3d}, imgcd);
      imgcd(cv::Rect(ksize, ksize, imgi.cols, imgi.rows)).copyTo(tmp1);
      tmp1.copyTo(img3d);
      img3d = img3d * 255;
      cv::imwrite("Deblurred_3.png", img3d);

      // HLapl norm GCV
      stdev = NoiseDev(imgc1b);
      stdev *= regul / 3.12;
      HLNorm(imgc1b, imgcl);
      filt = sqrt(3.80424 * stdev * stdev - 0.00449728 * stdev + 2.14891e-5);
      FDFilter(imgc1b, imgb, imgc1d, imgcl, filt * filt);
      stdev = NoiseDev(imgc2b);
      stdev *= regul / 3.12;
      HLNorm(imgc2b, imgcl);
      filt = sqrt(3.80424 * stdev * stdev - 0.00449728 * stdev + 2.14891e-5);
      FDFilter(imgc2b, imgb, imgc2d, imgcl, filt * filt);
      stdev = NoiseDev(imgc3b);
      stdev *= regul / 3.12;
      HLNorm(imgc3b, imgcl);
      filt = sqrt(3.80424 * stdev * stdev - 0.00449728 * stdev + 2.14891e-5);
      FDFilter(imgc3b, imgb, imgc3d, imgcl, filt * filt);
      cv::merge(std::vector{imgc1d, imgc2d, imgc3d}, imgcd);
      imgcd(cv::Rect(ksize, ksize, imgi.cols, imgi.rows)).copyTo(tmp1);
      tmp1.copyTo(img3d);
      img3d = img3d * 255;
      cv::imwrite("Deblurred_4.png", img3d);

      // TV norm GCV
      stdev = NoiseDev(imgc1b);
      stdev *= regul / 3.12;
      TotVar(imgc1b, imgcl);
      filt = sqrt(23.9335 * stdev * stdev + 0.0705829 * stdev + 5.72816e-5);
      FDFilter(imgc1b, imgb, imgc1d, imgcl, filt * filt);
      stdev = NoiseDev(imgc2b);
      stdev *= regul / 3.12;
      TotVar(imgc2b, imgcl);
      filt = sqrt(23.9335 * stdev * stdev + 0.0705829 * stdev + 5.72816e-5);
      FDFilter(imgc2b, imgb, imgc2d, imgcl, filt * filt);
      stdev = NoiseDev(imgc3b);
      stdev *= regul / 3.12;
      TotVar(imgc3b, imgcl);
      filt = sqrt(23.9335 * stdev * stdev + 0.0705829 * stdev + 5.72816e-5);
      FDFilter(imgc3b, imgb, imgc3d, imgcl, filt * filt);
      cv::merge(std::vector{imgc1d, imgc2d, imgc3d}, imgcd);
      imgcd(cv::Rect(ksize, ksize, imgi.cols, imgi.rows)).copyTo(tmp1);
      tmp1.copyTo(img3d);
      img3d = img3d * 255;
      cv::imwrite("Deblurred_5.png", img3d);
    }
    // merge and show
    imgcd = cv::Mat(cv::Size(imgb.cols, imgb.rows), CV_32FC3);
    cv::merge(std::vector{imgc1d, imgc2d, imgc3d}, imgcd);
    // crop the borders
    imgcd(cv::Rect(ksize, ksize, imgi.cols, imgi.rows)).copyTo(tmp1);
    tmp1.copyTo(img3d);
    cv::imshow("DeblurredColor", img3d);
    // img3d = img3d/255;
    // cv::imwrite("Deblurred.png",img3d);
  } else {
    if (flag) {
      // blurring
      FCFilter(imgc1b, imgb, imgc1b);
    }

    // crop the blurred image
    // imgc1b(cv::Rect(kernel.cols, kernel.rows, imgi.cols,
    // imgi.rows)).copyTo(tmp);
    imgc1b(cv::Rect(ksize, ksize, imgi.cols, imgi.rows)).copyTo(tmp);
    tmp.copyTo(img1b);

    // deblurr
    if (flag) {
      FDFilter(imgc1b, imgb, imgc1d);
    } else {
      cv::Mat imgcl(cv::Size(imgb.cols, imgb.rows), CV_32FC1);

      // |x| norm, GCV
      stdev = NoiseDev(imgc1b);
      stdev *= regul / 3.12;
      filt = sqrt(14.709 * stdev * stdev + 0.241801 * stdev + 1e-6);
      FDFilter(imgc1b, imgb, imgc1d, imgc1b, filt * filt);
      imgc1d(cv::Rect(ksize, ksize, imgi.cols, imgi.rows)).copyTo(tmp1);
      tmp1.copyTo(img1d);
      img1d = img1d * 255;
      cv::imwrite("Deblurredbw_1.png", img1d);

      // |grad x| norm, GCV
      stdev = NoiseDev(imgc1b);
      stdev *= regul / 3.12;
      GradNorm(imgc1b, imgcl);
      filt = sqrt(14.8053 * stdev * stdev + 0.0383666 * stdev + 7.87368e-5);
      FDFilter(imgc1b, imgb, imgc1d, imgcl, filt * filt);
      imgc1d(cv::Rect(ksize, ksize, imgi.cols, imgi.rows)).copyTo(tmp1);
      tmp1.copyTo(img1d);
      img1d = img1d * 255;
      cv::imwrite("Deblurredbw_2.png", img1d);

      // Laplace norm, GCV
      stdev = NoiseDev(imgc1b);
      stdev *= regul / 3.12;
      cv::Laplacian(imgc1b, imgcl, -1);
      filt = sqrt(6.67533 * stdev * stdev + 0.00089085 * stdev + 2e-6);
      FDFilter(imgc1b, imgb, imgc1d, imgcl, filt * filt);
      imgc1d(cv::Rect(ksize, ksize, imgi.cols, imgi.rows)).copyTo(tmp1);
      tmp1.copyTo(img1d);
      img1d = img1d * 255;
      cv::imwrite("Deblurredbw_3.png", img1d);

      // HLapl norm, GCV
      stdev = NoiseDev(imgc1b);
      stdev *= regul / 3.12;
      // HLNorm(imgc1b, imgcl);
      filt = sqrt(3.80424 * stdev * stdev - 0.00449728 * stdev + 2.14891e-5);
      FDFilter(imgc1b, imgb, imgc1d, imgcl, filt * filt);
      imgc1d(cv::Rect(ksize, ksize, imgi.cols, imgi.rows)).copyTo(tmp1);
      tmp1.copyTo(img1d);
      img1d = img1d * 255;
      cv::imwrite("Deblurredbw_4.png", img1d);

      // TV norm, GCV
      stdev = NoiseDev(imgc1b);
      stdev *= regul / 3.12;
      // TotVar(imgc1b, imgcl);
      filt = sqrt(23.9335 * stdev * stdev + 0.0705829 * stdev + 5.72816e-5);
      FDFilter(imgc1b, imgb, imgc1d, imgcl, filt * filt);
      imgc1d(cv::Rect(ksize, ksize, imgi.cols, imgi.rows)).copyTo(tmp1);
      tmp1.copyTo(img1d);
      img1d = img1d * 255;
      cv::imwrite("Deblurredbw_5.png", img1d);
    }
    // crop the deblurred image
    imgc1d(cv::Rect(ksize, ksize, imgi.cols, imgi.rows)).copyTo(tmp1);
    tmp1.copyTo(img1d);

    cv::imshow("BlurredGray", img1b);
    cv::imshow("DeblurredGray", img1d);

    // img1d = img1d * 255;
    // cv::imwrite("Deblurredbw.png",img1d);
  }

  if (flag) {
    kernel = kernel * sqrt(xb1 * xb1 + yb1 * yb1);
  } else {
  }
  cv::imshow("Kernel", kernel);
}
