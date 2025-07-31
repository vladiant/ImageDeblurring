/*
 * This is a demo program
 * used to demonstrate the
 * motion blurr for images
 * based on real movements
 * Calculations in YCrCb space
 * Y channel deblurred only
 *
 * Created by Vladislav Antonov
 * December 2011
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

const int MAX_R = 20;  // maximal length of motion blurr

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
    kernelbig;  // kernel and image for blurring, one for all; image to show
                // all the kernels in image

int pos_Theta1 = 0, pos_r1 = 10, pos_Theta2 = 3,
    pos_dir = 0;             // Angle and radius of motion blurr (initial)
int pos_x0 = 0, pos_y0 = 0;  // Initial position of the base point

// old positions - checked for new blurring
int pos0_Theta1, pos0_r1, pos0_Theta2, pos0_dir, pos0_x0, pos0_y0;

float xb1, yb1, rot;  // Define vector of motion blurr
double sca = 1.0;     // scaling factor

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

/* Function to calculate
 * the norm of the noise
 */

float NoiseDevN(const cv::Mat &imga) {
  cv::Mat imgd = cv::Mat(cv::Size(imga.cols, imga.rows), CV_32FC1);
  float re, im, shar, sum;

  shar = 0.8;
  sum = 0.0;
  re = 0;
  im = 0;
  imgd = imga / 255.0;
  cv::dft(imgd, imgd);
  for (int row = 0; row < int(shar * imgd.rows / 2); row++) {
    for (int col = 0; col < int(shar * imgd.cols / 2); col++) {
      FSet2D(imgd, col, row, &re, &im);
    }
  }
  cv::dft(imgd, imgd, cv::DFT_INVERSE | cv::DFT_SCALE);
  sum = cv::norm(imgd, cv::NORM_L2);
  return (sum);
}

/*
 * Function which introduces
 * blurring kernel on image
 * with periodic boundary conditions
 * kernel is spatially dependent
 * x0, y0 are the initial point coordinates
 * xt, yt - direction of translation
 */

void BlurrPBCsv(cv::Mat &imga, int x0, int y0, float xt, float yt, float rot,
                cv::Mat &imgd, int chan = 0, int tr = 0) {
  float s1, s2, s3;
  int i, j, ksize;
  float xb, yb;
  int iter = 0, iter1 = 0;
  float step = (imga.rows) * (imga.cols) / 150.0;

  // kernel
  cv::Mat imgb, imgs = cv::Mat(cv::Size(150, 10), CV_8UC3);
  cv::Mat imgc = cv::Mat(cv::Size(imga.cols, imga.rows), CV_32FC1);
  cv::namedWindow("Progress", cv::WINDOW_AUTOSIZE);

  imgs = 0;
  cv::imshow("Progress", imgs);

  for (int row = 0; row < imga.rows; row++) {
    // cout << row << '\n';
    for (int col = 0; col < imga.cols; col++) {
      s2 = 0;

      // here the kernel is changed
      xb = xt + (col - x0) * (1 - cos(rot)) + (row - y0) * sin(rot);
      yb = yt + (row - y0) * (1 - cos(rot)) - (col - x0) * sin(rot);
      ksize = max(abs(int(xb)) + 1, abs(int(yb)) + 1);
      imgb = cv::Mat(cv::Size(2 * ksize + 1, 2 * ksize + 1), CV_32FC1);
      iter++;

      if ((xb == 0) && (yb == 0)) {
        imgb = 0;
        ((float *)(imgb.data +
                   int(imgb.cols / 2) * (imgb.step)))[int(imgb.rows / 2)] = 1.0;
      } else {
        if (tr == 0)
          ksetMoveXY(imgb, xb, yb);
        else
          ksetMoveXY(imgb, -xb, -yb);
      }

      for (int row1 = 0; row1 < imgb.rows; row1++) {
        for (int col1 = 0; col1 < imgb.cols; col1++) {
          s1 = ((float *)(imgb.data + row1 * imgb.step))[col1];

          if ((row - row1 + imgb.rows / 2) >= 0) {
            if ((row - row1 + imgb.rows / 2) < (imga.rows)) {
              i = row - row1 + imgb.rows / 2;
            } else {
              i = row - row1 + imgb.rows / 2 - imga.rows + 1;
            }

          } else {
            i = (row - row1 + imgb.rows / 2) + imga.rows - 1;
          }

          if ((col - col1 + imgb.cols / 2) >= 0) {
            if ((col - col1 + imgb.cols / 2) < (imga.cols)) {
              j = col - col1 + imgb.cols / 2;
            } else {
              j = col - col1 + imgb.cols / 2 - imga.cols + 1;
            }

          } else {
            j = (col - col1 + imgb.cols / 2) + imga.cols - 1;
          }

          s3 = ((float *)(imga.data + i * imga.step))[j] * s1;
          s2 += s3;
        }
      }
      ((float *)(imgc.data + row * imgc.step))[col] = s2;

      // progress bar indicator
      imgb = imgb * sqrt((xb + 1) * (xb + 1) + (yb + 1) * (yb + 1));

      if (iter > step) {
        iter = 0;
        switch (chan) {
          case 0:
            cv::line(imgs, cv::Point(iter1, 0), cv::Point(iter1, 10),
                     cv::Scalar(255, 255, 255), 1);
            break;
          case 1:
            cv::line(imgs, cv::Point(iter1, 0), cv::Point(iter1, 10),
                     cv::Scalar(0, 0, 255), 1);
            break;
          case 2:
            cv::line(imgs, cv::Point(iter1, 0), cv::Point(iter1, 10),
                     cv::Scalar(0, 255, 0), 1);
            break;
          case 3:
            cv::line(imgs, cv::Point(iter1, 0), cv::Point(iter1, 10),
                     cv::Scalar(255, 0, 0), 1);
            break;
        }
        iter1++;
      }
      cv::imshow("Progress", imgs);
      cv::waitKey(1);
    }
  }
  imgc.copyTo(imgd);
  cv::destroyWindow("Progress");
}

/*
 * Total variation calculation
 */

float TotVar(cv::Mat &imga) {
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
      s3 += sqrt(s1 * s1 + s2 * s2);
    }
  }
  return (s3);
}

void Process(int pos, void *);  // initial declaration of blurr function

// deblurring function
void ProcDeblur(int pos);

// callback for mouse events
void my_mouse_callback(int event, int x, int y, int flags, void *param);

// initial declaration of trackbar reading function
void GetPos(int pos, void *);

int main(int argc, char *argv[]) {
  int i, ksize;

  // image with action button to be pressed
  cv::Mat Action_button = cv::Mat(cv::Size(290, 90), CV_8UC1);
  cv::putText(Action_button, "Blur!", cv::Point(20, 50),
              cv::FONT_HERSHEY_TRIPLEX, 1.0, cv::Scalar(255));
  cv::putText(Action_button, "Deblur!", cv::Point(145, 50),
              cv::FONT_HERSHEY_TRIPLEX, 1.0, cv::Scalar(255));
  cv::line(Action_button, cv::Point(126, 0), cv::Point(126, 90),
           cv::Scalar(255), 3);

  // createst simple kernel for motion
  ksize = MAX_R;
  kernel = cv::Mat(cv::Size(2 * ksize + 1, 2 * ksize + 1), CV_32FC1);

  // creates initial image
  if ((argc == 2) &&
      (!(img = cv::imread(argv[1], cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR))
            .empty())) {
    imgi = cv::Mat(cv::Size(img.cols, img.rows), img.depth(), img.channels());
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
    }
  } else {
    sca = 1.0;
    int m = 640, n = 480;  // VGA size
    imgi = cv::Mat(cv::Size(m, n), CV_32FC1);
    Ximgset(imgi);
  }

  // displays image
  cv::namedWindow("Initial", cv::WINDOW_AUTOSIZE);
  cv::imshow("Initial", imgi);

  // convert to CIE Luv
  if (imgi.channels() != 1) cv::cvtColor(imgi, imgi, cv::COLOR_BGR2YCrCb);

  // space variant kernel
  kernelbig = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_8UC1);

  if (imgi.channels() != 1) {
    cv::namedWindow("BlurredColor", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("DeblurredColor", cv::WINDOW_AUTOSIZE);
  } else {
    cv::namedWindow("BlurredGray", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("DeblurredGray", cv::WINDOW_AUTOSIZE);
  }

  // this is the kernel image for blurring
  imgb = cv::Mat(
      cv::Size(imgi.cols + 2 * (kernel.cols), imgi.rows + 2 * (kernel.rows)),
      CV_32FC1);

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

  cv::namedWindow("Motion Blurr", cv::WINDOW_NORMAL);
  // cv::resizeWindow("Motion Blurr", 600, 100);
  cv::imshow("Motion Blurr", Action_button);
  cv::setMouseCallback("Motion Blurr", my_mouse_callback);
  cv::createTrackbar("Radius, pix", "Motion Blurr", &pos_r1, MAX_R, GetPos);
  cv::createTrackbar("Angle, deg ", "Motion Blurr", &pos_Theta1, 360, GetPos);
  cv::createTrackbar("Base x, %", "Motion Blurr", &pos_x0, 100, GetPos);
  cv::createTrackbar("Base y, %", "Motion Blurr", &pos_y0, 100, GetPos);
  cv::createTrackbar("Rot. angle, deg. x 100", "Motion Blurr", &pos_Theta2, 300,
                     GetPos);
  cv::createTrackbar("Direction, +/- ", "Motion Blurr", &pos_dir, 1, GetPos);

  Process(0, nullptr);

  while (1)  // wait till ESC is pressed
  {
    char c = cv::waitKey(0);
    if (c == 27) break;
  }

  cv::destroyWindow("Initial");
  cv::destroyWindow("Motion Blurr");
  cv::destroyWindow("Kernel");

  if (imgi.channels() != 1) {
    cv::destroyWindow("BlurredColor");
    cv::destroyWindow("DeblurredColor");
  } else {
    cv::destroyWindow("BlurredGray");
    cv::destroyWindow("DeblurredGray");
  }

  return 0;
}

void my_mouse_callback(int event, int x, int y, int flags, void *param) {
  if (event == cv::EVENT_LBUTTONDOWN)
    if (x <= 126)
      Process(0, nullptr);
    else {
      ProcDeblur(0);
    }
}

void GetPos(int pos, void *) {
  pos_r1 = cv::getTrackbarPos("Radius, pix", "Motion Blurr");
  pos_Theta1 = cv::getTrackbarPos("Angle, deg ", "Motion Blurr");
  pos_x0 = cv::getTrackbarPos("Base x, %", "Motion Blurr");
  pos_y0 = cv::getTrackbarPos("Base y, %", "Motion Blurr");
  pos_Theta2 = cv::getTrackbarPos("Rot. angle, deg. x 100", "Motion Blurr");
  pos_dir = cv::getTrackbarPos("Direction, +/- ", "Motion Blurr");
}

void Process(int pos, void *) {
  // motion blurr kernel preparation
  xb1 = pos_r1 * cos(pos_Theta1 * CV_PI / 180);  // translation ange
  yb1 = pos_r1 * sin(pos_Theta1 * CV_PI / 180);
  rot = (pos_Theta2 * CV_PI / 180) * (1 - pos_dir * 2) / 100.0;
  int x0 = int(pos_x0 * (imgc1.cols) / 100.0);
  int y0 = int(pos_y0 * (imgc1.rows) / 100.0);

  // impose border conditions
  cv::copyMakeBorder(imgc1, imgc1b, kernel.rows, kernel.rows, kernel.cols,
                     kernel.cols, cv::BORDER_REPLICATE);
  EdgeTaper(imgc1b, kernel.cols, kernel.cols);

  if (imgi.channels() != 1) {
    cv::copyMakeBorder(imgc2, imgc2b, kernel.rows, kernel.rows, kernel.cols,
                       kernel.cols, cv::BORDER_REPLICATE);
    EdgeTaper(imgc2b, kernel.cols, kernel.cols);
    cv::copyMakeBorder(imgc3, imgc3b, kernel.rows, kernel.rows, kernel.cols,
                       kernel.cols, cv::BORDER_REPLICATE);
    EdgeTaper(imgc3b, kernel.cols, kernel.cols);
  }

  if (imgi.channels() != 1) {
    // blurring - each channel separately
    BlurrPBCsv(imgc3b, x0, y0, xb1, yb1, rot, imgc3b, 1);
    BlurrPBCsv(imgc2b, x0, y0, xb1, yb1, rot, imgc2b, 2);
    BlurrPBCsv(imgc1b, x0, y0, xb1, yb1, rot, imgc1b, 3);

    // merge and show
    imgcb = cv::Mat(cv::Size(imgb.cols, imgb.rows), CV_32FC3);
    cv::merge(std::vector{imgc1b, imgc2b, imgc3b}, imgcb);
    // crop the borders

    imgcb(cv::Rect(kernel.cols, kernel.rows, imgi.cols, imgi.rows))
        .copyTo(img3b);

    // convert to RGB
    cv::cvtColor(img3b, img3b, cv::COLOR_YCrCb2BGR);
    cv::imshow("BlurredColor", img3b);

    // convert to CIE HLS
    cv::cvtColor(imgcb, imgcb, cv::COLOR_BGR2HLS);
    cv::split(imgcb, std::vector{imgc1b, imgc2b, imgc3b});

  } else {
    // blurring
    BlurrPBCsv(imgc1b, x0, y0, xb1, yb1, rot, imgc1b);

    // crop the blurred image
    imgc1b(cv::Rect(kernel.cols, kernel.rows, imgi.cols, imgi.rows))
        .copyTo(img1b);

    cv::imshow("BlurredGray", img1b);
  }

  kernel = kernel * sqrt(xb1 * xb1 + yb1 * yb1);

  kernelbig = 0;
  // motion vectors
  for (int row = 0; row < kernelbig.rows; row += 2 * MAX_R + 1)
    for (int col = 0; col < kernelbig.cols; col += 2 * MAX_R + 1) {
      cv::circle(kernelbig, cv::Point(col + MAX_R, row + MAX_R), 3,
                 cv::Scalar(255), -1);
      cv::line(kernelbig, cv::Point(col + MAX_R, row + MAX_R),
               cv::Point(col + MAX_R +
                             int(xb1 + (col - x0) * (1 - cos(rot)) +
                                 (row - y0) * sin(rot)),
                         row + MAX_R +
                             int(yb1 + (row - y0) * (1 - cos(rot)) -
                                 (col - x0) * sin(rot))),
               cv::Scalar(255), 1);
    }
  cv::imshow("Kernel", kernelbig);

  pos0_r1 = pos_r1;
  pos0_Theta1 = pos_Theta1;
  pos0_x0 = pos_x0;
  pos0_y0 = pos_y0;
  pos0_Theta2 = pos_Theta2;
  pos0_dir = pos_dir;

  // just copy the previously blurred image

  imgc1b.copyTo(imgc1d);
  if (imgi.channels() != 1) {
    imgc2b.copyTo(imgc2d);
    imgc3b.copyTo(imgc3d);
  }
}

void ProcDeblur(int pos) {
  float norm, oldnorm, norm1, norm2,
      stnorm = 1;  // norms for convergence of the Lucy-Richardson iterations
  float norm3, norm4, norm5, norm6;
  float lambda;      // adjustable parameter for control
  bool flag = true;  // flag for iteration control
  int itrs = 0;      // iterations
  // temporal images
  cv::Mat imgc1t = cv::Mat(cv::Size(imgc1d.cols, imgc1d.rows), CV_32FC1),
          imgc1tt = cv::Mat(cv::Size(imgc1d.cols, imgc1d.rows), CV_32FC1);

  // check for change - if so, new blurring
  if ((pos_r1 != pos0_r1) || (pos_Theta1 != pos0_Theta1) ||
      (pos_x0 != pos0_x0) || (pos_y0 != pos0_y0) ||
      (pos_Theta2 = pos0_Theta2) || (pos_dir = pos0_dir)) {
    Process(0, nullptr);
  }

  // motion blurr kernel preparation
  xb1 = pos_r1 * cos(pos_Theta1 * CV_PI / 180);  // translation ange
  yb1 = pos_r1 * sin(pos_Theta1 * CV_PI / 180);
  rot = (pos_Theta2 * CV_PI / 180) * (1 - pos_dir * 2) / 100.0;
  int x0 = int(pos_x0 * (imgc1.cols) / 100.0);
  int y0 = int(pos_y0 * (imgc1.rows) / 100.0);

  // impose border conditions - required for new images
  /*
  cv::copyMakeBorder(imgc1,imgc1b,ckernel.rows, kernel.rows, kernel.cols,
  kernel.cols, cv::BORDER_REPLICATE); EdgeTaper(imgc1b, kernel.cols,
  kernel.cols);

  if(imgi.channels()!=1)
          {
                  cv::copyMakeBorder(imgc2,imgc2b,kernel.rows, kernel.rows,
  kernel.cols, kernel.cols, cv::BORDER_REPLICATE); EdgeTaper(imgc2b,
  kernel.cols, kernel.cols); cv::copyMakeBorder(imgc3,imgc3b,kernel.rows,
  kernel.rows, kernel.cols, kernel.cols, cv::BORDER_REPLICATE);
  EdgeTaper(imgc3b, kernel.cols, kernel.cols);
          }
  */

  if (imgi.channels() != 1) {
    // deblurring - each channel separately
    // Lucy-Richardson
    do {
      itrs++;

      /*
      // channel 1
      cvSetZero(imgc1t);
      BlurrPBCsv(imgc3d, x0, y0, xb1, yb1, rot, imgc1t,1);
      //norm |Ax-b| calculation
      cv::subtract(imgc1t,imgc3b,imgc1tt);
      norm1=cv::norm( imgc1tt, cv::NORM_L2 );
      cv::divide(imgc3b,imgc1t,imgc1t);
      BlurrPBCsv(imgc1t, x0, y0, xb1, yb1, rot,imgc1t,1,1);
      cv::multiply(imgc1t,imgc3d,imgc3d);

      // channel 2

      imgc1t = 0;
      BlurrPBCsv(imgc2d, x0, y0, xb1, yb1, rot, imgc1t, 2);
      // norm |Ax-b| calculation
      cv::subtract(imgc1t, imgc2b, imgc1tt);
      norm1 += cv::norm(imgc1tt, cv::NORM_L2);
      cv::divide(imgc2b, imgc1t, imgc1t);
      BlurrPBCsv(imgc1t, x0, y0, xb1, yb1, rot, imgc1t, 2, 1);
      cv::multiply(imgc1t, imgc2d, imgc2d);
      */
      // channel 3

      imgc1t = 0;
      BlurrPBCsv(imgc1d, x0, y0, xb1, yb1, rot, imgc1t, 3);
      // norm |Ax-b| calculation
      cv::subtract(imgc1t, imgc1b, imgc1tt);
      norm1 += cv::norm(imgc1tt, cv::NORM_L2);
      cv::divide(imgc1b, imgc1t, imgc1t);
      BlurrPBCsv(imgc1t, x0, y0, xb1, yb1, rot, imgc1t, 3, 1);
      cv::multiply(imgc1t, imgc1d, imgc1d);

      if (flag) {
        flag = false;
        oldnorm = norm1;
      } else {
        stnorm = oldnorm - norm1;
        oldnorm = norm1;
      }

      // merge and show
      imgcd = cv::Mat(cv::Size(imgb.cols, imgb.rows), CV_32FC3);
      cv::merge(std::vector{imgc1d, imgc2d, imgc3d}, imgcd);
      // crop the borders

      imgcd(cv::Rect(kernel.cols, kernel.rows, imgi.cols, imgi.rows))
          .copyTo(img3d);
      // convert to RGB
      cv::cvtColor(img3d, img3d, cv::COLOR_YCrCb2BGR);
      cv::imshow("DeblurredColor", img3d);
      cout << itrs << " |Ax-b|=  " << norm1 << '\n';
    } while ((itrs < 100));

  } else {
    /*
    lambda=NoiseDev(imgc1b)*2.5*((imgc1b.cols)*(imgc1b.rows));
    cout << lambda << '\n';
     */

    // deblur Lucy-Richardson
    do {
      itrs++;
      imgc1t = 0;
      BlurrPBCsv(imgc1d, x0, y0, xb1, yb1, rot, imgc1t);

      // norm calculation
      //  |x|
      norm2 = cv::norm(imgc1d, cv::NORM_L2);
      // |Ax-b|
      cv::subtract(imgc1t, imgc1b, imgc1tt);
      norm1 = cv::norm(imgc1tt, cv::NORM_L2);
      // Total Variation
      norm3 = TotVar(imgc1d);
      // Laplacian norm
      cv::Laplacian(imgc1d, imgc1tt, -1);
      norm4 = cv::norm(imgc1tt, cv::NORM_L2);
      // hyper Laplacian norm
      cv::Sobel(imgc1d, imgc1tt, -1, 1, 0);
      norm5 = cv::norm(imgc1tt, cv::NORM_L2);
      cv::Sobel(imgc1d, imgc1tt, -1, 0, 1);
      norm5 += cv::norm(imgc1tt, cv::NORM_L2);
      norm5 = pow(norm5, 0.8);
      // norm=norm1+lambda*norm2;

      if (flag) {
        flag = false;
        oldnorm = norm1;
      } else {
        stnorm = oldnorm - norm1;
        oldnorm = norm1;
      }

      cv::divide(imgc1b, imgc1t, imgc1t);
      BlurrPBCsv(imgc1t, x0, y0, xb1, yb1, rot, imgc1t, 0, 1);
      cv::multiply(imgc1t, imgc1d, imgc1d);

      // crop the deblurred image
      imgc1d(cv::Rect(kernel.cols, kernel.rows, imgi.cols, imgi.rows))
          .copyTo(img1d);
      cv::imshow("DeblurredGray", img1d);
      cout << itrs << " |Ax-b|=  " << norm1 << " |x|= " << norm2
           << " TV= " << norm3 << "  Lapl= " << norm4 << " HLapl= " << norm5
           << '\n';

    } while ((stnorm > 0) && (itrs < 100));
  }
}
