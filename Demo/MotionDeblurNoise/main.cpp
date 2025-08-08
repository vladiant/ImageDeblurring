/*
 * This is a demo program
 * used to demonstrate the
 * motion blurr for images
 * Noise added to images
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

cv::Mat kernel, imgb;  // kernel and image for blurring, one for all

int pos_Theta = 10, pos_r = 10;  // Angle and radius of motion blurr (initial)

// parameters of the noise
//  Gaussian: st. dev., mean; Uniform noise deviation; Salt'n'Pepper part;
//  Poisson noise coefficient; Specle noise coefficient
int pos_n1 = 0, pos_n2 = 0, pos_n3 = 0, pos_n4 = 0, pos_n5 = 0, pos_n6 = 0;
float xb1, yb1;    // Define vector of motion blurr
double sca = 1.0;  // scaling factor

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

// callback for mouse events
void my_mouse_callback(int event, int x, int y, int flags, void *param);

// initial declaration of blurr and deblurr function
void Process(int pos, void *);

// initial declaration of trackbar reading function
void GetNois(int pos, void *);

int main(int argc, char *argv[]) {
  int i;

  // image with action button to be pressed
  cv::Mat Action_button = cv::Mat(cv::Size(210, 90), CV_8UC1);
  cv::putText(Action_button, "Calculate!", cv::Point(20, 50),
              cv::FONT_HERSHEY_TRIPLEX, 1.0, cv::Scalar(255));

  // create the mouse action handler

  // createst simple kernel for motion
  int ksize = MAX_R;
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

  cv::namedWindow("Motion Blurr", cv::WINDOW_NORMAL);
  cv::namedWindow("Kernel", cv::WINDOW_NORMAL);
  // cv::resizeWindow("Motion Blurr", 600, 1);
  cv::imshow("Motion Blurr", Action_button);
  cv::setMouseCallback("Motion Blurr", my_mouse_callback);
  cv::createTrackbar("Radius, pix", "Motion Blurr", &pos_r, MAX_R, GetNois);
  cv::createTrackbar("Angle, deg ", "Motion Blurr", &pos_Theta, 360, GetNois);
  cv::createTrackbar("Gaussian noise dev., % ", "Motion Blurr", &pos_n1, 100,
                     GetNois);
  cv::createTrackbar("Gaussian noise mean, % ", "Motion Blurr", &pos_n2, 100,
                     GetNois);
  cv::createTrackbar("Uniform noise dev, % ", "Motion Blurr", &pos_n3, 100,
                     GetNois);
  cv::createTrackbar("Bad Pixels, o/oo ", "Motion Blurr", &pos_n4, 100,
                     GetNois);
  cv::createTrackbar("Poisson noise coeff.", "Motion Blurr", &pos_n5, 500,
                     GetNois);
  cv::createTrackbar("Speckle noise coeff.", "Motion Blurr", &pos_n6, 100,
                     GetNois);

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

  return (0);
}

void my_mouse_callback(int event, int x, int y, int flags, void *param) {
  if (event == cv::EVENT_LBUTTONDOWN) Process(0, nullptr);
}

void GetNois(int pos, void *) {
  pos_r = cv::getTrackbarPos("Radius, pix", "Motion Blurr");
  pos_Theta = cv::getTrackbarPos("Angle, deg ", "Motion Blurr");
  pos_n1 = cv::getTrackbarPos("Gaussian noise dev., % ", "Motion Blurr");
  pos_n2 = cv::getTrackbarPos("Gaussian noise mean, % ", "Motion Blurr");
  pos_n3 = cv::getTrackbarPos("Uniform noise dev, % ", "Motion Blurr");
  pos_n4 = cv::getTrackbarPos("Bad Pixels, o/oo ", "Motion Blurr");
  pos_n5 = cv::getTrackbarPos("Poisson noise coeff.", "Motion Blurr");
  pos_n6 = cv::getTrackbarPos("Speckle noise coeff.", "Motion Blurr");
}

void Process(int pos, void *) {
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

  // creates Fourier smearing image
  imgb = 0;
  Blurrset(imgb, kernel);

  // impose border conditions
  cv::copyMakeBorder(imgc1, imgc1b, kernel.rows, kernel.rows, kernel.cols,
                     kernel.cols, cv::BORDER_REPLICATE);
  EdgeTaper(imgc1b, kernel.rows, kernel.rows);

  if (imgi.channels() != 1) {
    cv::copyMakeBorder(imgc2, imgc2b, kernel.rows, kernel.rows, kernel.cols,
                       kernel.cols, cv::BORDER_REPLICATE);
    EdgeTaper(imgc2b, kernel.rows, kernel.rows);
    cv::copyMakeBorder(imgc3, imgc3b, kernel.rows, kernel.rows, kernel.cols,
                       kernel.cols, cv::BORDER_REPLICATE);
    EdgeTaper(imgc3b, kernel.rows, kernel.rows);
  }

  if (imgi.channels() != 1) {
    // blurring - each channel separately
    FCFilter(imgc1b, imgb, imgc1b);
    FCFilter(imgc2b, imgb, imgc2b);
    FCFilter(imgc3b, imgb, imgc3b);

    // add noise
    Gnoise(imgc1b, pos_n2 / 100.0, pos_n1 / 100.0);
    Gnoise(imgc2b, pos_n2 / 100.0, pos_n1 / 100.0);
    Gnoise(imgc3b, pos_n2 / 100.0, pos_n1 / 100.0);
    Unoise(imgc1b, pos_n3 / 100.0, 1.0);
    Unoise(imgc2b, pos_n3 / 100.0, 1.0);
    Unoise(imgc3b, pos_n3 / 100.0, 1.0);
    SPnoise(imgc1b, pos_n4 / 2000.0);
    SPnoise(imgc2b, pos_n4 / 2000.0);
    SPnoise(imgc2b, pos_n4 / 2000.0);
    if (pos_n5 != 0) Pnoise(imgc1b, pos_n5);
    if (pos_n5 != 0) Pnoise(imgc2b, pos_n5);
    if (pos_n5 != 0) Pnoise(imgc3b, pos_n5);
    SPEnoise(imgc1b, pos_n6 / 100.0);
    SPEnoise(imgc2b, pos_n6 / 100.0);
    SPEnoise(imgc3b, pos_n6 / 100.0);

    // merge and show
    imgcb = cv::Mat(cv::Size(imgb.cols, imgb.rows), CV_32FC3);
    cv::merge(std::vector{imgc1b, imgc2b, imgc3b}, imgcb);
    // crop the borders
    // imgcb(cv::Rect(kernel.cols, kernel.rows,
    // imgi.cols, imgi.rows)).copyTo(tmp);
    imgcb(cv::Rect(kernel.rows, kernel.rows, imgi.cols, imgi.rows)).copyTo(tmp);
    tmp.copyTo(img3b);
    cv::imshow("BlurredColor", img3b);

    // deblurring - each channel separately
    FDFilter(imgc1b, imgb, imgc1d);
    FDFilter(imgc2b, imgb, imgc2d);
    FDFilter(imgc3b, imgb, imgc3d);
    // merge and show
    imgcd = cv::Mat(cv::Size(imgb.cols, imgb.rows), CV_32FC3);
    cv::merge(std::vector{imgc1d, imgc2d, imgc3d}, imgcd);
    // crop the borders
    imgcd(cv::Rect(kernel.rows, kernel.rows, imgi.cols, imgi.rows))
        .copyTo(tmp1);
    tmp1.copyTo(img3d);
    cv::imshow("DeblurredColor", img3d);
    // img3d = img3d/255;
    // cv::imwrite("Deblurred.png",img3d);
  } else {
    // blurring
    FCFilter(imgc1b, imgb, imgc1b);

    // add noise
    Gnoise(imgc1b, pos_n2 / 100.0, pos_n1 / 100.0);
    Unoise(imgc1b, pos_n3 / 100.0, 1.0);
    SPnoise(imgc1b, pos_n4 / 2000.0);
    if (pos_n5 != 0) Pnoise(imgc1b, pos_n5);
    SPEnoise(imgc1b, pos_n6 / 100.0);

    // crop the blurred image
    imgc1b(cv::Rect(kernel.rows, kernel.rows, imgi.cols, imgi.rows))
        .copyTo(tmp);
    tmp.copyTo(img1b);

    // deblurr
    FDFilter(imgc1b, imgb, imgc1d);
    // crop the deblurred image
    imgc1d(cv::Rect(kernel.rows, kernel.rows, imgi.cols, imgi.rows))
        .copyTo(tmp1);
    tmp1.copyTo(img1d);

    cv::imshow("BlurredGray", img1b);
    cv::imshow("DeblurredGray", img1d);
  }

  kernel = kernel * sqrt(xb1 * xb1 + yb1 * yb1);
  cv::imshow("Kernel", kernel);
}
