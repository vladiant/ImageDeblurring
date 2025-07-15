/*
 * Simple program to test
 * Gabor filtering functions
 */

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

void Ximgset(cv::Mat& imga)  // generates the image
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

// Real Gabor Function
float psfRGabor(int x, int y, float sigma, float theta, float lambda,
                float psi = 0, float gamma = 1) {
  const double pi = 3.14159265358979;
  float x1 = x * cos(theta) + y * sin(theta),
        y1 = -x * sin(theta) + y * cos(theta), result;

  result = exp(-0.5 * (x1 * x1 + gamma * gamma * y1 * y1) / (sigma * sigma)) *
           cos(psi + 2 * pi * x1 / lambda);
  return (result);
}

// Imaginary Gabor Function
float psfIGabor(int x, int y, float sigma, float theta, float lambda,
                float psi = 0, float gamma = 1) {
  const double pi = 3.14159265358979;
  float x1 = x * cos(theta) + y * sin(theta),
        y1 = -x * sin(theta) + y * cos(theta), result;

  result = exp(-0.5 * (x1 * x1 + gamma * gamma * y1 * y1) / (sigma * sigma)) *
           sin(psi + 2 * pi * x1 / lambda);
  return (result);
}

// kernel span for Gabor functions
void kspanGabor(int* x, int* y, float sigma, float theta, float lambda,
                float psi = 0, float gamma = 1) {
  int nstds = 3;

  float sigma_x = sigma;
  float sigma_y = sigma / gamma;

  int xmax =
      max(abs(nstds * sigma_x * cos(theta)), abs(nstds * sigma_y * sin(theta)));
  xmax = ceil(max(1, xmax));
  int ymax =
      max(abs(nstds * sigma_x * sin(theta)), abs(nstds * sigma_y * cos(theta)));
  ymax = ceil(max(1, ymax));

  *x = xmax;
  *y = ymax;
}

// Sets kernel for a function with five variables for Gabor filter

void kset(cv::Mat& kernel,
          float (*ptLV1f)(int, int, float, float, float, float, float),
          float a1, float a2, float a3, float a4, float a5) {
  cv::Scalar ksum;

  kernel = 0;

  for (int row = 0; row < kernel.rows; row++) {
    for (int col = 1; col < kernel.cols; col++) {
      ((float*)(kernel.data + row * kernel.step))[col] += (*ptLV1f)(
          row - (kernel.cols / 2), col - (kernel.rows / 2), a1, a2, a3, a4, a5);
    }
  }

  ksum = cv::sum(kernel);
  kernel = kernel / ksum.val[0];
}

int main(int argc, char* argv[]) {
  cv::Mat img, imgi, imgii;  // initial
  cv::Mat img1, img2, img3, img4, img5, img6, img7, img8, imgb;
  cv::Mat img9, img10, img11, img12, img13, img14, img15, img16;
  int m = 320, n = 240,
      ksize;       // image dimensions and radius of blurring, kernel size
  cv::Mat kernel;  // kernel for blurring
  int xk, yk;      // kernel size
  const int x1 = 10, y1 = 10;  // Define vector of motion blurr

  float sigma, theta, lambda;  // parameters for Gobor functions

  // creates initial image
  if ((argc == 2) && (!(imgi = cv::imread(argv[1], 1)).empty())) {
    img = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
    if (imgi.channels() != 1) {
      imgii = cv::Mat(cv::Size(imgi.cols, imgi.rows), imgi.depth(), 1);
      cv::cvtColor(imgi, imgii, cv::COLOR_RGBA2GRAY);
      imgii.convertTo(img, CV_32F, 1.0 / 255.0);
    } else {
      imgi.convertTo(img, CV_32F, 1.0 / 255.0);
    }
  } else {
    img = cv::Mat(cv::Size(m, n), CV_32FC1);
    Ximgset(img);
  }

  // parameters for Gabor function
  sigma = pow(sqrt(2.0), 1 + 0);
  theta = CV_PI * 0.25 * 0.0;
  lambda = 2 * sigma;

  // creates filter kernel
  kspanGabor(&xk, &yk, sigma, theta, lambda);
  kernel = cv::Mat(cv::Size(2 * xk + 1, 2 * yk + 1), CV_32FC1);
  kset(kernel, psfRGabor, sigma, theta, lambda, 0, 1);

  // direct blurring via kernel
  imgb = img.clone();
  Blurrset(imgb, kernel);
  img1 = img.clone();
  FCFilter(img, imgb, img1);

  kset(kernel, psfIGabor, sigma, theta, lambda, 0, 1);
  Blurrset(imgb, kernel);
  img2 = img.clone();
  FCFilter(img, imgb, img2);

  // next turn
  theta = CV_PI * 0.25 * 1.0;

  // creates filter kernel
  kspanGabor(&xk, &yk, sigma, theta, lambda);
  kernel = cv::Mat(cv::Size(2 * xk + 1, 2 * yk + 1), CV_32FC1);
  kset(kernel, psfRGabor, sigma, theta, lambda, 0, 1);

  // direct blurring via kernel
  imgb = img.clone();
  Blurrset(imgb, kernel);
  img3 = img.clone();
  FCFilter(img, imgb, img3);

  kset(kernel, psfIGabor, sigma, theta, lambda, 0, 1);
  Blurrset(imgb, kernel);
  img4 = img.clone();
  FCFilter(img, imgb, img4);

  // next turn
  theta = CV_PI * 0.25 * 2.0;

  // creates filter kernel
  kspanGabor(&xk, &yk, sigma, theta, lambda);
  kernel = cv::Mat(cv::Size(2 * xk + 1, 2 * yk + 1), CV_32FC1);
  kset(kernel, psfRGabor, sigma, theta, lambda, 0, 1);

  // direct blurring via kernel
  imgb = img.clone();
  Blurrset(imgb, kernel);
  img5 = img.clone();
  FCFilter(img, imgb, img5);

  kset(kernel, psfIGabor, sigma, theta, lambda, 0, 1);
  Blurrset(imgb, kernel);
  img6 = img.clone();
  FCFilter(img, imgb, img6);

  // next turn
  theta = CV_PI * 0.25 * 3.0;

  // creates filter kernel
  kspanGabor(&xk, &yk, sigma, theta, lambda);
  kernel = cv::Mat(cv::Size(2 * xk + 1, 2 * yk + 1), CV_32FC1);
  kset(kernel, psfRGabor, sigma, theta, lambda, 0, 1);

  // direct blurring via kernel
  imgb = img.clone();
  Blurrset(imgb, kernel);
  img7 = img.clone();
  FCFilter(img, imgb, img7);

  kset(kernel, psfIGabor, sigma, theta, lambda, 0, 1);
  Blurrset(imgb, kernel);
  img8 = img.clone();
  FCFilter(img, imgb, img8);

  sigma = pow(sqrt(2.0), 1 + 1);
  theta = CV_PI * 0.25 * 0.0;
  lambda = 2 * sigma;

  // creates filter kernel
  kspanGabor(&xk, &yk, sigma, theta, lambda);
  kernel = cv::Mat(cv::Size(2 * xk + 1, 2 * yk + 1), CV_32FC1);
  kset(kernel, psfRGabor, sigma, theta, lambda, 0, 1);

  // direct blurring via kernel
  imgb = img.clone();
  Blurrset(imgb, kernel);
  img9 = img.clone();
  FCFilter(img, imgb, img9);

  kset(kernel, psfIGabor, sigma, theta, lambda, 0, 1);
  Blurrset(imgb, kernel);
  img10 = img.clone();
  FCFilter(img, imgb, img10);

  // next turn
  theta = CV_PI * 0.25 * 1.0;

  // creates filter kernel
  kspanGabor(&xk, &yk, sigma, theta, lambda);
  kernel = cv::Mat(cv::Size(2 * xk + 1, 2 * yk + 1), CV_32FC1);
  kset(kernel, psfRGabor, sigma, theta, lambda, 0, 1);

  // direct blurring via kernel
  imgb = img.clone();
  Blurrset(imgb, kernel);
  img11 = img.clone();
  FCFilter(img, imgb, img11);

  kset(kernel, psfIGabor, sigma, theta, lambda, 0, 1);
  Blurrset(imgb, kernel);
  img12 = img.clone();
  FCFilter(img, imgb, img12);

  // next turn
  theta = CV_PI * 0.25 * 2.0;

  // creates filter kernel
  kspanGabor(&xk, &yk, sigma, theta, lambda);
  kernel = cv::Mat(cv::Size(2 * xk + 1, 2 * yk + 1), CV_32FC1);
  kset(kernel, psfRGabor, sigma, theta, lambda, 0, 1);

  // direct blurring via kernel
  imgb = img.clone();
  Blurrset(imgb, kernel);
  img13 = img.clone();
  FCFilter(img, imgb, img13);

  kset(kernel, psfIGabor, sigma, theta, lambda, 0, 1);
  Blurrset(imgb, kernel);
  img14 = img.clone();
  FCFilter(img, imgb, img14);

  // next turn
  theta = CV_PI * 0.25 * 3.0;

  // creates filter kernel
  kspanGabor(&xk, &yk, sigma, theta, lambda);
  kernel = cv::Mat(cv::Size(2 * xk + 1, 2 * yk + 1), CV_32FC1);
  kset(kernel, psfRGabor, sigma, theta, lambda, 0, 1);

  // direct blurring via kernel
  imgb = img.clone();
  Blurrset(imgb, kernel);
  img15 = img.clone();
  FCFilter(img, imgb, img15);

  kset(kernel, psfIGabor, sigma, theta, lambda, 0, 1);
  Blurrset(imgb, kernel);
  img16 = img.clone();
  FCFilter(img, imgb, img16);

  // displays images
  cv::namedWindow("Initial", cv::WINDOW_AUTOSIZE);
  cv::imshow("Initial", img);
  cv::namedWindow("RG00", cv::WINDOW_AUTOSIZE);
  cv::imshow("RG00", img1);
  cv::namedWindow("IG00", cv::WINDOW_AUTOSIZE);
  cv::imshow("IG00", img2);
  cv::namedWindow("RG01", cv::WINDOW_AUTOSIZE);
  cv::imshow("RG01", img3);
  cv::namedWindow("IG01", cv::WINDOW_AUTOSIZE);
  cv::imshow("IG01", img4);
  cv::namedWindow("RG02", cv::WINDOW_AUTOSIZE);
  cv::imshow("RG02", img5);
  cv::namedWindow("IG02", cv::WINDOW_AUTOSIZE);
  cv::imshow("IG02", img6);
  cv::namedWindow("RG03", cv::WINDOW_AUTOSIZE);
  cv::imshow("RG03", img7);
  cv::namedWindow("IG03", cv::WINDOW_AUTOSIZE);
  cv::imshow("IG03", img8);
  cv::namedWindow("RG10", cv::WINDOW_AUTOSIZE);
  cv::imshow("RG10", img9);
  cv::namedWindow("IG10", cv::WINDOW_AUTOSIZE);
  cv::imshow("IG10", img10);
  cv::namedWindow("RG11", cv::WINDOW_AUTOSIZE);
  cv::imshow("RG11", img11);
  cv::namedWindow("IG11", cv::WINDOW_AUTOSIZE);
  cv::imshow("IG11", img12);
  cv::namedWindow("RG12", cv::WINDOW_AUTOSIZE);
  cv::imshow("RG12", img13);
  cv::namedWindow("IG12", cv::WINDOW_AUTOSIZE);
  cv::imshow("IG12", img14);
  cv::namedWindow("RG13", cv::WINDOW_AUTOSIZE);
  cv::imshow("RG13", img15);
  cv::namedWindow("IG13", cv::WINDOW_AUTOSIZE);
  cv::imshow("IG13", img16);

  cv::waitKey(0);

  cv::destroyWindow("Initial");
  cv::destroyWindow("RG00");
  cv::destroyWindow("IG00");
  cv::destroyWindow("RG01");
  cv::destroyWindow("IG01");
  cv::destroyWindow("RG02");
  cv::destroyWindow("IG02");
  cv::destroyWindow("RG03");
  cv::destroyWindow("IG03");
  cv::destroyWindow("RG00");
  cv::destroyWindow("IG10");
  cv::destroyWindow("RG11");
  cv::destroyWindow("IG11");
  cv::destroyWindow("RG12");
  cv::destroyWindow("IG12");
  cv::destroyWindow("RG13");
  cv::destroyWindow("IG13");

  return (0);
}
