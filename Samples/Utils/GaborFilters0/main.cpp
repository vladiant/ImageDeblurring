#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat src;
cv::Mat src_f;
cv::Mat image;
cv::Mat dest;
cv::Mat dest_mag;
cv::Mat kernelimg;
cv::Mat big_kernelimg;
cv::Mat kernel;

int kernel_size = 21;
int pos_var = 50;
int pos_w = 5;
int pos_phase = 0;
int pos_psi = 90;

void Process(int pos, void*) {
  // CvFileStorage *fs = cvOpenFileStorage("kernel.xml",NULL,CV_STORAGE_WRITE);

  int x, y;
  float kernel_val;
  float var = (float)pos_var / 10;
  float w = (float)pos_w / 10;
  float phase = (float)pos_phase * CV_PI / 180;
  float psi = CV_PI * pos_psi / 180;

  kernel = 0;
  for (x = -kernel_size / 2; x <= kernel_size / 2; x++) {
    for (y = -kernel_size / 2; y <= kernel_size / 2; y++) {
      kernel_val = exp(-((x * x) + (y * y)) / (2 * var)) *
                   cos(w * x * cos(phase) + w * y * sin(phase) + psi);
      kernel.at<float>(y + kernel_size / 2, x + kernel_size / 2) = kernel_val;
      kernelimg.at<float>(y + kernel_size / 2, x + kernel_size / 2) =
          kernel_val / 2 + 0.5;
      //      printf("%f\n",kernel_val);
    }
  }
  // cvWrite( fs, "kernel", kernel, cvAttrList(0,0) );
  // cvReleaseFileStorage(&fs);

  cv::filter2D(src_f, dest, -1, kernel);
  cv::imshow("Process window", dest);
  cv::resize(kernelimg, big_kernelimg, big_kernelimg.size());
  cv::imshow("Kernel", big_kernelimg);
  cv::pow(dest, 2, dest_mag);
  //  cvPow(dest_mag,dest_mag,0.5);
  cv::imshow("Mag", dest_mag);
}

int main(int argc, char** argv) {
  char* filename = argv[1];
  if (argc == 2)
    image = cv::imread(filename, 1);
  else
    return 1;

  if (kernel_size % 2 == 0) kernel_size++;
  kernel = cv::Mat(kernel_size, kernel_size, CV_32FC1);
  kernelimg = cv::Mat(cv::Size(kernel_size, kernel_size), CV_32FC1);
  big_kernelimg =
      cv::Mat(cv::Size(kernel_size * 20, kernel_size * 20), CV_32FC1);

  src = cv::Mat(cv::Size(image.cols, image.rows), CV_8UC1);
  src_f = cv::Mat(cv::Size(image.cols, image.rows), CV_32FC1);

  cv::cvtColor(image, src, cv::COLOR_BGR2GRAY);
  src.convertTo(src_f, CV_32F, 1.0 / 255);
  dest = src_f.clone();
  dest_mag = src_f.clone();

  cv::namedWindow("Process window", 1);
  cv::namedWindow("Kernel", 1);
  cv::namedWindow("Mag", 1);
  cv::createTrackbar("Variance", "Process window", &pos_var, 100, Process);
  cv::createTrackbar("Pulsation", "Process window", &pos_w, 30, Process);
  cv::createTrackbar("Phase", "Process window", &pos_phase, 180, Process);
  cv::createTrackbar("Psi", "Process window", &pos_psi, 360, Process);

  Process(0, nullptr);

  cv::waitKey(0);

  cv::destroyAllWindows();
  return 0;
}
