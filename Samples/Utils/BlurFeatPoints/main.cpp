/*
 * A program to calculate
 * the feature points in
 * blurred images based on:
 *
 * Hiroshi Kano, Haruo Hatanaka, Shimpei Fukumoto and Haruhiko Murata
 * "Motion blur estimation of handheld camera using regular- and short-exposure
 * image pair" 2009 16th IEEE International Conference on Image Processing
 * (ICIP), 7-10 Nov. 2009, Cairo pp. 1317 - 1320
 *
 * Created by Vladislav Antonov
 * 19 March 2012
 */

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

int main(int argc, char *argv[]) {
  // initial two images
  cv::Mat img, imgi;
  // temporal images
  cv::Mat img1, img2, img3, img4;
  double mn, mx;  // minimal and maximal values

  // loads initial image
  if (argc == 2) {
    if (!(imgi = cv::imread(argv[1], cv::IMREAD_GRAYSCALE)).empty()) {
      img = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
      float sca;  // value to rescale image to float
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
      imgi.convertTo(img, CV_32F, 1.0 / sca);
    } else {
      cout << '\n' << argv[1] << " couldn't be opened.\n";
      return 0;
    }

  } else {
    cout << "\nA program to calculate the feature points in blurred images "
            "based on:\n";
    cout << " Hiroshi Kano, Haruo Hatanaka, Shimpei Fukumoto and Haruhiko "
            "Murata\n";
    cout << "\"Motion blur estimation of handheld camera using regular- and "
            "short-exposure image pair\"\n";
    cout << "2009 16th IEEE International Conference on Image Processing "
            "(ICIP), 7-10 Nov. 2009, Cairo \n";
    cout << "pp. 1317 - 1320 \n\n";
    cout << "Usage: " << argv[0] << "  <initial_image>\n";
    return 0;
  }

  img1 = img.clone();
  img2 = img.clone();
  img3 = img.clone();
  img4 = img.clone();

  // Horizontal edges
  cv::Sobel(img, img1, -1, 1, 0, 5);

  // Vertical edges
  cv::Sobel(img, img2, -1, 0, 1, 5);

  // their differences
  cv::subtract(img1, img2, img3);
  cv::subtract(img2, img1, img4);

  // smooth them for noise reduction
  cv::blur(img3, img1, cv::Size(5, 5));
  cv::blur(img4, img2, cv::Size(5, 5));

  // multiply
  cv::multiply(img1, img2, img4);

  // caclulate min and max
  float kmin = 0, kmax = 0;
  for (int row = 0; row < img4.rows; row++) {
    for (int col = 0; col < img4.cols; col++) {
      float val1 = ((float *)(img4.data + row * img4.step))[col];
      if (val1 > kmax) kmax = val1;
      if (val1 < kmin) kmin = val1;
    }
  }

  // map values in [0:1]
  for (int row = 0; row < img4.rows; row++) {
    for (int col = 0; col < img4.cols; col++) {
      float val1 = ((float *)(img4.data + row * img4.step))[col];
      ((float *)(img4.data + row * img4.step))[col] =
          1 - (val1 - kmin) / (kmax - kmin);
    }
  }

  // image window control
  int IMG_WIN;

  if ((img1.cols > 1080) || (img1.rows > 720))
    IMG_WIN = 0;
  else
    IMG_WIN = cv::WINDOW_AUTOSIZE;

  // displays image
  cv::namedWindow("Initial", IMG_WIN);
  cv::imshow("Initial", img);
  cv::namedWindow("Features", IMG_WIN);

  if (IMG_WIN == cv::WINDOW_NORMAL) {
    cv::resizeWindow("Initial", 1080, 720);
    cv::resizeWindow("Features", 1080, 720);
  }

  // show them
  cv::imshow("Features", img4);

  cout << "\nPress \"S\" to save the feature image.\n";

  while (1)  // wait till ESC is pressed
  {
    char c = cv::waitKey(0);
    if (c == 27) break;
    if ((c == 's') || (c == 'S')) {
      img3 = img4;
      cv::imwrite("Features.tiff", img3);
    }
  }

  cv::destroyWindow("Initial");
  cv::destroyWindow("Features");

  return 0;
}