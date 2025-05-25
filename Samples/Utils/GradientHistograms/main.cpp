/*
 * Simple program to create
 * to split image color channels
 * and to analyze them
 */

#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

using namespace std;

void FGet2D(const cv::Mat &Y, int k, int i, float *re, float *im) {
  int x, y;                       // pixel coordinates of Re Y(i,k).
  float *Yptr = (float *)Y.data;  // pointer to Re Y(i,k)
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
 * Function to calculate the standard deviation
 * of Fourier transform at the edges
 * which is supposed to be the error
 * of the noise estimation
 */

float NoiseDevErr(const cv::Mat &imga) {
  cv::Mat imgd(cv::Size(imga.cols, imga.rows), CV_32FC1);
  float re, im, shar, sum, sigm;

  shar = 0.8;
  sum = 0.0;
  sigm = NoiseDev(imga);
  imga.convertTo(imgd, CV_32F);
  imgd = imgd / 255.0;
  sigm *= sqrt((imgd.rows) * (imgd.cols));
  cv::dft(imgd, imgd);
  for (int row = int(shar * imgd.rows / 2); row < (imgd.rows / 2) - 1; row++) {
    for (int col = int(shar * imgd.cols / 2); col < (imgd.cols / 2) - 1;
         col++) {
      FGet2D(imgd, col, row, &re, &im);
      sum +=
          (sqrt(re * re + im * im) - sigm) * (sqrt(re * re + im * im) - sigm);
    }
  }
  sum = sqrt(sum);
  sum /= ((imgd.rows / 2) - int(shar * imgd.rows / 2)) *
         ((imgd.cols / 2) - int(shar * imgd.cols / 2)) *
         sqrt((imgd.rows) * (imgd.cols));
  return (sum);
}

/* Function to create
 * the histogram of
 * the gradients in the image
 */

cv::Mat GradHist(cv::Mat &imgw) {
  double min, max, min1, max1, min2, max2;

  cv::Mat imgdt1 = imgw.clone();
  cv::Mat imgdt2 = imgw.clone();

  cv::Sobel(imgw, imgdt1, -1, 1, 0);
  cv::Sobel(imgw, imgdt2, -1, 0, 1);
  cv::minMaxLoc(imgdt1, &min1, &max1, nullptr, nullptr);
  cv::minMaxLoc(imgdt1, &min2, &max2, nullptr, nullptr);

  min = std::min(min1, min2);
  max = std::min(max1, max2);

  cv::Mat planes[] = {imgdt1, imgdt2};

  int bins = 256;  // bins of the histogram

  cv::Mat hist;

  {
    const int hist_size[] = {bins};
    float range[] = {static_cast<float>(abs(min) < abs(max) ? -max : min),
                     static_cast<float>(abs(max) < abs(min) ? -min : max)};
    const float *ranges[] = {range};

    int channels[] = {0};
    cv::calcHist(planes, 2, channels, cv::Mat{}, hist, 1, hist_size, ranges);
  }

  // cv::equalizeHist(hist, hist);

  return hist;
}

/*
 * Double Lorentian used
 * used to fit the gradient histograms
 */

double DL(double x, double a0, double a1, double a2, double a3, double a4,
          double a5) {
  double result;
  result = a0 + (a1 / ((x - a2) * (x - a2) + a3 * a3)) +
           (a4 / ((x - a2) * (x - a2) + a5 * a5));
  return (result);
}

int main(int argc, char *argv[]) {
  cv::Mat img, imgi, imgc1, imgc2, imgc3, imgc1i, imgc2i, imgc3i;
  double sca = 1.0;
  char *nmR, *nmG, *nmB, *dtR, *dtG, *dtB, *dt;
  int i;
  float re, im;
  cv::Mat imgdt1, imgdt2;  // test image for the derivatives
  cv::Mat hist_img, hist_imgR, hist_imgG, hist_imgB;  // histogram images
  cv::Mat hist, histR, histG, histB;
  // results from a test fit
  cv::Mat a0(6, 1, CV_32FC1);
  cv::Mat a1(6, 1, CV_32FC1);

  // creates initial image
  if ((argc == 2) &&
      !((imgi = cv::imread(argv[1], cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR))
            .empty())) {
    img = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);

    imgdt1 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
    imgdt2 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);

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

    // displays image
    cv::namedWindow("Initial", cv::WINDOW_AUTOSIZE);

    // bilateral filtering with OpenCV
    // cv::Mat imgbl=imgi.clone();
    // cv::bilateralFilter(imgi,imgbl,2,0.5,5,5);
    // imgi=imgbl.clone();

    cv::imshow("Initial", imgi);

    imgc1 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);

    if (imgi.channels() != 1) {
      imgc1i = cv::Mat(cv::Size(imgi.cols, imgi.rows), imgi.depth(), 1);
      imgc2i = cv::Mat(cv::Size(imgi.cols, imgi.rows), imgi.depth(), 1);
      imgc3i = cv::Mat(cv::Size(imgi.cols, imgi.rows), imgi.depth(), 1);
      imgc2 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
      imgc3 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
      cv::split(imgi, std::vector{imgc1i, imgc2i, imgc3i});
      imgc1i.convertTo(imgc1, CV_32F);
      imgc2i.convertTo(imgc2, CV_32F);
      imgc3i.convertTo(imgc3, CV_32F);
      imgc1 = imgc1 / sca;
      imgc2 = imgc2 / sca;
      imgc3 = imgc3 / sca;
    } else {
      imgi.convertTo(imgc1, CV_32F);
      imgc1 = imgc1 / sca;
    }

    // cout << "Image Channels:  " << imgi.channels() << '\n';

    nmR = new char[strlen(argv[1]) + 3];
    nmG = new char[strlen(argv[1]) + 3];
    nmB = new char[strlen(argv[1]) + 3];

    dtR = new char[strlen(argv[1]) + 3];
    dtG = new char[strlen(argv[1]) + 3];
    dtB = new char[strlen(argv[1]) + 3];
    dt = new char[strlen(argv[1]) + 3];

    i = 0;

    do {
      *(nmR + i) = *(argv[1] + i);
      *(nmG + i) = *(argv[1] + i);
      *(nmB + i) = *(argv[1] + i);
      *(dtR + i) = *(argv[1] + i);
      *(dtG + i) = *(argv[1] + i);
      *(dtB + i) = *(argv[1] + i);
      *(dt + i) = *(argv[1] + i);
      i++;
    } while (*(argv[1] + i) != '.');

    *(nmR + i) = '_';
    *(nmG + i) = '_';
    *(nmB + i) = '_';
    *(dtR + i) = '_';
    *(dtG + i) = '_';
    *(dtB + i) = '_';
    *(dt + i) = '_';

    *(nmR + i + 1) = 'R';
    *(nmG + i + 1) = 'G';
    *(nmB + i + 1) = 'B';
    *(dtR + i + 1) = 'R';
    *(dtG + i + 1) = 'G';
    *(dtB + i + 1) = 'B';
    *(dt + i + 1) = '.';
    *(dtR + i + 2) = '.';
    *(dtG + i + 2) = '.';
    *(dtB + i + 2) = '.';
    *(dt + i + 2) = 't';
    *(dtR + i + 3) = 't';
    *(dtG + i + 3) = 't';
    *(dtB + i + 3) = 't';
    *(dt + i + 3) = 'x';
    *(dtR + i + 4) = 'x';
    *(dtG + i + 4) = 'x';
    *(dtB + i + 4) = 'x';
    *(dt + i + 4) = 't';
    *(dtR + i + 5) = 't';
    *(dtG + i + 5) = 't';
    *(dtB + i + 5) = 't';
    *(dt + i + 5) = '\0';
    *(dtR + i + 6) = '\0';
    *(dtG + i + 6) = '\0';
    *(dtB + i + 6) = '\0';

    do {
      *(nmR + i + 2) = *(argv[1] + i);
      *(nmG + i + 2) = *(argv[1] + i);
      *(nmB + i + 2) = *(argv[1] + i);
      i++;
    } while (*(argv[1] + i - 1) != '\0');

    /*
    cout <<'"' << dtR << '"'<< '\n';
    cout <<'"' << dtG << '"'<< '\n';
    cout <<'"' << dtB << '"'<< '\n';
    cout <<'"' << dt << '"'<< '\n';
    */

    if (imgi.channels() != 1) {
      cout << "Noise strength Blue  :   " << NoiseDev(imgc1) << "  +/- "
           << NoiseDevErr(imgc1) << '\n';
      cout << "Noise strength Green :   " << NoiseDev(imgc2) << "  +/- "
           << NoiseDevErr(imgc2) << '\n';
      cout << "Noise strength Red   :   " << NoiseDev(imgc3) << "  +/- "
           << NoiseDevErr(imgc3) << '\n';

      // creates gradient histogram
      histB = GradHist(imgc1);
      histG = GradHist(imgc2);
      histR = GradHist(imgc3);

      int bins = 256;
      int scale = 1;
      hist_imgR = cv::Mat(cv::Size(bins * scale, bins * scale), CV_8UC3);
      hist_imgG = cv::Mat(cv::Size(bins * scale, bins * scale), CV_8UC3);
      hist_imgB = cv::Mat(cv::Size(bins * scale, bins * scale), CV_8UC3);
      hist_imgR = 0;
      hist_imgG = 0;
      hist_imgB = 0;

      double max_valueR = 0;
      double max_valueG = 0;
      double max_valueB = 0;
      cv::minMaxLoc(histR, nullptr, &max_valueR, nullptr, nullptr);
      cv::minMaxLoc(histG, nullptr, &max_valueG, nullptr, nullptr);
      cv::minMaxLoc(histB, nullptr, &max_valueB, nullptr, nullptr);

      for (int h = 0; h < bins; h++) {
        float bin_valR = histR.at<float>(h);
        float bin_valG = histG.at<float>(h);
        float bin_valB =
            histB.at<float>(h);  // = cvQueryHistValue_1D(histB, h);
        int intensityR = cvRound(0.99 * bin_valR * bins * scale / max_valueR);
        int intensityG = cvRound(0.99 * bin_valG * bins * scale / max_valueG);
        int intensityB = cvRound(0.99 * bin_valB * bins * scale / max_valueB);
        cv::line(hist_imgR, cv::Point(h * scale, bins * scale),
                 cv::Point(h * scale, bins * scale - intensityR),
                 cv::Scalar(0, 0, 255));
        cv::line(hist_imgG, cv::Point(h * scale, bins * scale),
                 cv::Point(h * scale, bins * scale - intensityG),
                 cv::Scalar(0, 255, 0));
        cv::line(hist_imgB, cv::Point(h * scale, bins * scale),
                 cv::Point(h * scale, bins * scale - intensityB),
                 cv::Scalar(255, 0, 0));
        // cout << h << "  " << bin_valR << "  " << bin_valG << "  " << bin_valB
        // << '\n';
      }

      cv::namedWindow("Grad Hist_R", 1);
      cv::imshow("Grad Hist_R", hist_imgR);
      cv::namedWindow("Grad Hist_G", 1);
      cv::imshow("Grad Hist_G", hist_imgG);
      cv::namedWindow("Grad Hist_B", 1);
      cv::imshow("Grad Hist_B", hist_imgB);

      /*
      cv::dft(imgc1,imgc1);
      cv::dft(imgc2,imgc2);
      cv::dft(imgc3,imgc3);

      ofstream rfile (dtR);
      ofstream gfile (dtG);
      ofstream bfile (dtB);

      for(int row=0;row<imgc1.rows;row++)
      {
              for(int col=0;col<(imgc1.cols/2)+1;col++)
                      {
                              FGet2D(imgc1, col, row, &re, &im);
                              bfile<< row << " " << col << " " << re << " " <<
      im << '\n'; FGet2D(imgc2, col, row, &re, &im); gfile<< row << " " << col
      << " " << re << " " << im << '\n'; FGet2D(imgc3, col, row, &re, &im);
                              rfile<< row << " " << col << " " << re << " " <<
      im << '\n';

                      }
      }

      rfile.close();
      gfile.close();
      bfile.close();
      */
      cv::namedWindow("Channel 1 Blue", cv::WINDOW_AUTOSIZE);
      cv::imshow("Channel 1 Blue", imgc1);
      cv::namedWindow("Channel 2 Green", cv::WINDOW_AUTOSIZE);
      cv::imshow("Channel 2 Green", imgc2);
      cv::namedWindow("Channel 3 Red", cv::WINDOW_AUTOSIZE);
      cv::imshow("Channel 3 Red", imgc3);
    } else {
      cout << "Noise strength:    " << NoiseDev(imgc1) << "  +/- "
           << NoiseDevErr(imgc1) << '\n';

      // creates gradient histogram
      hist = GradHist(imgc1);

      int bins = 256;
      int scale = 1;
      hist_img = cv::Mat(cv::Size(bins * scale, bins * scale), CV_8UC1);
      hist_img = 0;

      double max_value = 0;
      cv::minMaxLoc(hist, nullptr, &max_value, nullptr, nullptr);

      for (int h = 0; h < bins; h++) {
        float bin_val = hist.at<float>(h);
        int intensity = cvRound(0.99 * bin_val * bins * scale / max_value);
        cv::line(hist_img, cv::Point(h * scale, bins * scale),
                 cv::Point(h * scale, bins * scale - intensity),
                 cv::Scalar(255));
        // cout << h << "  " << bin_val << '\n';
      }

      cv::namedWindow("Grad Hist", 1);
      cv::imshow("Grad Hist", hist_img);

      // equation solving
      float histfit[bins], temphist[bins], rsq;

      a0.at<float>(0, 0) = 0.05;
      a0.at<float>(1, 0) = 120.0;
      a0.at<float>(2, 0) = 127.5;
      a0.at<float>(3, 0) = 5.0;
      a0.at<float>(4, 0) = 28.0;
      a0.at<float>(5, 0) = 2.0;

      rsq = 0;
      for (int h = 0; h < bins; h++) {
        histfit[h] = hist.at<float>(h);
        temphist[h] =
            DL(h, a0.at<float>(0, 0), a0.at<float>(1, 0), a0.at<float>(2, 0),
               a0.at<float>(3, 0), a0.at<float>(4, 0), a0.at<float>(5, 0));
        rsq += (histfit[h] - temphist[h]) * (histfit[h] - temphist[h]);
      }

      cout << rsq << '\n';

      /*
      cv::dft(imgc1,imgc1);

      ofstream dfile (dt);

      for(int row=0;row<imgc1.rows;row++)
      {
              for(int col=0;col<(imgc1.cols/2)+1;col++)
                      {
                              FGet2D(imgc1, col, row, &re, &im);
                              dfile<< row << " " << col << " " << re << " " <<
      im << '\n';

                      }
      }
      dfile.close();
      */
      cv::namedWindow("Channel", cv::WINDOW_AUTOSIZE);
      cv::imshow("Channel", imgc1);
    }

    cv::waitKey(0);

    cv::destroyWindow("Initial");

    if (imgi.channels() != 1) {
      cv::destroyWindow("Channel 2 Green");
      cv::destroyWindow("Channel 3 Red");
      /*
      imgc1 *= 255;
      cv::imwrite(nmB,imgc1);
      imgc2 *= 255;
      cv::imwrite(nmG,imgc1);
      imgc3 *= 255;
      cv::imwrite(nmR,imgc1);
      */
    } else {
      cv::destroyWindow("Channel");
      cv::destroyWindow("Grad Hist");
    }

    delete[] nmR;
    delete[] nmG;
    delete[] nmB;
    delete[] dtR;
    delete[] dtG;
    delete[] dtB;
    delete[] dt;
  }

  return 0;
}