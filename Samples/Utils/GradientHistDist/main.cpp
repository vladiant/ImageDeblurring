//============================================================================
// Author      : Vladislav Antonov
// Version     :
// Description : Program to analyze gradient distribution and to
//             : estimate natural priors as defined in paper:
//             : D. Krishnan, R. Fergus.
//             : Fast Image Deconvolution using Hyper-Laplacian Priors
//             : Neural Information Processing Systems 2009
// Created on  : April 26, 2012
//============================================================================

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

double DL(double x, double a[]) {
  double result;
  result = a[0] + (a[1] / ((x - a[2]) * (x - a[2]) + a[3] * a[3])) +
           (a[4] / ((x - a[2]) * (x - a[2]) + a[5] * a[5]));
  return (result);
}

/*
 *  Function to calculate fit residual
 */

double Residual(const cv::Mat &hist, int bins, double a[]) {
  double rsq = 0;
  double histfit[bins], temphist[bins];

  int scale = 1;
  float minhist = 0.1 / (bins * scale);
  double max_value = 0;
  double min_value = 0;
  cv::minMaxLoc(hist, &min_value, &max_value, 0, 0);

  for (int h = 0; h < bins; h++) {
    histfit[h] = cvRound(log(hist.at<float>(h)));
    // hist.at<float>(h);
    // cvRound( 0.99 *
    // log(hist.at<float>(h)/(min_value+minhist))/log(max_value/(min_value+minhist)));
    temphist[h] = DL(h, a);
    rsq += (histfit[h] - temphist[h]) * (histfit[h] - temphist[h]);
  }
  return (rsq);
}

/*
 *  Function to calculate gradient of the residual
 */

double dRdx(cv::Mat &hist, int bins, double a[], int index) {
  double value = 0;
  double deps = 1e-3;

  double r0 = Residual(hist, bins, a);

  double tmp_a = a[index];
  a[index] *= 1.0 + deps;

  double r1 = Residual(hist, bins, a);
  a[index] = tmp_a;

  value = (r1 - r0) / deps;

  return (value);
}

/*
 *  Function to calculate gradient of the residual - exact derivatives
 */
/*
double dRdxE (cv::Mat&  hist, int bins, int index)
{
        double value = 0;


        return (value);
}
*/

int main(int argc, char *argv[]) {
  cv::Mat img, imgi, imgc1, imgc2, imgc3, imgc1i, imgc2i, imgc3i;
  double sca = 1.0;
  int i;

  cv::Mat imgdt1, imgdt2;  // test image for the derivatives
  cv::Mat hist_img, hist_imgR, hist_imgG, hist_imgB,
      hist_imgAll;  // histogram images
  cv::Mat hist, histR, histG, histB;

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
        break;
    }

    // displays image
    cv::namedWindow("Initial", cv::WINDOW_AUTOSIZE);

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

    i = 0;

    if (imgi.channels() != 1) {
      // creates gradient histogram
      histB = GradHist(imgc1);
      histG = GradHist(imgc2);
      histR = GradHist(imgc3);

      int bins = 256;
      int scale = 1;
      float minhist = 0.1 / (bins * scale);
      hist_imgR = cv::Mat(cv::Size(bins * scale, bins * scale), CV_8UC3);
      hist_imgG = cv::Mat(cv::Size(bins * scale, bins * scale), CV_8UC3);
      hist_imgB = cv::Mat(cv::Size(bins * scale, bins * scale), CV_8UC3);
      hist_imgAll = cv::Mat(cv::Size(bins * scale, bins * scale), CV_8UC3);
      hist_imgR = 0;
      hist_imgG = 0;
      hist_imgB = 0;
      hist_imgAll = 0;

      double max_valueR = 0;
      double max_valueG = 0;
      double max_valueB = 0;
      double min_valueR = 0;
      double min_valueG = 0;
      double min_valueB = 0;
      cv::minMaxLoc(histR, &min_valueR, &max_valueR, nullptr, nullptr);
      cv::minMaxLoc(histG, &min_valueG, &max_valueG, nullptr, nullptr);
      cv::minMaxLoc(histB, &min_valueB, &max_valueB, nullptr, nullptr);

      for (int h = 0; h < bins; h++) {
        float bin_valR = histR.at<float>(h);
        float bin_valG = histG.at<float>(h);
        float bin_valB = histB.at<float>(h);
        int intensityR = cvRound(0.99 * bins * scale *
                                 log(bin_valR / (min_valueR + minhist)) /
                                 log(max_valueR / (min_valueR + minhist)));
        int intensityG = cvRound(0.99 * bins * scale *
                                 log(bin_valG / (min_valueG + minhist)) /
                                 log(max_valueG / (min_valueG + minhist)));
        int intensityB = cvRound(0.99 * bins * scale *
                                 log(bin_valB / (min_valueB + minhist)) /
                                 log(max_valueB / (min_valueB + minhist)));
        cv::line(hist_imgR, cv::Point(h * scale, bins * scale),
                 cv::Point(h * scale, bins * scale - intensityR),
                 cv::Scalar(0, 0, 255));
        cv::line(hist_imgG, cv::Point(h * scale, bins * scale),
                 cv::Point(h * scale, bins * scale - intensityG),
                 cv::Scalar(0, 255, 0));
        cv::line(hist_imgB, cv::Point(h * scale, bins * scale),
                 cv::Point(h * scale, bins * scale - intensityB),
                 cv::Scalar(255, 0, 0));
      }

      cv::namedWindow("Grad Hist_R", 1);
      cv::imshow("Grad Hist_R", hist_imgR);
      cv::namedWindow("Grad Hist_G", 1);
      cv::imshow("Grad Hist_G", hist_imgG);
      cv::namedWindow("Grad Hist_B", 1);
      cv::imshow("Grad Hist_B", hist_imgB);

      hist_imgAll = hist_imgAll + hist_imgR;
      hist_imgAll = hist_imgAll + hist_imgG;
      hist_imgAll = hist_imgAll + hist_imgB;

      cv::namedWindow("Grad Hist", 1);
      cv::imshow("Grad Hist", hist_imgAll);

      cv::namedWindow("Channel 1 Blue", cv::WINDOW_AUTOSIZE);
      cv::imshow("Channel 1 Blue", imgc1);
      cv::namedWindow("Channel 2 Green", cv::WINDOW_AUTOSIZE);
      cv::imshow("Channel 2 Green", imgc2);
      cv::namedWindow("Channel 3 Red", cv::WINDOW_AUTOSIZE);
      cv::imshow("Channel 3 Red", imgc3);
    } else {
      // creates gradient histogram
      hist = GradHist(imgc1);

      int bins = 256;
      int scale = 1;
      float minhist = 0.1 / (bins * scale);
      hist_img = cv::Mat(cv::Size(bins * scale, bins * scale), CV_8UC1);
      hist_img = 0;

      double max_value = 0;
      double min_value = 0;
      cv::minMaxLoc(hist, &min_value, &max_value, nullptr, nullptr);

      for (int h = 0; h < bins; h++) {
        float bin_val = hist.at<float>(h);
        int intensity =
            cvRound(0.99 * bins * scale * log(bin_val / (min_value + minhist)) /
                    log(max_value / (min_value + minhist)));
        cv::line(hist_img, cv::Point(h * scale, bins * scale),
                 cv::Point(h * scale, bins * scale - intensity),
                 cv::Scalar(255));
      }

      cv::namedWindow("Grad Hist", 1);
      cv::imshow("Grad Hist", hist_img);

      /*
       *  example coefficients for fit
       *  a0 = 0.05; a1 = 120.0; a2 = 127.5; a3 = 5.0;
       *  a4 = 28.0; a5 = 2.0;
       */
      double FitCoef[6] = {0.05, 120.0, 127.5, 5.0, 28.0, 2.0};

      // for (int i=0; i<6; i++) cout << dRdx(hist, bins, FitCoef, i) << endl;

      // cout << '\n' << Residual(hist, bins, FitCoef) << endl;
      /*
                                                      for( int h = 0; h < bins;
         h++ )
                                                              {
                                                                      cout << h
         << "  " << ((histat<float>(h))) << endl;
                                                               }
      */
      // fit procedure starts here
      /*
                                                      //arrays for blind
         deblurring - x, r, p, A.p double Xbd[6], Rbd[6], Pbd[6], APbd[6];
                                                      //initial initialization
         of arrays double resid=0; for (int i=0; i<6; i++) Rbd[i] = -dRdx(hist,
         bins, FitCoef, i); for (int i=0; i<6; i++) resid+=Rbd[i]*Rbd[i]; for
         (int i=0; i<6; i++) Pbd[i] = Rbd[i]; for (int i=0; i<6; i++) Xbd[i] =
         FitCoef[i];

                                                      cout <<"x" << endl;
                                                      for (int i=0; i<6; i++)
         cout << Xbd[i] << "  "; cout << resid << endl;

                                                      //simple iteration loop
                                                      int iter=0;
                                                      double ak=1e-11;
                                                      do
                                                      {
                                                              resid = 0;
                                                              //x = x + a*r
                                                              for (int i=0; i<6;
         i++) Xbd[i]+=ak*Rbd[i];

                                                              //cout <<"x" << "
         ";
                                                              //for (int i=0;
         i<6; i++) cout << Rbd[i] << "  ";
                                                              //cout <<"\nr" <<
         "  ";
                                                              //for (int i=0;
         i<6; i++) cout << Rbd[i] << "  ";
                                                              //cout << endl;

                                                              //new residual
                                                              for (int i=0; i<6;
         i++) Rbd[i] = -dRdx(hist, bins, Xbd, i);

                                                              for (int i=0; i<6;
         i++) resid+=Rbd[i]*Rbd[i]; cout << resid << endl;

                                                              iter++;
                                                      } while (resid > 1e-8);

                                                      cout <<"x" << endl;
                                                      for (int i=0; i<6; i++)
         cout << Xbd[i] << "  "; cout << endl;
      */
      /*
                                                      //CG loop
                                                      double ak, bk;
                                                      int iter=0;
                                                      do
                                                      {
                                                              resid = 0;
                                                              // A.p
                                                              for (int i=0; i<6;
         i++) APbd[i] = dRdx(hist, bins, Pbd, i);

                                                              //alpha_k
                                                              ak=0;
                                                              double a1=0;
                                                              for (int i=0; i<6;
         i++)
                                                              {
                                                                      a1+=
         Rbd[i]*Rbd[i]; ak+= Pbd[i]*APbd[i];
                                                              }
                                                              ak=a1/ak;

                                                              //x = x + ak*pk
                                                              for (int i=0; i<6;
         i++) Xbd[i]+=ak*Pbd[i];

                                                              cout <<"x" <<
         endl; for (int i=0; i<6; i++) cout << Xbd[i] << "  ";

                                                              //betha_k
                                                              double b1=0;
                                                              for (int i=0; i<6;
         i++)
                                                              {
                                                                      //b1+=(Rbd[i]-ak*APbd[i])*Rbd[i];
                                                                      //r =
         r-ak*A.p Rbd[i]-=ak*APbd[i];

                                                                      //double
         a1 = Rbd[i];
                                                                      //Rbd[i] =
         -dRdx(hist, bins, Xbd, i);
                                                                      //b1+=(Rbd[i]-a1)*Rbd[i];

                                                                      b1+=-(Rbd[i]+ak*APbd[i])*Rbd[i];
                                                                      //b1+=Rbd[i]*Rbd[i];
                                                              }
                                                              bk = b1/a1;

                                                              //p = r + bk*p
                                                              for (int i=0; i<6;
         i++) Pbd[i]=Rbd[i]+bk*Pbd[i];

                                                              for (int i=0; i<6;
         i++) resid+=Rbd[i]*Rbd[i]; cout << resid << endl;

                                                              iter++;

                                                      } while (iter < 1000000);
      */
      cv::namedWindow("Channel", cv::WINDOW_AUTOSIZE);
      cv::imshow("Channel", imgc1);
    }

    cv::waitKey(0);

    cv::destroyWindow("Initial");

    if (imgi.channels() != 1) {
      cv::destroyWindow("Channel 1 Blue");
      cv::destroyWindow("Channel 2 Green");
      cv::destroyWindow("Channel 3 Red");
    } else {
      cv::destroyWindow("Channel");
      cv::destroyWindow("Grad Hist");
    }
  }

  return 0;
}
