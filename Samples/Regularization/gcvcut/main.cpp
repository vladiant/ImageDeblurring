/*
 * This code performs Fourier motion deblurring
 * GCV functional is used
 * with a simple thresholding of the
 * Fourier coefficients
 */

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "blurrf.hpp"
#include "fdb.hpp"
#include "noise.hpp"

using namespace std;

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
 * This function is used to calculate the
 * GCV functional based on three
 * Fourier transformed images:
 * imgh - image for blurr operator
 * imgq - image for regularization operator
 * imgg - blurred image
 */
float GCVcut(float lambda, cv::Mat &imgg, cv::Mat &imgh) {
  float a2, a3, s1 = 0, s2 = 0,
                sum = 0;  // temporal variables for matrix inversion
  int w, h, h2, w2;       // image width and height, help variables

  w = imgg.cols;
  h = imgg.rows;

  w2 = ((w % 2 == 0) ? w - 2 : w - 1);
  h2 = ((h % 2 == 0) ? h - 2 : h - 1);

  // proceeds upper left
  a2 = (((float *)(imgh.data))[0]) * (((float *)(imgh.data))[0]);
  a3 = (((float *)(imgg.data))[0]) * (((float *)(imgg.data))[0]);
  s1 += a3 / ((a2) * (a2));
  s2 += 1.0 / (a2);

  // set first column
  for (int row = 1; row < h2; row += 2) {
    a2 = (((float *)(imgh.data + row * imgh.step))[0]) *
             (((float *)(imgh.data + row * imgh.step))[0]) +
         (((float *)(imgh.data + (row + 1) * imgh.step))[0]) *
             (((float *)(imgh.data + (row + 1) * imgh.step))[0]);
    a3 = (((float *)(imgg.data + row * imgg.step))[0]) *
             (((float *)(imgg.data + row * imgg.step))[0]) +
         (((float *)(imgg.data + (row + 1) * imgg.step))[0]) *
             (((float *)(imgg.data + (row + 1) * imgg.step))[0]);
    if (row < (1 - lambda) * h2) {
      s1 += a3 / ((a2) * (a2));
      s2 += 1.0 / (a2);
    }
  }

  // sets down left if needed
  if (h % 2 == 0) {
    a2 = (((float *)(imgh.data + (h - 1) * imgh.step))[0]) *
         (((float *)(imgh.data + (h - 1) * imgh.step))[0]);
    a3 = (((float *)(imgg.data + (h - 1) * imgg.step))[0]) *
         (((float *)(imgg.data + (h - 1) * imgg.step))[0]);
    s1 += a3 / ((a2) * (a2));
    s2 += 1.0 / (a2);
  }

  if (w % 2 == 0) {
    // sets upper right
    a2 = (((float *)(imgh.data))[w - 1]) * (((float *)(imgh.data))[w - 1]);
    a3 = (((float *)(imgg.data))[w - 1]) * (((float *)(imgg.data))[w - 1]);
    if (lambda == 0) {
      s1 += a3 / ((a2) * (a2));
      s2 += 1.0 / (a2);
    }

    // set last column
    for (int row = 1; row < h2; row += 2) {
      a2 = (((float *)(imgh.data + row * imgh.step))[w - 1]) *
               (((float *)(imgh.data + row * imgh.step))[w - 1]) +
           (((float *)(imgh.data + (row + 1) * imgh.step))[w - 1]) *
               (((float *)(imgh.data + (row + 1) * imgh.step))[w - 1]);
      a3 = (((float *)(imgg.data + row * imgg.step))[w - 1]) *
               (((float *)(imgg.data + row * imgg.step))[w - 1]) +
           (((float *)(imgg.data + (row + 1) * imgg.step))[w - 1]) *
               (((float *)(imgg.data + (row + 1) * imgg.step))[w - 1]);
      if (lambda == 0) {
        s1 += a3 / ((a2) * (a2));
        s2 += 1.0 / (a2);
      }
    }

    // sets down right
    if (h % 2 == 0) {
      a2 = (((float *)(imgh.data + (h - 1) * imgh.step))[w - 1]) *
           (((float *)(imgh.data + (h - 1) * imgh.step))[w - 1]);
      a3 = (((float *)(imgg.data + (h - 1) * imgg.step))[w - 1]) *
           (((float *)(imgg.data + (h - 1) * imgg.step))[w - 1]);
      if (lambda == 0) {
        s1 += a3 / ((a2) * (a2));
        s2 += 1.0 / (a2);
      }
    }
  }

  for (int row = 0; row < h; row++) {
    for (int col = 1; col < w2; col += 2) {
      a2 = (((float *)(imgh.data + row * imgh.step))[col]) *
               (((float *)(imgh.data + row * imgh.step))[col]) +
           (((float *)(imgh.data + row * imgh.step))[col + 1]) *
               (((float *)(imgh.data + row * imgh.step))[col + 1]);
      a3 = (((float *)(imgg.data + row * imgg.step))[col]) *
               (((float *)(imgg.data + row * imgg.step))[col]) +
           (((float *)(imgg.data + row * imgg.step))[col + 1]) *
               (((float *)(imgg.data + row * imgg.step))[col + 1]);
      if (((row < (1 - lambda) * h2) || ((h - row) < (1 - lambda) * h2)) &&
          (col < (1 - lambda) * w2)) {
        s1 += a3 / ((a2) * (a2));
        s2 += 1.0 / (a2);
      }
    }
  }
  sum = s1 / (s2 * s2);
  return (sum);
}

/*
 * This procedure is used to return the inverse of
 * the Fourier transform matrix,
 * regularized by a given operator
 */
void FMatInvcut(cv::Mat &imgw, float gamma) {
  float a1, a2, a3, a4, sum;  // temporal variables for matrix inversion
  int w, h, h2, w2;           // image width and height, help variables

  w = imgw.cols;
  h = imgw.rows;

  w2 = ((w % 2 == 0) ? w - 2 : w - 1);
  h2 = ((h % 2 == 0) ? h - 2 : h - 1);

  // sets upper left
  ((float *)(imgw.data))[0] =
      ((float *)(imgw.data))[0] /
      (((float *)(imgw.data))[0] * ((float *)(imgw.data))[0]);

  // set first column
  for (int row = 1; row < h2; row += 2) {
    a1 = ((float *)(imgw.data + row * imgw.step))[0];
    a2 = ((float *)(imgw.data + (row + 1) * imgw.step))[0];
    if (row < (1 - gamma) * h2) {
      sum = a1 * a1 + a2 * a2;
      ((float *)(imgw.data + row * imgw.step))[0] = a1 / sum;
      ((float *)(imgw.data + (row + 1) * imgw.step))[0] = -a2 / sum;
    } else {
      ((float *)(imgw.data + row * imgw.step))[0] = 0;
      ((float *)(imgw.data + (row + 1) * imgw.step))[0] = 0;
    }
  }

  // sets down left if needed
  if (h % 2 == 0) {
    ((float *)(imgw.data + (h - 1) * imgw.step))[0] =
        ((float *)(imgw.data + (h - 1) * imgw.step))[0] /
        (((float *)(imgw.data + (h - 1) * imgw.step))[0] *
         ((float *)(imgw.data + (h - 1) * imgw.step))[0]);
  }

  if (w % 2 == 0) {
    // sets upper right
    if (gamma == 0)
      ((float *)(imgw.data))[w - 1] =
          ((float *)(imgw.data))[w - 1] /
          (((float *)(imgw.data))[w - 1] * ((float *)(imgw.data))[w - 1]);
    else
      ((float *)(imgw.data))[w - 1] = 0;

    // set last column
    for (int row = 1; row < h2; row += 2) {
      a1 = ((float *)(imgw.data + row * imgw.step))[w - 1];
      a2 = ((float *)(imgw.data + (row + 1) * imgw.step))[w - 1];
      if (gamma == 0) {
        sum = a1 * a1 + a2 * a2;
        ((float *)(imgw.data + row * imgw.step))[w - 1] = a1 / sum;
        ((float *)(imgw.data + (row + 1) * imgw.step))[w - 1] = -a2 / sum;
      } else {
        ((float *)(imgw.data + row * imgw.step))[w - 1] = 0;
        ((float *)(imgw.data + (row + 1) * imgw.step))[w - 1] = 0;
      }
    }

    // sets down right
    if (h % 2 == 0) {
      if (gamma == 0)
        ((float *)(imgw.data + (h - 1) * imgw.step))[w - 1] =
            1.0 / ((float *)(imgw.data + (h - 1) * imgw.step))[w - 1];
      else
        ((float *)(imgw.data + (h - 1) * imgw.step))[w - 1] = 0;
    }
  }

  for (int row = 0; row < h; row++) {
    for (int col = 1; col < w2; col += 2) {
      a1 = ((float *)(imgw.data + row * imgw.step))[col];
      a2 = ((float *)(imgw.data + row * imgw.step))[col + 1];
      if (((row < (1 - gamma) * h2) || ((h - row) < (1 - gamma) * h2)) &&
          (col < (1 - gamma) * w2)) {
        sum = a1 * a1 + a2 * a2;
        ((float *)(imgw.data + row * imgw.step))[col] = a1 / sum;
        ((float *)(imgw.data + row * imgw.step))[col + 1] = (-a2 / sum);
      } else {
        ((float *)(imgw.data + row * imgw.step))[col] = 0;
        ((float *)(imgw.data + row * imgw.step))[col + 1] = 0;
      }
    }
  }
}

/*
 * This procedure does deconvolution
 * of two images via the DFT
 * with regularization
 * Note: Periodic Border Conditions!
 */

void FDFiltercut(cv::Mat &imga1, cv::Mat &imgb1, cv::Mat &imgc, float gamma) {
  cv::Mat imga = imga1.clone();
  cv::Mat imgb = imgb1.clone();

  // Forward Fourier Transform of initial and blurring image
  cv::dft(imga, imga);
  cv::dft(imgb, imgb);

  // inverts Fourier transformed blurred
  FMatInvcut(imgb, gamma);

  // blurring by multiplication
  cv::mulSpectrums(imga, imgb, imgb, 0);

  // Backward Fourier Transform
  cv::dft(imgb, imgc, cv::DFT_INVERSE | cv::DFT_SCALE);
}

int main(int argc, char **argv) {
  cv::Mat imgi, img0;  // initial images

  int m = 320, n = 240;   // test image dimensions (QVGA)
  float x1 = 10, y1 = 0;  // motion blurr vector
  int ksize, kw, kh;      // kernel width and height
  float alpha;            // this is alpha for Tikhonov regularization
  cv::Mat kernel;         // kernel for blurring
  cv::Mat img, imgl;
  // blurr and deblurred image
  cv::Mat img1, imgb;
  // images for the initial DFT and DCT transformations
  cv::Mat imgDFTi;
  // images for the DFT and DCT transformations
  cv::Mat imgDFT;
  // images for the DFT and DCT regularization operator
  cv::Mat imgDFTq, imgDFTqt;

  cv::Mat tmp;  // for border removal
  cv::Scalar ksum;

  // Norms declaration and solving variables
  float lambd, f1, f2, step;
  int iter;
  float lambd1, lambd2, lambd3, lambd4, lambd5;

  // regularized images
  cv::Mat img2, img3, img4, img5, img6;

  // cut off variables
  float re, im, stdev;

  // creates initial image
  if ((argc == 2) &&
      (!(imgi = cv::imread(argv[1], cv::IMREAD_GRAYSCALE)).empty())) {
    img0 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
    imgi.convertTo(img0, CV_32F);
    img0 = img0 / 255.0;
  } else {
    img0 = cv::Mat(cv::Size(m, n), CV_32FC1);
    Ximgset(img0);
  }

  img2 = cv::Mat(cv::Size(img0.cols, img0.rows), CV_8UC1);
  img3 = cv::Mat(cv::Size(img0.cols, img0.rows), CV_8UC1);

  // create blurr kernel
  ksize = max(abs(x1) + 1, abs(y1) + 1);
  kernel = cv::Mat(cv::Size(2 * ksize + 1, 2 * ksize + 1), CV_32FC1);
  ksetMoveXY(kernel, x1, y1);

  // declare blurred image
  img = cv::Mat(
      cv::Size(img0.cols + 2 * kernel.cols, img0.rows + 2 * kernel.rows),
      img0.depth(), 1);
  // temporal image
  imgl = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);
  // create border
  cv::copyMakeBorder(img0, img, kernel.rows, kernel.rows, kernel.cols,
                     kernel.cols, cv::BORDER_REPLICATE);
  // supress ringing from boundaries
  EdgeTaper(img, kernel.cols, kernel.cols);

  // creates Fourier smearing image
  imgb = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);
  Blurrset(imgb, kernel);
  FCFilter(img, imgb, img);

  // declare the output image
  img1 = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC1);

  // iteration starts here
  stdev = 0.00;
  step = 0.001;
  cv::namedWindow("Initial Image", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("RCut Image", cv::WINDOW_AUTOSIZE);

  do {
    // declaring initial images
    imgDFTi = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);
    imgDFT = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);
    imgDFTq = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);
    imgDFTqt = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC1);

    /*
    //use this in standalone deblurring
    //create border
    cv::copyMakeBorder(imgi,img,kernel.rows, kernel.rows, kernel.cols,
    kernel.cols, cv::BORDER_REPLICATE);
    //supress ringing from boundaries
    EdgeTaper(img, kernel.cols, kernel.cols);
     */

    img.copyTo(imgDFTi);

    // add some salt and pepper noise
    // SPnoise(imgDFTi, stdev);

    // add some gauss noise
    Gnoise(imgDFTi, 0.0, stdev);

    // add some poisson noise
    // Pnoise(imgDFTi, stdev*50000);

    // add some speckle noise
    // SPEnoise(imgDFTi, stdev);

    // add some uniform noise
    // Unoise(imgDFTi, 1.00, stdev);

    cv::imshow("Initial Image", imgDFTi);

    // deblurring starts here

    // temporal matrix for regularization operator
    imgDFTi.copyTo(imgDFTqt);
    // store the blurr operator - later image will be reused
    imgb.copyTo(imgDFT);
    cv::dft(imgDFT, imgDFT);  // imgb

    // Fourier inversion
    cv::dft(imgDFTqt, imgDFTqt);
    FMatInv(imgDFTqt, 0);

    // regularization operator applied
    imgDFTi.copyTo(imgDFTq);

    // Fourier transform
    cv::dft(imgDFTq, imgDFTq);
    // multiplication to get the values of q
    cv::mulSpectrums(imgDFTq, imgDFTqt, imgDFTq, 0);  // imgq

    // reuse the temporal image for blurred image
    cv::dft(imgDFTi, imgDFTqt);  // imgg

    // solve it
    lambd = 1.0;
    step = -2.0 / min(imgDFT.cols, imgDFT.rows);
    f1 = GCVcut(lambd, imgDFTqt, imgDFT);
    f2 = GCVcut(lambd + step, imgDFTqt, imgDFT);
    do {
      f1 = GCVcut(lambd, imgDFTqt, imgDFT);
      f2 = GCVcut(lambd + step, imgDFTqt, imgDFT);
      lambd += step;
      lambd = (lambd < 0) ? 0 : lambd;
      // cout << lambd << '\n';
    } while (f2 < f1);

    /*
    for (lambd =0; lambd<1; lambd+=1e-4)
    {
            cout << lambd << "  " << GCVcut(lambd, imgDFTqt, imgDFT) << '\n';
            //FDFiltercut(imgDFTi,imgb,imgDFT, lambd*lambd);
            //cv::imshow("RCut Image",imgDFT);
            //cout << lambd << '\n';
            //cv::waitKey(10);
    }
    */
    cout << stdev << "  " << lambd << "  " << '\n';
    // deblurring with regularization
    FDFiltercut(imgDFTi, imgb, imgDFT, lambd);
    cv::imshow("RCut Image", imgDFT);

    char c = cv::waitKey(10);
    if (c == 27) stdev = 1.0;

    // variables to be exported outside the function
    kw = kernel.cols;
    kh = kernel.rows;

    // cvScale(imgDFT,img2,255);
    // borders for boundary conditions
    // cv::Rect(img1,
    // cv::Point(kw,kh),cv::Point(img0.cols-kw,img0.rows-kh),cv::Scalar(255));

    stdev += 0.00005;
  } while (stdev < 0.01);

  // remove borders
  // cvGetSubRect(img1, &tmp, cvRect(kw, kh, img2.cols, img2.rows));
  // cvCopy(&tmp,img2);

  // borders for partially restored zone
  // cv::Rect(img2,
  // cv::Point(kw/2,kh/2),cv::Point(img0.cols-kw/2,img0.rows-kh/2),cv::Scalar(255));

  cout << "\n Press 'Esc' Key to Exit \n";
  while (1) {
    char c = cv::waitKey(0);
    if (c == 27) break;
  }
  cv::destroyWindow("Initial Image");
  cv::destroyWindow("RCut Image");

  return (0);
}