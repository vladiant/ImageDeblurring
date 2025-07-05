/*
 * Simple program to create
 * space variant motion blurr
 * Richardson Lucy algorithm
 * with partial sums
 * Gaussian distribution assumed
 * and noise is added
 */

#include <cv.h>
#include <highgui.h>

#include "/home/users/vantonov/Work/libs/blurrf.h"
#include "/home/users/vantonov/Work/libs/fdb.h"
#include "/home/users/vantonov/Work/libs/noise.h"

using namespace std;

// parameters of the noise
//  Gaussian: st. dev., mean; Uniform noise deviation; Salt'n'Pepper part;
//  Poisson noise coefficient; Specle noise coefficient
int pos_n1 = 0, pos_n2 = 0, pos_n3 = 0, pos_n4 = 0, pos_n5 = 0, pos_n6 = 0;

void Ximgset(IplImage *imga)  // generates the image
{
  cvZero(imga);
  // rectangle coordinates
  int i1 = round((imga->height) / 5), i2 = round(3 * (imga->width) / 5),
      j1 = round((imga->height) / 5), j2 = round(3 * (imga->width) / 5);

  // circle radius
  int r = round(max(imga->height, imga->width) / 5);

  // draws rectangle
  cvRectangle(imga, cvPoint(i1, j1), cvPoint(i2, j2), cvScalar(0.5), -1);

  // draws circle
  cvCircle(imga,
           cvPoint(round(5 * (imga->width) / 8), round(3 * (imga->height) / 5)),
           r, cvScalar(1.0), -1);
}

/*
 * Function which introduces
 * blurring kernel on image
 * with periodic boundary conditions
 * kernel is spatially dependent
 */

void BlurrPBCsv(IplImage *imga, IplImage *imgb, IplImage *imgc, int fl = 0) {
  float s1, s2, s3;
  int i, j;

  for (int row = 0; row < imga->height; row++) {
    // here the kernel is changed
    cvSetZero(imgb);
    if (fl == 0)
      ksetMoveXY(imgb, 5, 5);
    else
      ksetMoveXY(imgb, -5, -5);

    for (int col = 0; col < imga->width; col++) {
      s2 = 0;

      for (int row1 = 0; row1 < imgb->height; row1++) {
        for (int col1 = 0; col1 < imgb->width; col1++) {
          s1 = ((float *)(imgb->imageData + row1 * imgb->widthStep))[col1];

          if ((row - row1 + imgb->height / 2) >= 0) {
            if ((row - row1 + imgb->height / 2) < (imga->height)) {
              i = row - row1 + imgb->height / 2;
            } else {
              i = row - row1 + imgb->height / 2 - imga->height + 1;
            }

          } else {
            i = (row - row1 + imgb->height / 2) + imga->height - 1;
          }

          if ((col - col1 + imgb->width / 2) >= 0) {
            if ((col - col1 + imgb->width / 2) < (imga->width)) {
              j = col - col1 + imgb->width / 2;
            } else {
              j = col - col1 + imgb->width / 2 - imga->width + 1;
            }

          } else {
            j = (col - col1 + imgb->width / 2) + imga->width - 1;
          }

          s3 = ((float *)(imga->imageData + i * imga->widthStep))[j] * s1;
          s2 += s3;
        }
      }
      ((float *)(imgc->imageData + row * imgc->widthStep))[col] = s2;
    }
  }
}

/*
 * Total variation calculation
 */

float TotVar(IplImage *imga) {
  float s1, s2, s3 = 0;
  int x1, x2, y1, y2;

  for (int row = 0; row < imga->height; row++) {
    for (int col = 0; col < imga->width; col++) {
      if (row == (imga->height - 1)) {
        y1 = row;

      } else {
        y1 = row + 1;
      }

      if (col == (imga->width - 1)) {
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

      s1 = ((float *)(imga->imageData + row * imga->widthStep))[x2] -
           ((float *)(imga->imageData + row * imga->widthStep))[x1];
      s2 = ((float *)(imga->imageData + y2 * imga->widthStep))[col] -
           ((float *)(imga->imageData + y1 * imga->widthStep))[col];
      s3 += sqrt(s1 * s1 + s2 * s2);
    }
  }
  return (s3);
}

float NoiseDev(const IplImage *imga) {
  IplImage *imgd = cvCreateImage(cvGetSize(imga), IPL_DEPTH_32F, 1);
  float re, im, shar, sum;

  shar = 0.8;
  sum = 0.0;
  cvScale(imga, imgd, 1.0 / 255.0);
  cvDFT(imgd, imgd, CV_DXT_FORWARD, imgd->height);
  for (int row = int(shar * imgd->height / 2); row < (imgd->height / 2) - 1;
       row++) {
    for (int col = int(shar * imgd->width / 2); col < (imgd->width / 2) - 1;
         col++) {
      FGet2D(imgd, col, row, &re, &im);
      sum += sqrt(re * re + im * im);
    }
  }
  sum /= ((imgd->height / 2) - int(shar * imgd->height / 2)) *
         ((imgd->width / 2) - int(shar * imgd->width / 2)) *
         sqrt((imgd->height) * (imgd->width));
  cvReleaseImage(&imgd);
  return (sum);
}

// initial declaration of trackbar reading function
void GetNois(int pos);

int main(int argc, char *argv[]) {
  IplImage *img, *imgi, *img1, *img2, *img3, *img4, *img5,
      *imgl;  // initial, blurred, kernel, deblurred and noise image, laplace
  int m = 320, n = 240, r = 40,
      ksize;         // image dimensions and radius of blurring, kernel size
  IplImage *kernel;  // kernel for blurring
  int it;            // iteration counter
  const int x1 = 5, y1 = 5;                 // Define vector of motion blurr
  float norm, norm1, norm2, oldnorm, delt;  // norm of the images
  float lambda, gi;                         // regularization parameters

  // creates initial image
  if ((argc == 2) &&
      ((imgi = cvLoadImage(argv[1], CV_LOAD_IMAGE_GRAYSCALE)) != 0)) {
    img = cvCreateImage(cvGetSize(imgi), IPL_DEPTH_32F, 1);
    cvScale(imgi, img, 1.0 / 255.0);
  } else {
    img = cvCreateImage(cvSize(m, n), IPL_DEPTH_32F, 1);
    Ximgset(img);
  }

  img1 = cvCloneImage(img);
  imgl = cvCloneImage(img);

  // createst simple kernel for motion
  ksize = max(abs(x1), abs(y1));

  // creates blurring kernel
  kernel =
      cvCreateImage(cvSize(2 * ksize + 1, 2 * ksize + 1), IPL_DEPTH_32F, 1);

  // motion blurr
  // ksetMoveXY(kernel,x1,y1);

  // direct blurring via kernel
  BlurrPBCsv(img, kernel, img1);

  // displays image
  cvNamedWindow("Initial", CV_WINDOW_AUTOSIZE);
  cvShowImage("Initial", img);
  cvNamedWindow("Blurred", CV_WINDOW_AUTOSIZE);

  cvNamedWindow("Motion Blurr", CV_WINDOW_NORMAL);
  cvResizeWindow("Motion Blurr", 600, 1);
  cvCreateTrackbar("Gaussian noise dev., % ", "Motion Blurr", &pos_n1, 100,
                   GetNois);
  cvCreateTrackbar("Gaussian noise mean, % ", "Motion Blurr", &pos_n2, 100,
                   GetNois);
  cvCreateTrackbar("Uniform noise dev, % ", "Motion Blurr", &pos_n3, 100,
                   GetNois);
  cvCreateTrackbar("Bad Pixels, o/oo ", "Motion Blurr", &pos_n4, 100, GetNois);
  cvCreateTrackbar("Poisson noise coeff.", "Motion Blurr", &pos_n5, 500,
                   GetNois);
  cvCreateTrackbar("Speckle noise coeff.", "Motion Blurr", &pos_n6, 100,
                   GetNois);

  // set the noise
  do {
    cvCopy(img1, imgl);
    Gnoise(imgl, pos_n2 / 100.0, pos_n1 / 100.0);
    Unoise(imgl, pos_n3 / 100.0, 1.0);
    SPnoise(imgl, pos_n4 / 2000.0);
    if (pos_n5 != 0) Pnoise(imgl, pos_n5);
    SPEnoise(imgl, pos_n6 / 100.0);
    cvShowImage("Blurred", imgl);

    char c = cvWaitKey(10);
    if (c == 10) break;  // press Enter to continue
  } while (1);

  cvCopy(imgl, img1);

  // initial parameter of the norm
  gi = 2 * cvNorm(img1, 0, CV_L2);

  // displays image
  cvShowImage("Blurred", img1);
  cvNamedWindow("Deblurred", CV_WINDOW_AUTOSIZE);

  // Fourier derivation of the kernel
  img2 = cvCloneImage(img1);  // old image
  img3 = cvCloneImage(img1);  // new image
  img4 = cvCloneImage(img1);  // blurred old image
  img5 = cvCloneImage(img1);

  BlurrPBCsv(img2, kernel, img4);
  // norm2=TotVar(img2)/((img2->width)*(img2->height));
  cvLaplace(img2, imgl);
  norm2 = cvNorm(imgl, 0, CV_L2) / ((img2->width) * (img2->height));
  cvSub(img4, img1, img5);
  norm1 = cvNorm(img5, 0, CV_L2) / ((img2->width) * (img2->height));
  lambda = norm1 / (gi - norm2);
  cout << "Initial lambda= " << lambda << '\n';
  oldnorm = norm1 + lambda * norm2 + 1;

  // Richardson-Lucy starts here
  it = 0;
  do {
    // Mk=H*Ik
    BlurrPBCsv(img2, kernel, img4);

    // norm2=TotVar(img2)/((img2->width)*(img2->height));
    cvLaplace(img2, imgl);
    norm2 = cvNorm(imgl, 0, CV_L2) / ((img2->width) * (img2->height));
    cvSub(img4, img1, img5);  // r=Ax-b
    norm1 = cvNorm(img5, 0, CV_L2) / ((img2->width) * (img2->height));
    lambda = norm1 / (gi - norm2);

    // Ht*(Mk)
    BlurrPBCsv(img4, kernel, img5, 1);
    // Ht*Ib
    BlurrPBCsv(img1, kernel, img4);
    // Ht*Ib/(Ht*(Mk))
    cvDiv(img4, img5, img4);
    // pixel by pixel multiply
    cvMul(img4, img2, img3);

    // additional part to add coefficient
    cvSub(img3, img2, img3);
    cvAddWeighted(img2, 1.0, img3, 1.0, 0.0, img3);

    norm = norm1 + lambda * norm2;
    delt = oldnorm - norm;
    oldnorm = norm;

    img2 = cvCloneImage(img3);

    cvShowImage("Deblurred", img3);
    char c = cvWaitKey(10);

    if (c == 27) break;

    it++;
    cout << it << "  " << delt << "  " << norm << "  " << norm1 << "  " << norm2
         << "  " << lambda << '\n';
  } while (abs(delt) > 1e-7);

  cvWaitKey(0);

  cvDestroyWindow("Initial");
  cvDestroyWindow("Blurred");
  cvDestroyWindow("Deblurred");
  cvDestroyWindow("Deblurred");

  cvReleaseImage(&img);
  cvReleaseImage(&img1);
  cvReleaseImage(&img2);
  cvReleaseImage(&img3);
  cvReleaseImage(&kernel);
  return 0;
}

void GetNois(int pos) {
  pos_n1 = cvGetTrackbarPos("Gaussian noise dev., % ", "Motion Blurr");
  pos_n2 = cvGetTrackbarPos("Gaussian noise mean, % ", "Motion Blurr");
  pos_n3 = cvGetTrackbarPos("Uniform noise dev, % ", "Motion Blurr");
  pos_n4 = cvGetTrackbarPos("Bad Pixels, o/oo ", "Motion Blurr");
  pos_n5 = cvGetTrackbarPos("Poisson noise coeff.", "Motion Blurr");
  pos_n6 = cvGetTrackbarPos("Speckle noise coeff.", "Motion Blurr");
}