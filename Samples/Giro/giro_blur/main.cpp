/*
 * This is a demo program
 * used to introduce
 * motion blur for images
 * using a gyro data calculated kernel
 * Boundaries condition: replicate
 *
 * Gyro data read from cin
 * (can be given from file)
 *
 * Created by Vladislav Antonov
 * March 2012
 *
 * ./giro_blur image < giro_data.txt
 *
 */

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

using namespace std;

bool flag = true;  // false if kernel loaded

float filt;        // Filter value
double sca = 1.0;  // scaling factor

// time for readout in seconds
const float time_readout = 0.03283;

// exposure time in seconds - to be read from EXIF Data!!!
const float time_exposure = 0.0599;

// Field of view for intrinsic camera matrix K
const float FOV = 2592 * 195.5 / 189;

// time delay for the timestamps - to be calibrated
const float time_delay = 0.02;

// initial, loaded, merged (float), blue/gray, green, red (float),  blue/gray,
// green, red (initial) images
cv::Mat imgi, img, imgc, imgc1, imgc2, imgc3, imgc1i, imgc2i, imgc3i;

// images to be blurred
cv::Mat imgcb;
// images to be deblurred
cv::Mat imgcd, imgc1d, imgc2d, imgc3d;
// images to be shown - first blurred, then deblurred
cv::Mat img1b, img3b, img1d, img3d;

cv::Mat kernel, imgb,
    krnl;  // kernel, image for blurring, one for all, kernel image

/*
 * This function gives axes to the elements
 * of the packed Fourier transformed matrix.
 * It was found in internet address
 * http://mildew.ee.engr.uky.edu/~weisu/OpenCV/OpenCV_archives_16852-19727.htm
 * and proposed by Vadim Pisarevsky
 */
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
  imga.convertTo(imgd, CV_32F, 1.0 / 255.0);
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
 * Function which introduces
 * blurring kernel on image
 * with reflective boundary conditions
 * kernel is spatially dependent
 */

void BlurrPBCsv(cv::Mat &imga, int bsamples, float *timeb, float *Thetabx,
                float *Thetaby, float *Thetabz, float time_capture_start,
                cv::Mat &imgc1, int fl = 0) {
  float s1, s2, s3, s4, s5;
  int i, j;

  int imw = imga.cols, imh = imga.rows;  // image width, image height
  float u0 = imw / 2.0, v0 = imh / 2.0;  // principal point in pixels
  int r0x = 0 / 2,
      r0y = 0 / 2;  // pixel where the movement points will be drawn
  float x1, y1;     // current coordinates

  cv::Mat imgc = imgc1.clone();

  for (int row = 0; row < imga.rows; row++) {
    for (int col = 0; col < imga.cols; col++) {
      s2 = 0;

      // here comes the kernel calculation procedure
      //  non homogenious coordinates
      float x0 = (r0x - u0) / (FOV), y0 = (r0y - v0) / (FOV);

      // (x,y) coordinates of points, which correspond to blurring
      // alongside with angles an time
      float *xb = new float[bsamples];
      float *yb = new float[bsamples];
      float xbi, ybi, xbf, ybf, xbmax = 0, ybmax = 0;
      int *bstep = new int[bsamples - 1];  // steps for interpolation
      int bsteps = 0;

      // set the array of (x,y) points for blurring
      // point movement for the whole time interval
      for (int jk = 0; jk < bsamples; jk++) {
        float Rt[9];  // rotation matrix
        float c, s, cc;

        // calculation of the rotation matrix, based on actual angle,
        // to be applied to initial point
        float urx = Thetabx[jk], ury = Thetaby[jk], urz = Thetabz[jk];
        float sur = sqrt(urx * urx + ury * ury + urz * urz);
        urx /= sur;
        ury /= sur;
        urz /= sur;

        c = cos(sur), s = sin(sur), cc = 1 - c;
        Rt[0] = urx * urx * cc + c;
        Rt[1] = urx * ury * cc - urz * s;
        Rt[2] = urx * urz * cc + ury * s;
        Rt[3] = ury * urx * cc + urz * s;
        Rt[4] = ury * ury * cc + c;
        Rt[5] = ury * urz * cc - urx * s;
        Rt[6] = urz * urx * cc - ury * s;
        Rt[7] = urz * ury * cc + urx * s;
        Rt[8] = urz * urz * cc + c;

        x1 = (Rt[0] * x0 + Rt[1] * y0 + Rt[2] * 1.0) /
             (Rt[6] * x0 + Rt[7] * y0 + Rt[8] * 1.0);
        y1 = (Rt[3] * x0 + Rt[4] * y0 + Rt[5] * 1.0) /
             (Rt[6] * x0 + Rt[7] * y0 + Rt[8] * 1.0);
        xb[jk] = x1 * (FOV) + u0 - r0x;
        yb[jk] = y1 * (FOV) + v0 - r0y;
      }

      // set the initial and final points
      xbi = (xb[1] - xb[0]) * (time_capture_start + time_delay - timeb[0]) /
                (timeb[1] - timeb[0]) +
            xb[0];
      ybi = (yb[1] - yb[0]) * (time_capture_start + time_delay - timeb[0]) /
                (timeb[1] - timeb[0]) +
            yb[0];

      xbf = (xb[bsamples - 1] - xb[bsamples - 2]) *
                (time_capture_start + time_exposure + time_delay -
                 timeb[bsamples - 2]) /
                (timeb[bsamples - 1] - timeb[bsamples - 2]) +
            xb[bsamples - 2];
      ybf = (yb[bsamples - 1] - yb[bsamples - 2]) *
                (time_capture_start + time_exposure + time_delay -
                 timeb[bsamples - 2]) /
                (timeb[bsamples - 1] - timeb[bsamples - 2]) +
            yb[bsamples - 2];

      // the final set of points
      xb[0] = 0;
      yb[0] = 0;
      xb[bsamples - 1] = xbf - xbi;
      yb[bsamples - 1] = ybf - ybi;
      for (int jk = 1; jk < bsamples - 1; jk++) {
        xb[jk] -= xbi;
        yb[jk] -= ybi;
        if (abs(xb[jk]) > abs(xbmax)) xbmax = xb[jk];
        if (abs(yb[jk]) > abs(ybmax)) ybmax = yb[jk];
      }
      if (abs(xb[0]) > abs(xbmax)) xbmax = xb[0];
      if (abs(yb[0]) > abs(ybmax)) ybmax = yb[0];
      if (abs(xb[bsamples - 1]) > abs(xbmax)) xbmax = xb[bsamples - 1];
      if (abs(yb[bsamples - 1]) > abs(ybmax)) ybmax = yb[bsamples - 1];

      // calculate number of steps
      for (int jk = 0; jk < bsamples - 1; jk++) {
        bstep[jk] =
            int(2.0 * sqrt((xb[jk + 1] - xb[jk]) * (xb[jk + 1] - xb[jk]) +
                           (yb[jk + 1] - yb[jk]) * (yb[jk + 1] - yb[jk])) +
                0.5);
        bsteps += bstep[jk];
      }

      // sparse kernel
      int dims[] = {2 * int(abs(ybmax) + 0.5) + 11,
                    2 * int(abs(xbmax) + 0.5) + 11};
      cv::SparseMat skernel(2, dims, CV_32F);

      // set the kernel values
      for (int jk = 0; jk < bsamples - 1; jk++) {
        // float step=1.0/bstep[jk];
        for (int jj = 0; jj < bstep[jk]; jj++) {
          float xd =
              -((yb[jk + 1] - yb[jk]) * jj / bstep[jk] + yb[jk]) + dims[0] / 2;
          float yd =
              -((xb[jk + 1] - xb[jk]) * jj / bstep[jk] + xb[jk]) + dims[1] / 2;

          // sparse kernel
          int idx[] = {int(xd), int(yd)};
          skernel.ref<float>(idx) += (int(xd) - xd + 1) * (int(yd) - yd + 1);
          idx[0] = int(xd);
          idx[1] = int(yd + 1);
          skernel.ref<float>(idx) += (int(xd) - xd + 1) * (yd - int(yd));
          idx[0] = int(xd + 1);
          idx[1] = int(yd);
          skernel.ref<float>(idx) += (xd - int(xd)) * (int(yd) - yd + 1);
          idx[0] = int(xd + 1);
          idx[1] = int(yd + 1);
          skernel.ref<float>(idx) += (xd - int(xd)) * (yd - int(yd));
        }
      }

      float kernelsum = 0;
      // sparse kernel sum
      cv::SparseMatIterator it;
      for (it = skernel.begin(); it != skernel.end(); ++it) {
        kernelsum += it.value<float>();
      }

      // normalize sparse kernel
      for (it = skernel.begin(); it != skernel.end(); ++it) {
        it.value<float>() /= kernelsum;
      }

      // apply the kernel
      for (it = skernel.begin(); it != skernel.end(); ++it) {
        cv::SparseMat::Node *node = it.node();
        int *idx = node->idx;
        float val = it.value<float>();

        int row1, col1;

        // here the kernel is transposed
        if (fl != 0) {
          row1 = idx[1];
          col1 = idx[0];
        } else {
          row1 = dims[1] - 1 - idx[1];
          col1 = dims[0] - 1 - idx[0];
        }

        // mirror boundary conditions
        if ((row - row1 + dims[1] / 2) >= 0) {
          if ((row - row1 + dims[1] / 2) < (imga.rows)) {
            i = row - row1 + dims[1] / 2;
          } else {
            i = 2 * (imga.rows - 1) - (row - row1 + dims[1] / 2);
          }

        } else {
          i = abs(row - row1 + dims[1] / 2);
        }

        if ((col - col1 + dims[0] / 2) >= 0) {
          if ((col - col1 + dims[0] / 2) < (imga.cols)) {
            j = col - col1 + dims[0] / 2;
          } else {
            j = (imga.cols - 1) * 2 - (col - col1 + dims[0] / 2);
          }

        } else {
          j = abs(col - col1 + dims[0] / 2);
        }

        s3 = ((float *)(imga.data + i * imga.step))[j] * val;
        s2 += s3;
      }

      ((float *)(imgc.data + row * imgc.step))[col] = s2;
      delete[] xb;
      delete[] yb;
      delete[] bstep;
    }
  }

  imgc.copyTo(imgc1);
}

void Process(
    int pos, int bsamples, float *timeb, float *Thetabx, float *Thetaby,
    float *Thetabz,
    float time_capture_start);  // initial declaration of blurr function

int main(int argc, char *argv[]) {
  int i;

  // time for capture end in seconds
  float time_capture_end;

  // time for capture start in seconds
  float time_capture_start;

  if (argc < 2) {
    cout << "\nUsage: " << argv[0] << " <jpeg file> \n";
    return 0;
  }

  // set number of samples for further treatment
  int N;

  long capture_timestamp;  // capture end timestamp

  // read the values without parser
  cin >> N;
  cin >> capture_timestamp;

  // timestamp, angular velocities
  float *wx = new float[N];       // angular velocity x [rad/s]
  float *wy = new float[N];       // angular velocity y [rad/s]
  float *wz = new float[N];       // angular velocity z [rad/s]
  long *timestamp = new long[N];  // timestamp [ns]
  int j = 0;                      // position for data transfer

  /*
   * This segment is for transformation of
   * gyro data to angular positions
   */

  // time, angular positions
  float *time = new float[N];    // time [s]
  float *Thetax = new float[N];  // angle x
  float *Thetay = new float[N];  // angle y
  float *Thetaz = new float[N];  // angle z

  for (int j = 0; j < N; j++) {
    // read the values without parser
    cin >> timestamp[j] >> wx[j] >> wy[j] >> wz[j];

    if (j > 0) {
      float Rt[9];      // rotation matrix
      float Rti[9];     // inverse of rotation matrix
      float ThetaR[3];  // angle of rotation
      float dTheta[3];  // angle of rotation change

      // calculate time interval
      time[j] = (timestamp[j] - timestamp[0]) * 1e-9;

      // calculate unit vector of rotation
      float urx = Thetax[j - 1];
      float ury = Thetay[j - 1];
      float urz = Thetaz[j - 1];
      float sur = sqrt(urx * urx + ury * ury + urz * urz);
      float c, s, cc;

      if (sur != 0) {
        urx /= sur;
        ury /= sur;
        urz /= sur;
        c = cos(sur), s = sin(sur), cc = 1 - c;
        // inverse of the rotation matrix
        Rti[0] = urx * urx * cc + c;
        Rti[1] = urx * ury * cc + urz * s;
        Rti[2] = urx * urz * cc - ury * s;
        Rti[3] = ury * urx * cc - urz * s;
        Rti[4] = ury * ury * cc + c;
        Rti[5] = ury * urz * cc + urx * s;
        Rti[6] = urz * urx * cc + ury * s;
        Rti[7] = urz * ury * cc - urx * s;
        Rti[8] = urz * urz * cc + c;
      } else {
        Rti[0] = 1.0;
        Rti[1] = 0.0;
        Rti[2] = 0.0;
        Rti[3] = 0.0;
        Rti[4] = 1.0;
        Rti[5] = 0.0;
        Rti[6] = 0.0;
        Rti[7] = 0.0;
        Rti[8] = 1.0;
      }

      dTheta[0] = Rti[0] * wx[j - 1] + Rti[1] * wy[j - 1] + Rti[2] * wz[j - 1];
      dTheta[1] = Rti[3] * wx[j - 1] + Rti[4] * wy[j - 1] + Rti[5] * wz[j - 1];
      dTheta[2] = Rti[6] * wx[j - 1] + Rti[7] * wy[j - 1] + Rti[8] * wz[j - 1];

      // new value of rotation
      Thetax[j] = Thetax[j - 1] + dTheta[0] * (time[j] - time[j - 1]);
      Thetay[j] = Thetay[j - 1] + dTheta[1] * (time[j] - time[j - 1]);
      Thetaz[j] = Thetaz[j - 1] + dTheta[2] * (time[j] - time[j - 1]);
    } else {
      Thetax[0] = 0.0;
      Thetay[0] = 0.0;
      Thetaz[0] = 0.0;
      time[0] = 0.0;
    }
  }

  time_capture_end = (capture_timestamp - timestamp[0]) / 1e9;
  time_capture_start = time_capture_end - time_exposure - time_readout;
  cout << "# Capture time start " << time_capture_start << "\n";
  cout << "# Capture time end " << time_capture_end << "\n\n";

  // selection of the points for kernel interpolation
  int j_i, j_f, flag1 = true,
                flag2 = true;  // initial point, final point, control flags

  for (int j = 0; j < N; j++) {
    if ((time[j] > time_capture_start + time_delay) && (flag1)) {
      j_i = j - 1;
      flag1 = false;
    }

    if ((time[j] > time_capture_start + time_exposure + time_delay) &&
        (flag2)) {
      j_f = j;
      flag2 = false;
    }
  }

  cout << "# Border time stamps of blurring: from  " << j_i << " to " << j_f
       << '\n';

  // extract the needed Theta and time
  int bsamples = j_f - j_i + 1;
  float *Thetabx = new float[bsamples];
  float *Thetaby = new float[bsamples];
  float *Thetabz = new float[bsamples];
  float *timeb = new float[bsamples];

  for (int j = j_i; j < j_f + 1; j++) {
    Thetabx[j - j_i] = Thetax[j];
    Thetaby[j - j_i] = Thetay[j];
    Thetabz[j - j_i] = Thetaz[j];
    timeb[j - j_i] = time[j];
  }

  // release the memory - samples are not needed anymore
  delete[] timestamp;
  delete[] time;
  delete[] wx;
  delete[] wy;
  delete[] wz;
  delete[] Thetax;
  delete[] Thetay;
  delete[] Thetaz;

  // creates initial image
  if (!(img = cv::imread(argv[1], cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR))
           .empty()) {
    imgi = cv::Mat(cv::Size(img.cols, img.rows), img.depth(), img.channels());
    kernel = img.clone();
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
    cout << '\n' << argv[1] << " couldn't be opened.\n";
    return 0;
  }

  int imw = imgi.cols, imh = imgi.rows;  // image width, image height
  float u0 = imw / 2.0, v0 = imh / 2.0;  // principal point in pixels
  int r0x = 0 / 2,
      r0y = 0 / 2;       // pixel where the movement points will be drawn
  float x1, y1, x0, y0;  // current coordinates

  // image window control
  int IMG_WIN;

  if ((imw > 1080) || (imh > 720))
    IMG_WIN = 0;
  else
    IMG_WIN = cv::WINDOW_AUTOSIZE;

  if (imgi.channels() != 1) {
    cv::namedWindow("InitialColor", IMG_WIN);
    cv::namedWindow("BlurredColor", IMG_WIN);
    if (IMG_WIN == 0) {
      cv::resizeWindow("InitialColor", 1080, 720);
      cv::resizeWindow("BlurredColor", 1080, 720);
    }
  } else {
    cv::namedWindow("InitialGray", IMG_WIN);
    cv::namedWindow("BlurredGray", IMG_WIN);
    if (IMG_WIN == 0) {
      cv::resizeWindow("InitialGray", 1080, 720);
      cv::resizeWindow("BlurredGray", 1080, 720);
    }
  }

  // loops to draw the kernels
  for (int r0x = 50; r0x < kernel.rows; r0x += 100)
    for (int r0y = 50; r0y < kernel.cols; r0y += 100) {
      // non homogenious coordinates
      float x0 = (r0x - u0) / (FOV), y0 = (r0y - v0) / (FOV);

      // (x,y) coordinates of points, which correspond to blurring
      // alongside with angles an time
      float *xb = new float[bsamples];
      float *yb = new float[bsamples];
      float xbi, ybi, xbf, ybf, xbmax = 0, ybmax = 0;
      int *bstep = new int[bsamples - 1];  // steps for interpolation
      int bsteps = 0;

      // set the array of (x,y) points for blurring
      // point movement for the whole time interval
      for (int j = 0; j < bsamples; j++) {
        float Rt[9];  // rotation matrix
        float c, s, cc;

        // calculation of the rotation matrix, based on actual angle,
        // to be applied to initial point
        float urx = Thetabx[j], ury = Thetaby[j], urz = Thetabz[j];
        float sur = sqrt(urx * urx + ury * ury + urz * urz);
        urx /= sur;
        ury /= sur;
        urz /= sur;

        c = cos(sur), s = sin(sur), cc = 1 - c;
        Rt[0] = urx * urx * cc + c;
        Rt[1] = urx * ury * cc - urz * s;
        Rt[2] = urx * urz * cc + ury * s;
        Rt[3] = ury * urx * cc + urz * s;
        Rt[4] = ury * ury * cc + c;
        Rt[5] = ury * urz * cc - urx * s;
        Rt[6] = urz * urx * cc - ury * s;
        Rt[7] = urz * ury * cc + urx * s;
        Rt[8] = urz * urz * cc + c;

        x1 = (Rt[0] * x0 + Rt[1] * y0 + Rt[2] * 1.0) /
             (Rt[6] * x0 + Rt[7] * y0 + Rt[8] * 1.0);
        y1 = (Rt[3] * x0 + Rt[4] * y0 + Rt[5] * 1.0) /
             (Rt[6] * x0 + Rt[7] * y0 + Rt[8] * 1.0);
        xb[j] = x1 * (FOV) + u0 - r0x;
        yb[j] = y1 * (FOV) + v0 - r0y;
      }

      // set the initial and final points
      xbi = (xb[1] - xb[0]) * (time_capture_start + time_delay - timeb[0]) /
                (timeb[1] - timeb[0]) +
            xb[0];
      ybi = (yb[1] - yb[0]) * (time_capture_start + time_delay - timeb[0]) /
                (timeb[1] - timeb[0]) +
            yb[0];

      xbf = (xb[bsamples - 1] - xb[bsamples - 2]) *
                (time_capture_start + time_exposure + time_delay -
                 timeb[bsamples - 2]) /
                (timeb[bsamples - 1] - timeb[bsamples - 2]) +
            xb[bsamples - 2];
      ybf = (yb[bsamples - 1] - yb[bsamples - 2]) *
                (time_capture_start + time_exposure + time_delay -
                 timeb[bsamples - 2]) /
                (timeb[bsamples - 1] - timeb[bsamples - 2]) +
            yb[bsamples - 2];

      // the final set of points
      xb[0] = 0;
      yb[0] = 0;
      xb[bsamples - 1] = xbf - xbi;
      yb[bsamples - 1] = ybf - ybi;
      for (int j = 1; j < bsamples - 1; j++) {
        xb[j] -= xbi;
        yb[j] -= ybi;
        if (abs(xb[j]) > abs(xbmax)) xbmax = xb[j];
        if (abs(yb[j]) > abs(ybmax)) ybmax = yb[j];
      }
      if (abs(xb[0]) > abs(xbmax)) xbmax = xb[0];
      if (abs(yb[0]) > abs(ybmax)) ybmax = yb[0];
      if (abs(xb[bsamples - 1]) > abs(xbmax)) xbmax = xb[bsamples - 1];
      if (abs(yb[bsamples - 1]) > abs(ybmax)) ybmax = yb[bsamples - 1];

      // calculate number of steps
      for (int j = 0; j < bsamples - 1; j++) {
        bstep[j] = int(2.0 * sqrt((xb[j + 1] - xb[j]) * (xb[j + 1] - xb[j]) +
                                  (yb[j + 1] - yb[j]) * (yb[j + 1] - yb[j])) +
                       0.5);
        bsteps += bstep[j];
      }

      // sparse kernel
      int dims[] = {2 * int(abs(ybmax) + 0.5) + 10,
                    2 * int(abs(xbmax) + 0.5) + 10};
      cv::SparseMat skernel = cv::SparseMat(2, dims, CV_32F);

      // set the kernel values
      for (int j = 0; j < bsamples - 1; j++) {
        float step = 1.0 / bstep[j];
        for (int jj = 0; jj < bstep[j]; jj++) {
          float xd =
              -((yb[j + 1] - yb[j]) * jj / bstep[j] + yb[j]) + dims[0] / 2;
          float yd =
              -((xb[j + 1] - xb[j]) * jj / bstep[j] + xb[j]) + dims[1] / 2;

          // sparse kernel
          int idx[] = {int(xd), int(yd)};
          skernel.ref<float>(idx) +=
              (int(xd) - xd + 1) * (int(yd) - yd + 1) * step;
          idx[0] = int(xd);
          idx[1] = int(yd + 1);
          skernel.ref<float>(idx) += (int(xd) - xd + 1) * (yd - int(yd)) * step;
          idx[0] = int(xd + 1);
          idx[1] = int(yd);
          skernel.ref<float>(idx) += (xd - int(xd)) * (int(yd) - yd + 1) * step;
          idx[0] = int(xd + 1);
          idx[1] = int(yd + 1);
          skernel.ref<float>(idx) += (xd - int(xd)) * (yd - int(yd)) * step;
        }
      }

      float kernelsum = 0;
      // sparse kernel sum
      cv::SparseMatIterator it;
      for (it = skernel.begin(); it != skernel.end(); ++it) {
        float val = it.value<float>();
        if (kernelsum < val) kernelsum = val;
        // kernelsum+=val;
      }

      // normalize sparse kernel
      for (it = skernel.begin(); it != skernel.end(); ++it) {
        it.value<float>() /= kernelsum;
      }
      // set the big kernel image from sparse matrix
      for (it = skernel.begin(); it != skernel.end(); ++it) {
        cv::SparseMat::Node *node = it.node();
        int *idx = node->idx;
        float val = it.value<float>();
        if ((int(r0x + idx[1] - dims[1] / 2) < kernel.rows) &&
            (int(r0x + idx[1] - dims[1] / 2) >= 0) &&
            (int(r0y + idx[0] - dims[0] / 2) < kernel.cols) &&
            (int(r0y + idx[0] - dims[0] / 2) >= 0))
          if (val > 0) {
            // red channel
            ((uchar *)(kernel.data +
                       int(r0x + idx[1] - dims[1] / 2) * (kernel.step)))
                [int(r0y + idx[0] - dims[0] / 2) * kernel.channels() + 2] =
                    val * 255;
            ((uchar *)(kernel.data +
                       int(r0x + idx[1] - dims[1] / 2) * (kernel.step)))
                [int(r0y + idx[0] - dims[0] / 2) * kernel.channels() + 1] =
                    0;  // green
            ((uchar *)(kernel.data +
                       int(r0x + idx[1] - dims[1] / 2) * (kernel.step)))
                [int(r0y + idx[0] - dims[0] / 2) * kernel.channels() + 0] =
                    0;  // blue
          }
      }

      delete[] xb;
      delete[] yb;
      delete[] bstep;
    }

  // this is the kernel image for blurring
  imgb = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);

  // the initial splitting of channels
  imgc1 = cv::Mat(cv::Size(imgi.cols, imgi.rows), CV_32FC1);
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

  cv::namedWindow("Kernel", IMG_WIN);
  if (IMG_WIN == 0) cv::resizeWindow("Kernel", 1080, 720);

  Process(0, bsamples, timeb, Thetabx, Thetaby, Thetabz, time_capture_start);

  while (1)  // wait till ESC is pressed
  {
    char c = cv::waitKey(0);
    if (c == 27) break;
  }

  cv::destroyWindow("Kernel");

  if (imgi.channels() != 1) {
    cv::destroyWindow("InitialColor");
    cv::destroyWindow("BlurredColor");
  } else {
    cv::destroyWindow("InitialGray");
    cv::destroyWindow("BlurredGray");
  }

  delete[] Thetabx;
  delete[] Thetaby;
  delete[] Thetabz;
  delete[] timeb;

  return (0);
}

void Process(int pos, int bsamples, float *timeb, float *Thetabx,
             float *Thetaby, float *Thetabz, float time_capture_start) {
  cv::imshow("Kernel", kernel);

  if (imgi.channels() != 1) {
    // merge and show
    cv::merge(std::vector{imgc1, imgc2, imgc3}, img3b);

    cv::imshow("InitialColor", imgi);

    // blurring - each channel separately
    BlurrPBCsv(imgc1, bsamples, timeb, Thetabx, Thetaby, Thetabz,
               time_capture_start, imgc1d);
    BlurrPBCsv(imgc2, bsamples, timeb, Thetabx, Thetaby, Thetabz,
               time_capture_start, imgc2d);
    BlurrPBCsv(imgc3, bsamples, timeb, Thetabx, Thetaby, Thetabz,
               time_capture_start, imgc3d);

    // merge and show
    cv::merge(std::vector{imgc1d, imgc2d, imgc3d}, img3d);

    cv::imshow("BlurredColor", img3d);
    img3d = img3d * 255;
    cv::imwrite("Blurred.tif", img3d);
  } else {
    cv::imshow("BlurredGray", imgc1);

    // deblur
    BlurrPBCsv(imgc1, bsamples, timeb, Thetabx, Thetaby, Thetabz,
               time_capture_start, imgc1d);

    cv::imshow("BlurredGray", imgc1d);

    imgc1d = imgc1d * 255;
    cv::imwrite("Blurredbw.tif", imgc1d);
  }
}
