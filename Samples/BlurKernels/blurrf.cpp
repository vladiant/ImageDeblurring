#include "blurrf.hpp"

#include <algorithm>
#include <opencv2/imgproc/imgproc.hpp>

const double pi = 3.14159265358979;

/*
 * This is the minimal value,
 * below which the function is
 * accepted equal to zero
 * used in kspan
 */

const float eps = 1e-4;

float psfMoffat(int x, int y, float r, float s1, float s2, float bet) {
  float s;
  s = pow(1 + (x * x * s2 * s2 - 2 * r * r * x * y + y * y * s1 * s1) /
                  (s1 * s1 * s2 * s2 - r * r * r * r),
          -bet);
  return (s);
}

float psfGauss(int x, int y, float r, float s1, float s2) {
  float s;
  s = exp(-0.5 * (x * x * s2 * s2 - 2 * r * r * x * y + y * y * s1 * s1) /
          (s1 * s1 * s2 * s2 - r * r * r * r));
  return (s);
}

float psfDefoc(int x, int y, float r) {
  float s;
  if ((x * x + y * y) < (r * r)) {
    if (((x + 1) * (x + 1) + (y + 1) * (y + 1)) > (r * r)) {
      s = (sqrt(r * r - x * x) - y) * (sqrt(r * r - y * y) - x) * 0.5;
    } else {
      s = 1.0;
    }
  } else {
    s = 0.0;
  }

  return (s);
}

int kspan(float (*ptLV1f)(int, int, float), float a1) {
  bool flag;
  int i = 0, j = 0;
  float s = 0;
  flag = true;
  do {
    j = 0;
    flag = false;
    do {
      if (((*ptLV1f)(i, j, a1)) > eps) flag = true;
      j++;
    } while (j < i);
    i++;
  } while (flag);
  return (std::max(i, j));
}

int kspan(float (*ptLV1f)(int, int, float, float, float), float a1, float a2,
          float a3) {
  bool flag;
  int i = 0, j = 0;
  float s = 0;
  flag = true;
  do {
    j = 0;
    flag = false;
    do {
      if (((*ptLV1f)(i, j, a1, a2, a3)) > eps) flag = true;
      j++;
    } while (j < i);
    i++;
  } while (flag);
  return (std::max(i, j));
}

int kspan(float (*ptLV1f)(int, int, float, float, float, float), float a1,
          float a2, float a3, float a4) {
  bool flag;
  int i = 0, j = 0;
  float s = 0;
  flag = true;
  do {
    j = 0;
    flag = false;
    do {
      if (((*ptLV1f)(i, j, a1, a2, a3, a4)) > eps) flag = true;
      j++;
    } while (j < i);
    i++;
  } while (flag);
  return (std::max(i, j));
}

void kset(cv::Mat& kernel, float (*ptLV1f)(int, int, float), float a1) {
  cv::Scalar ksum;

  kernel = 0;

  for (int row = 0; row < kernel.rows; row++) {
    for (int col = 1; col < kernel.cols; col++) {
      ((float*)(kernel.data + row * kernel.step))[col] +=
          (*ptLV1f)(row - (kernel.cols / 2), col - (kernel.rows / 2), a1);
    }
  }

  ksum = cv::sum(kernel);
  kernel = kernel / ksum.val[0];
}

void kset(cv::Mat& kernel, float (*ptLV1f)(int, int, float, float, float),
          float a1, float a2, float a3) {
  cv::Scalar ksum;

  kernel = 0;

  for (int row = 0; row < kernel.rows; row++) {
    for (int col = 1; col < kernel.cols; col++) {
      ((float*)(kernel.data + row * kernel.step))[col] += (*ptLV1f)(
          row - (kernel.cols / 2), col - (kernel.rows / 2), a1, a2, a3);
    }
  }

  ksum = cv::sum(kernel);
  kernel = kernel / ksum.val[0];
}

void kset(cv::Mat& kernel,
          float (*ptLV1f)(int, int, float, float, float, float), float a1,
          float a2, float a3, float a4) {
  cv::Scalar ksum;

  kernel = 0;

  for (int row = 0; row < kernel.rows; row++) {
    for (int col = 1; col < kernel.cols; col++) {
      ((float*)(kernel.data + row * kernel.step))[col] += (*ptLV1f)(
          row - (kernel.cols / 2), col - (kernel.rows / 2), a1, a2, a3, a4);
    }
  }

  ksum = cv::sum(kernel);
  kernel = kernel / ksum.val[0];
}

void ksetDefocus(cv::Mat& kernel, int r) {
  cv::Scalar ksum;

  kernel = 0;
  cv::circle(kernel, cv::Point(kernel.cols / 2, kernel.rows / 2), r,
             cv::Scalar(1.0), -1, cv::LINE_AA);
  ksum = cv::sum(kernel);
  kernel = kernel / ksum.val[0];
}

void ksetMoveXY(cv::Mat& kernel, int x, int y, int x0, int y0) {
  cv::Scalar ksum;

  kernel = 0;
  cv::line(kernel, cv::Point(x0 + kernel.cols / 2, y0 + kernel.rows / 2),
           cv::Point(x + kernel.cols / 2, y + kernel.rows / 2), cv::Scalar(1.0),
           1, cv::LINE_AA);
  ksum = cv::sum(kernel);
  kernel = kernel / ksum.val[0];
}

void ksetMoveXYxw(cv::Mat& kernel, float x, float y, float x0, float y0) {
  cv::Scalar ksum;
  float x1, x2, y1, y2, dx, dy, gradient, tmp, xc, yc;
  float xend, yend, xgap, ygap, xpxl1, ypxl1, xpxl2, ypxl2, interx, intery;

  kernel = 0;

  x1 = x0 + kernel.cols / 2;
  x2 = x + kernel.cols / 2;
  y1 = y0 + kernel.rows / 2;
  y2 = y + kernel.rows / 2;

  dx = x2 - x1;
  dy = y2 - y1;

  if (abs(dx) < abs(dy)) {
    if (y2 < y1) {
      tmp = x1;
      x1 = x2;
      x2 = tmp;

      tmp = y1;
      y1 = y2;
      y2 = tmp;
    }

    gradient = dx / dy;

    // handle first endpoint
    yend = int(y1 + 0.5);
    xend = x1 + gradient * (yend - y1);
    ygap = 1 - (y1 + 0.5 - int(y1 + 0.5));
    ypxl1 = yend;  // this will be used in the main loop
    xpxl1 = int(xend);

    ((float*)(kernel.data + int(ypxl1) * kernel.step))[int(xpxl1)] =
        (1 - (xend - int(xend))) * ygap;
    ((float*)(kernel.data + int(ypxl1 + 1) * kernel.step))[int(xpxl1)] =
        (xend - int(xend)) * ygap;
    interx = xend + gradient;  // first y-intersection for the main loop

    // handle second endpoint
    yend = int(y2 + 0.5);
    xend = x2 + gradient * (yend - y2);
    ygap = (y2 + 0.5) - int(y2 + 0.5);
    ypxl2 = yend;  // this will be used in the main loop
    xpxl2 = int(xend);
    ((float*)(kernel.data + int(ypxl2) * kernel.step))[int(xpxl2)] =
        (1 - (xend - int(xend))) * ygap;
    ((float*)(kernel.data + int(ypxl2 + 1) * kernel.step))[int(xpxl2)] =
        (xend - int(xend)) * ygap;

    // main loop
    for (yc = ypxl1 + 1; yc < ypxl2; yc++) {
      ((float*)(kernel.data + int(int(yc)) * (kernel.step)))[int(interx)] =
          (1 - (interx - int(interx))) * ygap;
      ((float*)(kernel.data + int(int(yc)) * (kernel.step)))[int(interx) + 1] =
          (interx - int(interx)) * ygap;
      interx = interx + gradient;
    }
  } else {
    if (x2 < x1) {
      tmp = x1;
      x1 = x2;
      x2 = tmp;

      tmp = y1;
      y1 = y2;
      y2 = tmp;
    }

    gradient = dy / dx;

    // handle first endpoint
    xend = int(x1 + 0.5);
    yend = y1 + gradient * (xend - x1);
    xgap = 1 - (x1 + 0.5 - int(x1 + 0.5));
    xpxl1 = xend;  // this will be used in the main loop
    ypxl1 = int(yend);

    ((float*)(kernel.data + int(ypxl1) * kernel.step))[int(xpxl1)] =
        (1 - (yend - int(yend))) * xgap;
    ((float*)(kernel.data + int(ypxl1 + 1) * kernel.step))[int(xpxl1)] =
        (yend - int(yend)) * xgap;
    intery = yend + gradient;  // first y-intersection for the main loop

    // handle second endpoint
    xend = int(x2 + 0.5);
    yend = y2 + gradient * (xend - x2);
    xgap = (x2 + 0.5) - int(x2 + 0.5);
    xpxl2 = xend;  // this will be used in the main loop
    ypxl2 = int(yend);
    ((float*)(kernel.data + int(ypxl2) * kernel.step))[int(xpxl2)] =
        (1 - (yend - int(yend))) * xgap;
    ((float*)(kernel.data + int(ypxl2 + 1) * kernel.step))[int(xpxl2)] =
        (yend - int(yend)) * xgap;

    // main loop
    for (xc = xpxl1 + 1; xc < xpxl2; xc++) {
      ((float*)(kernel.data + int(int(intery)) * (kernel.step)))[int(xc)] =
          (1 - (intery - int(intery))) * xgap;
      ((float*)(kernel.data + int(int(intery) + 1) * (kernel.step)))[int(xc)] =
          (intery - int(intery)) * xgap;
      intery = intery + gradient;
    }
  }

  ksum = cv::sum(kernel);
  kernel = kernel / ksum.val[0];
}

void ksetMoveXYb(cv::Mat& kernel, float x, float y, float x0, float y0) {
  cv::Scalar ksum;
  float x1, x2, y1, y2;
  float xt, yt, dx, dy;
  float a, b, i1, i2, i3, i4;
  float xl1, yl1, xl2, yl2;
  int mstep, step, pos, stepx, posx;

  kernel = 0;

  x1 = x0 + (kernel.cols - 1) / 2.0;
  x2 = x + (kernel.cols - 1) / 2.0;
  y1 = y0 + (kernel.rows - 1) / 2.0;
  y2 = y + (kernel.rows - 1) / 2.0;

  // handle first endpoint
  if (x1 != 0) {
    i1 = 1.0 * trunc(x1) / x1;
    i2 = 1 - i1;
  } else {
    i1 = 1;
    i2 = 0;
  }

  if (y1 != 0) {
    i3 = 1.0 * trunc(y1) / y1;
    i4 = 1 - i3;
  } else {
    i3 = 1;
    i4 = 0;
  }

  ((float*)(kernel.data + int(y1) * (kernel.step)))[int(x1)] = i1 * i3;
  if (x1 < kernel.cols - 1)
    ((float*)(kernel.data + int(y1) * (kernel.step)))[int(x1 + 1)] = i2 * i3;
  if (y1 < kernel.rows - 1)
    ((float*)(kernel.data + int(y1 + 1) * (kernel.step)))[int(x1)] = i1 * i4;
  if ((x1 < kernel.cols - 1) && (y1 < kernel.rows - 1))
    ((float*)(kernel.data + int(y1 + 1) * (kernel.step)))[int(x1 + 1)] =
        i2 * i4;

  // handle second endpoint
  if (x2 != 0) {
    i1 = 1.0 + trunc(x2) - x2;
    i2 = x2 - trunc(x2);
  } else {
    i1 = 1;
    i2 = 0;
  }

  if (y2 != 0) {
    i3 = 1.0 + trunc(y2) - y2;
    i4 = y2 - trunc(y2);
  } else {
    i3 = 1;
    i4 = 0;
  }

  ((float*)(kernel.data + int(y2) * (kernel.step)))[int(x2)] = i1 * i3;
  if (x2 < kernel.cols - 1)
    ((float*)(kernel.data + int(y2) * (kernel.step)))[int(x2 + 1)] = i2 * i3;
  if (y2 < kernel.rows - 1)
    ((float*)(kernel.data + int(y2 + 1) * (kernel.step)))[int(x2)] = i1 * i4;
  if ((x2 < kernel.cols - 1) && (y2 < kernel.rows - 1))
    ((float*)(kernel.data + int(y2 + 1) * (kernel.step)))[int(x2 + 1)] =
        i2 * i4;

  dy = y2 - y1;
  dx = x2 - x1;
  yt = y1;

  // main loop
  if (abs(dx) > abs(dy)) {
    a = -dy / (-x1 * dy + y1 * dx);
    b = dx / (-x1 * dy + y1 * dx);
    mstep = int(x2) - int(x1);
    step = dx / abs(dx);

    yl1 = (1 - a * (int(x1) + 0.5 * step)) / b;
    if ((int(x1) < kernel.cols) && (int(x1) >= 0)) {
      if (int(yl1) != int(y1))
        ((float*)(kernel.data + int(yl1) * (kernel.step)))[int(x1)] +=
            (1.0 + trunc(yl1) - yl1) * 0.5;
      else
        ((float*)(kernel.data + int(yl1 + 1) * (kernel.step)))[int(x1)] +=
            (yl1 - trunc(yl1)) * 0.5;
    }

    for (pos = step; abs(pos) < abs(mstep); pos += step) {
      xl1 = trunc(x1 + pos) - 0.5;
      xl2 = trunc(x1 + pos) + 0.5;
      yl1 = (1 - a * xl1) / b;
      yl2 = (1 - a * xl2) / b;
      i1 = 1.0 + trunc(yl1) - yl1;
      i2 = yl1 - trunc(yl1);
      i3 = 1.0 + trunc(yl2) - yl2;
      i4 = yl2 - trunc(yl2);
      ((float*)(kernel.data + int(yl1) * (kernel.step)))[int(x1 + pos)] +=
          i1 * 0.5;
      ((float*)(kernel.data + int(yl1 + 1) * (kernel.step)))[int(x1 + pos)] +=
          i2 * 0.5;
      ((float*)(kernel.data + int(yl2) * (kernel.step)))[int(x1 + pos)] +=
          i3 * 0.5;
      ((float*)(kernel.data + int(yl2 + 1) * (kernel.step)))[int(x1 + pos)] +=
          i4 * 0.5;
    }

    yl1 = (1 - a * (int(x2) - 0.5 * step)) / b;
    if ((int(x1 + pos) < kernel.cols) && (int(x1 + pos) >= 0)) {
      if (int(yl1) != int(y2))
        ((float*)(kernel.data + int(yl1) * (kernel.step)))[int(x1 + pos)] +=
            (1.0 + trunc(yl1) - yl1) * 0.5;
      else
        ((float*)(kernel.data + int(yl1 + 1) * (kernel.step)))[int(x1 + pos)] +=
            (yl1 - trunc(yl1)) * 0.5;
    }

    // correction of the endpoints
    if (x1 != int(x1)) {
      yl1 = (1 - a * (int(x1) + 0.5 * step)) / b;
      ((float*)(kernel.data +
                int(yl1 + 1) * (kernel.step)))[int(x1) - (1 - step) / 2] +=
          (-trunc(yl1) + yl1);
      ((float*)(kernel.data +
                int(yl1) * (kernel.step)))[int(x1) - (1 - step) / 2] +=
          (1 + trunc(yl1) - yl1);
    }

    if (x2 != int(x2)) {
      yl1 = (1 - a * (int(x2) - 0.5 * step)) / b;
      ((float*)(kernel.data +
                int(yl1 + 1) * (kernel.step)))[int(x2) + (1 - step) / 2] +=
          (-trunc(yl1) + yl1);
      ((float*)(kernel.data +
                int(yl1) * (kernel.step)))[int(x2) + (1 - step) / 2] +=
          (1 + trunc(yl1) - yl1);
    }

  } else {
    if (abs(dx) == abs(dy)) {
      mstep = int(dy);
      step = dy / abs(dy);
      stepx = dx / abs(dx);

      ((float*)(kernel.data + int(y1) * (kernel.step)))[int(x1)] += 1.0;
      ((float*)(kernel.data + int(y1) * (kernel.step)))[int(x1 + stepx)] += 0.5;

      for (pos = step, posx = stepx; abs(pos) < abs(mstep);
           pos += step, posx += stepx) {
        ((float*)(kernel.data +
                  int(y1 + pos) * (kernel.step)))[int(x1 + posx)] += 1.0;
        ((float*)(kernel.data +
                  int(y1 + pos) * (kernel.step)))[int(x1 + posx + 1)] += 0.5;
        ((float*)(kernel.data +
                  int(y1 + pos) * (kernel.step)))[int(x1 + posx - 1)] += 0.5;
      }
      ((float*)(kernel.data +
                int(y1 + pos) * (kernel.step)))[int(x1 + posx - stepx)] += 0.5;
      ((float*)(kernel.data + int(y2) * (kernel.step)))[int(x2)] += 1.0;
    } else {
      a = dy / (x1 * dy - y1 * dx);
      b = -dx / (x1 * dy - y1 * dx);
      mstep = int(y2) - int(y1);
      step = dy / abs(dy);

      xl1 = (1 - b * (int(y1) + 0.5 * step)) / a;
      if ((int(y1) < kernel.rows) && (int(y1) >= 0)) {
        if (int(xl1) != int(x1))
          ((float*)(kernel.data + int(y1) * (kernel.step)))[int(xl1)] +=
              (1.0 + trunc(xl1) - xl1) * 0.5;
        else
          ((float*)(kernel.data + int(y1) * (kernel.step)))[int(xl1 + 1)] +=
              (xl1 - trunc(xl1)) * 0.5;
      }

      for (pos = step; abs(pos) < abs(mstep); pos += step) {
        yl1 = trunc(y1 + pos) - 0.5;
        yl2 = trunc(y1 + pos) + 0.5;
        xl1 = (1 - b * yl1) / a;
        xl2 = (1 - b * yl2) / a;
        i1 = 1.0 + trunc(xl1) - xl1;
        i2 = xl1 - trunc(xl1);
        i3 = 1.0 + trunc(xl2) - xl2;
        i4 = xl2 - trunc(xl2);
        ((float*)(kernel.data + int(y1 + pos) * (kernel.step)))[int(xl1)] +=
            i1 * 0.5;
        ((float*)(kernel.data + int(y1 + pos) * (kernel.step)))[int(xl1 + 1)] +=
            i2 * 0.5;
        ((float*)(kernel.data + int(y1 + pos) * (kernel.step)))[int(xl2)] +=
            i3 * 0.5;
        ((float*)(kernel.data + int(y1 + pos) * (kernel.step)))[int(xl2 + 1)] +=
            i4 * 0.5;
      }

      xl1 = (1 - b * (int(y2) - 0.5 * step)) / a;

      if ((int(y1 + pos) < kernel.rows) && (int(y1 + pos) >= 0)) {
        if (int(xl1) != int(x2))
          ((float*)(kernel.data + int(y1 + pos) * (kernel.step)))[int(xl1)] +=
              (1.0 + trunc(xl1) - xl1) * 0.5;
        else
          ((float*)(kernel.data +
                    int(y1 + pos) * (kernel.step)))[int(xl1 + 1)] +=
              (xl1 - trunc(xl1)) * 0.5;
      }

      // correction of the endpoints
      if (y1 != int(y1)) {
        xl1 = (1 - b * (int(y1) + 0.5 * step)) / a;
        ((float*)(kernel.data +
                  (int(y1) - (1 - step) / 2) * (kernel.step)))[int(xl1 + 1)] +=
            (-trunc(xl1) + xl1);
        ((float*)(kernel.data +
                  (int(y1) - (1 - step) / 2) * (kernel.step)))[int(xl1)] +=
            (1 + trunc(xl1) - xl1);
      }

      if (y2 != int(y2)) {
        xl1 = (1 - b * (int(y2) - 0.5 * step)) / a;
        ((float*)(kernel.data +
                  (int(y2) + (1 - step) / 2) * (kernel.step)))[int(xl1 + 1)] +=
            (-trunc(xl1) + xl1);
        ((float*)(kernel.data +
                  (int(y2) + (1 - step) / 2) * (kernel.step)))[int(xl1)] +=
            (1 + trunc(xl1) - xl1);
      }
    }
  }

  ksum = cv::sum(kernel);
  kernel = kernel / ksum.val[0];
}

void ksetMoveXYbcont(cv::Mat& kernel, float x, float y, float x0, float y0) {
  cv::Scalar ksum;
  float x1, x2, y1, y2;
  float xt, yt, dx, dy;
  float a, b, i1, i2, i3, i4;
  float xl1, yl1, xl2, yl2;
  int mstep, step, pos, stepx, posx;

  kernel = 0;

  x1 = x0 + (kernel.cols - 1) / 2.0;
  x2 = x + (kernel.cols - 1) / 2.0;
  y1 = y0 + (kernel.rows - 1) / 2.0;
  y2 = y + (kernel.rows - 1) / 2.0;

  /*
    // handle first endpoint
    if(x1!=0)
    {
            i1=1.0*trunc(x1)/x1;
            i2=1-i1;
    }
    else
    {
            i1=1; i2=0;
    }

    if(y1!=0)
    {
            i3=1.0*trunc(y1)/y1;
            i4=1-i3;
    }
    else
    {
            i3=1; i4=0;
    }

    ((float*)(kernel.data + int(y1)*(kernel.step)))[int(x1)]=i1*i3;
    if (x1<kernel.cols-1) ((float*)(kernel.data +
    int(y1)*(kernel.step)))[int(x1+1)]=i2*i3; if (y1<kernel.rows-1)
    ((float*)(kernel.data +
    int(y1+1)*(kernel.step)))[int(x1)]=i1*i4; if
    ((x1<kernel.cols-1)&&(y1<kernel.rows-1)) ((float*)(kernel.data +
    int(y1+1)*(kernel.step)))[int(x1+1)]=i2*i4;
  */

  // handle second endpoint
  if (x2 != 0) {
    i1 = 1.0 + trunc(x2) - x2;
    i2 = x2 - trunc(x2);
  } else {
    i1 = 1;
    i2 = 0;
  }

  if (y2 != 0) {
    i3 = 1.0 + trunc(y2) - y2;
    i4 = y2 - trunc(y2);
  } else {
    i3 = 1;
    i4 = 0;
  }

  ((float*)(kernel.data + int(y2) * (kernel.step)))[int(x2)] = i1 * i3;
  if (x2 < kernel.cols - 1)
    ((float*)(kernel.data + int(y2) * (kernel.step)))[int(x2 + 1)] = i2 * i3;
  if (y2 < kernel.rows - 1)
    ((float*)(kernel.data + int(y2 + 1) * (kernel.step)))[int(x2)] = i1 * i4;
  if ((x2 < kernel.cols - 1) && (y2 < kernel.rows - 1))
    ((float*)(kernel.data + int(y2 + 1) * (kernel.step)))[int(x2 + 1)] =
        i2 * i4;

  dy = y2 - y1;
  dx = x2 - x1;
  yt = y1;

  // main loop
  if (abs(dx) > abs(dy)) {
    a = -dy / (-x1 * dy + y1 * dx);
    b = dx / (-x1 * dy + y1 * dx);
    mstep = int(x2) - int(x1);
    step = dx / abs(dx);

    yl1 = (1 - a * (int(x1) + 0.5 * step)) / b;
    if ((int(x1) < kernel.cols) && (int(x1) >= 0)) {
      if (int(yl1) != int(y1))
        ((float*)(kernel.data + int(yl1) * (kernel.step)))[int(x1)] +=
            (1.0 + trunc(yl1) - yl1) * 0.5;
      else
        ((float*)(kernel.data + int(yl1 + 1) * (kernel.step)))[int(x1)] +=
            (yl1 - trunc(yl1)) * 0.5;
    }

    for (pos = step; abs(pos) < abs(mstep); pos += step) {
      xl1 = trunc(x1 + pos) - 0.5;
      xl2 = trunc(x1 + pos) + 0.5;
      yl1 = (1 - a * xl1) / b;
      yl2 = (1 - a * xl2) / b;
      i1 = 1.0 + trunc(yl1) - yl1;
      i2 = yl1 - trunc(yl1);
      i3 = 1.0 + trunc(yl2) - yl2;
      i4 = yl2 - trunc(yl2);
      ((float*)(kernel.data + int(yl1) * (kernel.step)))[int(x1 + pos)] +=
          i1 * 0.5;
      ((float*)(kernel.data + int(yl1 + 1) * (kernel.step)))[int(x1 + pos)] +=
          i2 * 0.5;
      ((float*)(kernel.data + int(yl2) * (kernel.step)))[int(x1 + pos)] +=
          i3 * 0.5;
      ((float*)(kernel.data + int(yl2 + 1) * (kernel.step)))[int(x1 + pos)] +=
          i4 * 0.5;
    }

    yl1 = (1 - a * (int(x2) - 0.5 * step)) / b;
    if ((int(x1 + pos) < kernel.cols) && (int(x1 + pos) >= 0)) {
      if (int(yl1) != int(y2))
        ((float*)(kernel.data + int(yl1) * (kernel.step)))[int(x1 + pos)] +=
            (1.0 + trunc(yl1) - yl1) * 0.5;
      else
        ((float*)(kernel.data + int(yl1 + 1) * (kernel.step)))[int(x1 + pos)] +=
            (yl1 - trunc(yl1)) * 0.5;
    }

    // correction of the endpoints
    if (x1 != int(x1)) {
      yl1 = (1 - a * (int(x1) + 0.5 * step)) / b;
      ((float*)(kernel.data +
                int(yl1 + 1) * (kernel.step)))[int(x1) - (1 - step) / 2] +=
          (-trunc(yl1) + yl1);
      ((float*)(kernel.data +
                int(yl1) * (kernel.step)))[int(x1) - (1 - step) / 2] +=
          (1 + trunc(yl1) - yl1);
    }

    if (x2 != int(x2)) {
      yl1 = (1 - a * (int(x2) - 0.5 * step)) / b;
      ((float*)(kernel.data +
                int(yl1 + 1) * (kernel.step)))[int(x2) + (1 - step) / 2] +=
          (-trunc(yl1) + yl1);
      ((float*)(kernel.data +
                int(yl1) * (kernel.step)))[int(x2) + (1 - step) / 2] +=
          (1 + trunc(yl1) - yl1);
    }

  } else {
    if (abs(dx) == abs(dy)) {
      mstep = int(dy);
      step = dy / abs(dy);
      stepx = dx / abs(dx);

      ((float*)(kernel.data + int(y1) * (kernel.step)))[int(x1)] += 1.0;
      ((float*)(kernel.data + int(y1) * (kernel.step)))[int(x1 + stepx)] += 0.5;

      for (pos = step, posx = stepx; abs(pos) < abs(mstep);
           pos += step, posx += stepx) {
        ((float*)(kernel.data +
                  int(y1 + pos) * (kernel.step)))[int(x1 + posx)] += 1.0;
        ((float*)(kernel.data +
                  int(y1 + pos) * (kernel.step)))[int(x1 + posx + 1)] += 0.5;
        ((float*)(kernel.data +
                  int(y1 + pos) * (kernel.step)))[int(x1 + posx - 1)] += 0.5;
      }
      ((float*)(kernel.data +
                int(y1 + pos) * (kernel.step)))[int(x1 + posx - stepx)] += 0.5;
      ((float*)(kernel.data + int(y2) * (kernel.step)))[int(x2)] += 1.0;
    } else {
      a = dy / (x1 * dy - y1 * dx);
      b = -dx / (x1 * dy - y1 * dx);
      mstep = int(y2) - int(y1);
      step = dy / abs(dy);

      xl1 = (1 - b * (int(y1) + 0.5 * step)) / a;
      if ((int(y1) < kernel.rows) && (int(y1) >= 0)) {
        if (int(xl1) != int(x1))
          ((float*)(kernel.data + int(y1) * (kernel.step)))[int(xl1)] +=
              (1.0 + trunc(xl1) - xl1) * 0.5;
        else
          ((float*)(kernel.data + int(y1) * (kernel.step)))[int(xl1 + 1)] +=
              (xl1 - trunc(xl1)) * 0.5;
      }

      for (pos = step; abs(pos) < abs(mstep); pos += step) {
        yl1 = trunc(y1 + pos) - 0.5;
        yl2 = trunc(y1 + pos) + 0.5;
        xl1 = (1 - b * yl1) / a;
        xl2 = (1 - b * yl2) / a;
        i1 = 1.0 + trunc(xl1) - xl1;
        i2 = xl1 - trunc(xl1);
        i3 = 1.0 + trunc(xl2) - xl2;
        i4 = xl2 - trunc(xl2);
        ((float*)(kernel.data + int(y1 + pos) * (kernel.step)))[int(xl1)] +=
            i1 * 0.5;
        ((float*)(kernel.data + int(y1 + pos) * (kernel.step)))[int(xl1 + 1)] +=
            i2 * 0.5;
        ((float*)(kernel.data + int(y1 + pos) * (kernel.step)))[int(xl2)] +=
            i3 * 0.5;
        ((float*)(kernel.data + int(y1 + pos) * (kernel.step)))[int(xl2 + 1)] +=
            i4 * 0.5;
      }

      xl1 = (1 - b * (int(y2) - 0.5 * step)) / a;

      if ((int(y1 + pos) < kernel.rows) && (int(y1 + pos) >= 0)) {
        if (int(xl1) != int(x2))
          ((float*)(kernel.data + int(y1 + pos) * (kernel.step)))[int(xl1)] +=
              (1.0 + trunc(xl1) - xl1) * 0.5;
        else
          ((float*)(kernel.data +
                    int(y1 + pos) * (kernel.step)))[int(xl1 + 1)] +=
              (xl1 - trunc(xl1)) * 0.5;
      }

      // correction of the endpoints
      if (y1 != int(y1)) {
        xl1 = (1 - b * (int(y1) + 0.5 * step)) / a;
        ((float*)(kernel.data +
                  (int(y1) - (1 - step) / 2) * (kernel.step)))[int(xl1 + 1)] +=
            (-trunc(xl1) + xl1);
        ((float*)(kernel.data +
                  (int(y1) - (1 - step) / 2) * (kernel.step)))[int(xl1)] +=
            (1 + trunc(xl1) - xl1);
      }

      if (y2 != int(y2)) {
        xl1 = (1 - b * (int(y2) - 0.5 * step)) / a;
        ((float*)(kernel.data +
                  (int(y2) + (1 - step) / 2) * (kernel.step)))[int(xl1 + 1)] +=
            (-trunc(xl1) + xl1);
        ((float*)(kernel.data +
                  (int(y2) + (1 - step) / 2) * (kernel.step)))[int(xl1)] +=
            (1 + trunc(xl1) - xl1);
      }
    }
  }

  ksum = cv::sum(kernel);
  kernel = kernel / ksum.val[0];
}
