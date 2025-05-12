//  PSF - Generates point spread functions for use with deconvolution fns.
//
//  This function can generate a variety function shapes based around the
//  Butterworth filter.  In plan view the filter can be elliptical and at
//  any orientation.  The `squareness/roundness' of the shape can also be
//  manipulated.
//
//  Usage:  h = psf(sze, order, ang, eccen, rc, sqrness)
//
//   sze   - two element array specifying size of filter [rows cols]
//   order - an even integer specifying the order of the Butterworth filter.
//           This controls the sharpness of the cutoff.
//   ang   - angle of rotation of the filter in radians
//   eccen - ratio of eccentricity of the filter shape (major/minor axis ratio)
//   rc    - mean radius of the filter in pixels
//   sqrness - even integer specifying `squareness' of the filter shape
//             a value of 2 gives a circular filter (if eccen = 1), higher
//             values make the shape squarer.
//
// Copyright (c) 1999 Peter Kovesi
// School of Computer Science & Software Engineering
// The University of Western Australia
// http://www.csse.uwa.edu.au/
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// The Software is provided "as is", without warranty of any kind.
//
// June 1999

#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

void psf(int order, double ang, double eccen, double rc, int sqrness,
         cv::Mat &h) {
  if (sqrness % 2 != 0) {
    printf("Error: squareness parameter must be an even integer\n");
    return;
  }

  int rows = h.rows;
  int cols = h.cols;

  // Allocate memory for x, y, xp, yp, and radius
  std::vector<std::vector<float>> x(rows);
  std::vector<std::vector<float>> y(rows);
  std::vector<std::vector<float>> xp(rows);
  std::vector<std::vector<float>> yp(rows);
  std::vector<std::vector<float>> radius(rows);

  for (int i = 0; i < rows; i++) {
    x[i].resize(cols);
    y[i].resize(cols);
    xp[i].resize(cols);
    yp[i].resize(cols);
    radius[i].resize(cols);
  }

  // Fill in the x and y arrays
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      x[i][j] = j - (cols / 2.0);
      y[i][j] = i - (rows / 2.0);
    }
  }

  // Apply rotation
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      xp[i][j] = x[i][j] * cos(ang) - y[i][j] * sin(ang);
      yp[i][j] = x[i][j] * sin(ang) + y[i][j] * cos(ang);
    }
  }

  // Apply eccentricity scaling
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      x[i][j] = sqrt(eccen) * xp[i][j];
      y[i][j] = yp[i][j] / sqrt(eccen);
    }
  }

  // Apply squareness to calculate radius
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      radius[i][j] =
          pow(pow(std::abs(x[i][j]), sqrness) + pow(std::abs(y[i][j]), sqrness),
              1.0 / sqrness);
    }
  }

  // Butterworth filter and normalization
  float sum_h = 0.0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      float value = 1.0 / (1.0 + pow(radius[i][j] / rc, order));
      ((float *)(h.data + i * h.step))[j] = value;
      sum_h += value;
    }
  }

  // Normalize the filter
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      ((float *)(h.data + i * h.step))[j] /= sum_h;
    }
  }
}

int main() {
  cv::Mat image(cv::Size(256, 256), CV_32FC1);
  image = 0;
  // Example usage of the psf function
  int order = 4;           // Order of the Butterworth filter
  float ang = M_PI / 4.0;  // Angle of rotation in radians (45 degrees)
  float eccen = 1.5;       // Eccentricity (major/minor axis ratio)
  float rc = 50.0;         // Mean radius in pixels
  int sqrness = 2;         // Squareness of the filter

  cv::namedWindow("PSF", 0);

  // Call the psf function
  psf(order, ang, eccen, rc, sqrness, image);

  cv::imshow("PSF", image * 5000);
  cv::waitKey(0);

  return 0;
}
