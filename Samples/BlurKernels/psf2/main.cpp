//  PSF2 - Generates point spread functions for use with deconvolution fns.
//
//  This function can generate a variety function shapes based around the
//  Butterworth filter.  In plan view the filter can be elliptical and at
//  any orientation.  The 'squareness/roundness' of the shape can also be
//  manipulated.
//
//  Usage:  h = psf2(sze, order, ang, lngth, width, sqrness)
//
//   sze   - two element array specifying size of filter [rows cols]
//   order - an even integer specifying the order of the Butterworth filter.
//           This controls the sharpness of the cutoff.
//   ang   - angle of rotation of the filter in radians.
//   lngth - length of the filter in pixels along its major axis.
//   width - width of the filter in pixels along its minor axis.
//   sqrness - even integer specifying 'squareness' of the filter shape
//             a value of 2 gives a circular filter (if lngth == width), higher
//             values make the shape squarer.
//
// This function is almost identical to psf, it just has a different way of
// specifying the function shape whereby length and width are defined
// explicitly (rather than an average radius), this may be more convenient for
// some applications.
//
// Copyright (c) 1999-2003 Peter Kovesi
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
// May  2003 - Changed arguments so that psf is specified in terms of a length
//             and width rather than an average radius.

#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

void psf2(int order, float ang, float lngth, float width, int sqrness,
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

  // Create meshgrid for x and y coordinates
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      x[i][j] = j - cols / 2.0;
      y[i][j] = i - rows / 2.0;
    }
  }

  // Apply rotation to x and y coordinates
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      xp[i][j] = x[i][j] * cos(ang) - y[i][j] * sin(ang);
      yp[i][j] = x[i][j] * sin(ang) + y[i][j] * cos(ang);
    }
  }

  // Adjust y-coordinate to have correct aspect ratio
  float rc = lngth / 2.0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      yp[i][j] *= lngth / width;
    }
  }

  // Calculate the radius based on the squareness parameter
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      radius[i][j] = pow(
          pow(std::abs(xp[i][j]), sqrness) + pow(std::abs(yp[i][j]), sqrness),
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
  // Example usage of the psf2 function
  int order = 4;           // Order of the Butterworth filter
  float ang = M_PI / 4.0;  // Angle of rotation in radians (45 degrees)
  float lngth = 50.0;      // Length of the filter (major axis)
  float width = 30.0;      // Width of the filter (minor axis)
  int sqrness = 2;         // Squareness of the filter

  cv::namedWindow("PSF", 0);

  // Call the psf2 function
  psf2(order, ang, lngth, width, sqrness, image);

  cv::imshow("PSF", image * 500);
  cv::waitKey(0);

  return 0;
}
