/*
 ================================================================================
 * (c) Copyright 2007-2012, . All Rights Reserved.
 *
 * Use of this software is controlled by the terms and conditions found
 * in the license agreement under which this software has been supplied.
 *
 ================================================================================
 */
/**
 *
 * @file SparseBlurkernel_test.cpp
 *
 * ^path /sparseBlurKernel/src/SparseBlurkernel_test.cpp
 *
 * @author Vladislav Antonov
 *
 * @date April 12, 2012
 *
 * @version 1.00
 *
 */
/* -----------------------------------------------------------------------------
 *!
 *! Revision History
 *! ===================================
 *! April 12, 2012 : Vladislav Antonov
 *! Created.
 *!
 * ===========================================================================
 */

#include "SparseBlurKernel.h"

#include <stdlib.h>

#include <limits>

namespace Test {
namespace Deblurring {

SparseBlurKernel::SparseBlurKernel() {
  // TODO Auto-generated constructor stub
}

SparseBlurKernel::~SparseBlurKernel() {
  // TODO Auto-generated destructor stub
}

void SparseBlurKernel::clear() { dataContainer.clear(); }

void SparseBlurKernel::setPointValue(point_coord_t x, point_coord_t y,
                                     point_value_t value) {
  KernelPointCoords currentPoint(x, y);

  dataContainer[currentPoint.key] = value;
}

void SparseBlurKernel::addToPointValue(point_coord_t x, point_coord_t y,
                                       point_value_t valueToAdd) {
  point_value_t pointValue = getPointValue(x, y);

  pointValue += valueToAdd;

  setPointValue(x, y, pointValue);
}

void SparseBlurKernel::clearPointValue(point_coord_t x, point_coord_t y) {
  KernelPointCoords currentPoint(x, y);

  dataContainerIterator = dataContainer.find(currentPoint.key);

  if (dataContainerIterator != dataContainer.end()) {
    dataContainer.erase(currentPoint.key);
  }
}

point_value_t SparseBlurKernel::getPointValue(point_coord_t x,
                                              point_coord_t y) {
  point_value_t pointValue = 0;

  KernelPointCoords currentPoint(x, y);

  dataContainerIterator = dataContainer.find(currentPoint.key);

  if (dataContainerIterator != dataContainer.end()) {
    pointValue = dataContainer[currentPoint.key];
  }

  return pointValue;
}

int SparseBlurKernel::getKernelSize() const { return dataContainer.size(); }

void SparseBlurKernel::extractKernelPoints(point_coord_t* x, point_coord_t* y,
                                           point_value_t* value) {
  int i = 0;
  for (dataContainerIterator = dataContainer.begin();
       dataContainerIterator != dataContainer.end();
       ++dataContainerIterator, ++i) {
    x[i] = ((KernelPointCoords)dataContainerIterator->first).x;
    y[i] = ((KernelPointCoords)dataContainerIterator->first).y;
    value[i] = dataContainerIterator->second;
  }
}

void SparseBlurKernel::extractKernelPoints(std::vector<point_coord_t>& x,
                                           std::vector<point_coord_t>& y,
                                           std::vector<point_value_t>& value) {
  for (dataContainerIterator = dataContainer.begin();
       dataContainerIterator != dataContainer.end(); ++dataContainerIterator) {
    x.push_back(((KernelPointCoords)dataContainerIterator->first).x);
    y.push_back(((KernelPointCoords)dataContainerIterator->first).y);
    value.push_back(dataContainerIterator->second);
  }
}

point_value_t SparseBlurKernel::calcSumOfElements() {
  point_value_t sumOfElements = 0;

  for (dataContainerIterator = dataContainer.begin();
       dataContainerIterator != dataContainer.end(); ++dataContainerIterator) {
    sumOfElements += dataContainerIterator->second;
  }

  return sumOfElements;
}

void SparseBlurKernel::normalize() {
  point_value_t sumOfElements = calcSumOfElements();

  if (sumOfElements == 0) {
    return;
  }

  for (dataContainerIterator = dataContainer.begin();
       dataContainerIterator != dataContainer.end(); ++dataContainerIterator) {
    point_value_t value = dataContainerIterator->second;
    value /= sumOfElements;
    dataContainerIterator->second = value;
  }
}

void SparseBlurKernel::calcCoordsSpan(point_value_t* xCoordMin,
                                      point_value_t* yCoordMin,
                                      point_value_t* xCoordMax,
                                      point_value_t* yCoordMax) {
  point_value_t minXcoord, minYcoord, maxXcoord, maxYcoord;

  minXcoord = std::numeric_limits<point_value_t>::max();
  minYcoord = std::numeric_limits<point_value_t>::max();

  if (std::numeric_limits<point_value_t>::is_signed) {
    maxXcoord = -std::numeric_limits<point_value_t>::max();
    maxYcoord = -std::numeric_limits<point_value_t>::max();
  } else {
    maxXcoord = std::numeric_limits<point_value_t>::min();
    maxYcoord = std::numeric_limits<point_value_t>::min();
  }

  for (dataContainerIterator = dataContainer.begin();
       dataContainerIterator != dataContainer.end(); ++dataContainerIterator) {
    point_value_t xCoord = ((KernelPointCoords)dataContainerIterator->first).x;
    point_value_t yCoord = ((KernelPointCoords)dataContainerIterator->first).y;

    if (xCoord < minXcoord) {
      minXcoord = xCoord;
    }

    if (yCoord < minYcoord) {
      minYcoord = yCoord;
    }

    if (xCoord > maxXcoord) {
      maxXcoord = xCoord;
    }

    if (yCoord > maxYcoord) {
      maxYcoord = yCoord;
    }
  }

  if (xCoordMin != NULL) {
    *xCoordMin = minXcoord;
  }

  if (yCoordMin != NULL) {
    *yCoordMin = minYcoord;
  }

  if (xCoordMax != NULL) {
    *xCoordMax = maxXcoord;
  }

  if (yCoordMax != NULL) {
    *yCoordMax = maxYcoord;
  }
}

}  // namespace Deblurring
}  // namespace Test
