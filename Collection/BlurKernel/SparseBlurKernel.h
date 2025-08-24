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
 * @file SparseBlurkernel_test.h
 *
 * ^path /sparseBlurKernel/include/SparseBlurkernel_test.h
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

#ifndef SPARSEBLURKERNEL_H_
#define SPARSEBLURKERNEL_H_

#include <stdint.h>

#include <map>
#include <vector>

namespace Test {
namespace Deblurring {

/// Key to map point coordinates.
typedef uint32_t point_key_t;

/// Type of point coordinates.
typedef int16_t point_coord_t;

/// Type of point values.
typedef float point_value_t;

//
union KernelPointCoords {
  point_coord_t x;
  point_coord_t y;

  point_key_t key;

  KernelPointCoords() { key = 0; }

  KernelPointCoords(point_key_t inputKey) { key = inputKey; }

  KernelPointCoords(point_coord_t inputX, point_coord_t inputY) {
    x = inputX;
    y = inputY;
  }
};

/// @brief Class to manage a blur kernel with sparse data points.
/// Uses the map container from the STL library.
class SparseBlurKernel {
 public:
  /// @brief Default constructor.
  SparseBlurKernel();

  /// @brief Default destructor.
  ~SparseBlurKernel();

  /// @brief Clears the points stored in the kernel.
  void clear();

  /// @brief Sets a point in the sparse kernel.
  void setPointValue(point_coord_t x, point_coord_t y, point_value_t value);

  /// @brief Sums a point with the input value.
  void addToPointValue(point_coord_t x, point_coord_t y,
                       point_value_t valueToAdd);

  /// @brief Sets a point to zero the sparse kernel.
  void clearPointValue(point_coord_t x, point_coord_t y);

  /// @brief Returns the value of the point.
  /// If it is not set previously returns zero.
  point_value_t getPointValue(point_coord_t x, point_coord_t y);

  /// @brief Returns the number of the points of the kernel.
  int getKernelSize() const;

  /// @brief Extracts the coordinates and the values of the
  /// kernel points into the supplied buffers.
  void extractKernelPoints(point_coord_t* x, point_coord_t* y,
                           point_value_t* value);

  /// @brief Extracts the coordinates and the values of the
  /// kernel points into the supplied vectors.
  void extractKernelPoints(std::vector<point_coord_t>& x,
                           std::vector<point_coord_t>& y,
                           std::vector<point_value_t>& value);

  /// @brief Calculates sum of kernel elements.
  point_value_t calcSumOfElements();

  /// @brief Normalizes kernel to have a sum of elements equal to one.
  void normalize();

  /// @brief Calculates span of the kernel.
  /// Returns the minimum and maximum values of the kernel coordinates.
  void calcCoordsSpan(point_value_t* xCoordMin, point_value_t* yCoordMin,
                      point_value_t* xCoordMax, point_value_t* yCoordMax);

 private:
  /// Map container of the kernel points
  /// Key: coordinates of the data point.
  /// Value: Pixel value of that point.
  std::map<point_key_t, point_value_t> dataContainer;

  /// Iterator variable to cycle through the kernel data points.
  std::map<point_key_t, point_value_t>::iterator dataContainerIterator;
};

}  // namespace Deblurring
}  // namespace Test
#endif /* SPARSEBLURKERNEL_H_ */
