#pragma once

// #include "CoreCommon.h"
// #include "Matrix.h"
// #include "Time.h"

namespace Test {
namespace OpticalFlow {

/// Abstract base for representing orientation during a time interval
template <typename NumT>
class OrientationPath {
 public:
  typedef NumT NumberT;
  virtual ~OrientationPath();

  virtual void orientation(Time time,
                           OUT Math::Matrix<NumberT, 3, 3>& orientation) = 0;
  virtual void orientation(Time time,
                           OUT Math::Versor<NumberT>& orientation) = 0;
  // The interval in which the this object can be queried to return orientation.
  virtual void domainInterval(OUT Math::Vector<Time, 2>& interval) = 0;
};

template <typename NumT>
OrientationPath<NumT>::~OrientationPath() {}

}  // namespace OpticalFlow
}  // namespace Test
