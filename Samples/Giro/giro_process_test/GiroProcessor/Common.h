#pragma once

#ifdef __cplusplus

#include "Time.h"
#include "Vector.h"
#include "Versor.h"

namespace Test {
namespace OpticalFlow {

template <typename NumT>
struct TimeAngularVelocity {
  typedef NumT NumberT;
  typedef Math::Vector<NumT, 3> VelocityT;  // Radians / sec

  TimeAngularVelocity();
  TimeAngularVelocity(Time time, const VelocityT& velocity);

  Time time;
  VelocityT velocity;  // Radians / sec
};

template <typename NumT>
struct TimeAcceleration {
  typedef NumT NumberT;
  typedef Math::Vector<NumT, 3> Acceleration;  // meters / sec

  TimeAcceleration();
  TimeAcceleration(Time time, const Math::Vector<NumT, 3>& acc);

  Time time;
  Acceleration acceleration;  // meters / sec
};

template <typename RotT>
struct TimeOrientation {
  typedef RotT OrientationT;

  TimeOrientation();
  TimeOrientation(Time time, const OrientationT& orientation);

  Time time;
  OrientationT orientation;
};

template <typename Num, typename Stream>
Stream& operator<<(Stream& s, const TimeAngularVelocity<Num>& w) {
  s << w.time << " " << w.velocity[0] << " " << w.velocity[1] << " "
    << w.velocity[2];
  return s;
}

template <typename Num, typename Stream>
Stream& operator<<(Stream& s, const TimeAcceleration<Num>& acc) {
  s << acc.time << " " << acc.acceleration[0] << " " << acc.acceleration[1]
    << " " << acc.acceleration[2];
  return s;
}

template <typename NumT>
TimeAngularVelocity<NumT>::TimeAngularVelocity() {}

template <typename NumT>
TimeAngularVelocity<NumT>::TimeAngularVelocity(Time time,
                                               const VelocityT& velocity)
    : time(time), velocity(velocity) {}

template <typename NumT>
TimeAcceleration<NumT>::TimeAcceleration() {}

template <typename NumT>
TimeAcceleration<NumT>::TimeAcceleration(Time time,
                                         const Math::Vector<NumT, 3>& acc)
    : time(time), acceleration(acc) {}

template <typename NumT>
TimeOrientation<NumT>::TimeOrientation() {}

template <typename NumT>
TimeOrientation<NumT>::TimeOrientation(Time time,
                                       const OrientationT& orientation)
    : time(time), orientation(orientation) {}

typedef enum {
  TEST_OPTICAL_FLOW_OK = 0,
  TEST_OPTICAL_FLOW_ALLOCATION_FAILURE,
  TEST_OPTICAL_FLOW_BUFFER_OVERFLOW_NONFATAL,
  TEST_OPTICAL_FLOW_GYRO_TIMESTAMPS_UNORDERED,
  TEST_OPTICAL_FLOW_GYRO_DATA_ALREADY_DISCARDED,
  TEST_OPTICAL_FLOW_GYRO_DATA_NOT_AVAIBLE_YET,
  TEST_OPTICAL_FLOW_SAMPLE_NOT_INTEGRATED_SYSTEM_STARTING_UP,
  TEST_OPTICAL_FLOW_STATUS_COUNT,
  TEST_OPTICAL_FLOW_GUARANTEE_INT_SIZE = 0xFFFFFFFF,
} ReturnCode;

}  // namespace OpticalFlow
}  // namespace Test

#endif  // #ifdef __cplusplus
