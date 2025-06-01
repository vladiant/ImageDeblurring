#pragma once

#include <algorithm>
#include <deque>
#include <utility>

#include "CircularBuffer.h"
#include "Common.h"
// #include "CoreCommon.h"
#include "OrientationPath.h"

namespace Test {
namespace OpticalFlow {

template <typename RotT>
class DiscreteOrientationPath
    : public OrientationPath<typename RotT::value_type> {
 public:
  typedef OrientationPath<typename RotT::value_type> Base;
  typedef RotT RotationT;
  typedef Test::Container::CircularBuffer<TimeOrientation<RotationT> > BufferT;
  DiscreteOrientationPath(unsigned orientationsBufferSize);
  void orientation(Time time,
                   OUT Math::Matrix<typename Base::NumberT, 3, 3>& orientation);
  void orientation(Time time,
                   OUT Math::Versor<typename Base::NumberT>& orientation);
  void domainInterval(OUT Math::Vector<Time, 2>& interval);
  BufferT& orientations();
  const BufferT& orientations() const;
  // Discards all rotation data before time t. This means that "orientation" can
  // not be called for times before t.
  void discardOrientationDataBeforeTime(Time t);

  static bool isLess(typename BufferT::const_reference a,
                     typename BufferT::const_reference b);

 protected:
  // The inheritor of that class must reserve the necessary space.
  BufferT m_Orientations;
};

template <typename RotT>
DiscreteOrientationPath<RotT>::DiscreteOrientationPath(
    unsigned orientationsBufferSize)
    : m_Orientations(orientationsBufferSize) {}

template <typename RotT>
void DiscreteOrientationPath<RotT>::orientation(
    Time time,
    OUT Math::Matrix<typename DiscreteOrientationPath<RotT>::Base::NumberT, 3,
                     3>& orientation) {
  RotT rot;
  this->orientation(time, rot);
  rot.toRotationMatrix(orientation);
}

template <typename RotT>
void DiscreteOrientationPath<RotT>::orientation(
    Time time, OUT Math::Versor<typename Base::NumberT>& orientation) {
  typename BufferT::iterator it =
      std::lower_bound(m_Orientations.begin(), m_Orientations.end(),
                       typename BufferT::value_type(time, RotT()), isLess);
  typename BufferT::iterator prevIt = it - 1;
  typename BufferT::iterator nextIt = it + 1;
  typename RotT::value_type prevItDist = Math::abs(prevIt->time - time);
  typename RotT::value_type itDist = Math::abs(it->time - time);
  typename RotT::value_type nextItDist = Math::abs(nextIt->time - time);
  if (prevItDist < itDist) {
    orientation = prevIt->orientation;
  } else if (nextItDist < itDist) {
    orientation = nextIt->orientation;
  } else {
    orientation = it->orientation;
  }
}

template <typename RotT>
void DiscreteOrientationPath<RotT>::domainInterval(
    OUT Math::Vector<Time, 2>& interval) {
  interval.x() = m_Orientations.front().time;
  interval.y() = m_Orientations.back().time;
}

template <typename RotT>
typename DiscreteOrientationPath<RotT>::BufferT&
DiscreteOrientationPath<RotT>::orientations() {
  return m_Orientations;
}

template <typename RotT>
const typename DiscreteOrientationPath<RotT>::BufferT&
DiscreteOrientationPath<RotT>::orientations() const {
  return m_Orientations;
}

template <typename RotT>
void DiscreteOrientationPath<RotT>::discardOrientationDataBeforeTime(Time t) {
  typename BufferT::iterator it =
      std::lower_bound(m_Orientations.begin(), m_Orientations.end(),
                       typename BufferT::value_type(t, RotationT()), isLess);
  m_Orientations.erase(m_Orientations.begin(), it);
}

template <typename RotT>
bool DiscreteOrientationPath<RotT>::isLess(
    typename BufferT::const_reference a, typename BufferT::const_reference b) {
  return a.time < b.time;
}

}  // namespace OpticalFlow
}  // namespace Test
