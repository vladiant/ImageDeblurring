#pragma once

#include <utility>

// #include "Common.h"
#include "CoreCommon.h"
#include "DiscreteOrientationPath.h"

namespace Test {
namespace OpticalFlow {

template <typename RotT, typename AngularVelocityNumT>
class DiscreteAngularVelocitiesIntegrator {
 public:
  typedef RotT RotationT;
  typedef AngularVelocityNumT AngularVelocityNumberT;
  DiscreteAngularVelocitiesIntegrator(unsigned orientationsBufferSize);

  // Returns true if the angular velocities have been added. If a buffer
  // overflow would occur the entry is not added and false is returned.
  bool addAngularVelocity(
      const TimeAngularVelocity<AngularVelocityNumberT>& anglularVelocity);

  const DiscreteOrientationPath<RotT>& orientations() const;
  DiscreteOrientationPath<RotT>& orientations();

 protected:
  DiscreteOrientationPath<RotT> m_DiscreteOrientationPath;
};

template <typename RotT, typename AngularVelocityNumT>
DiscreteAngularVelocitiesIntegrator<RotT, AngularVelocityNumT>::
    DiscreteAngularVelocitiesIntegrator(unsigned orientationsBufferSize)
    : m_DiscreteOrientationPath(orientationsBufferSize) {}

template <typename RotT, typename AngularVelocityNumT>
bool DiscreteAngularVelocitiesIntegrator<RotT, AngularVelocityNumT>::
    addAngularVelocity(
        const TimeAngularVelocity<AngularVelocityNumberT>& anglularVelocity) {
  if (m_DiscreteOrientationPath.orientations().size() ==
      m_DiscreteOrientationPath.orientations().capacity()) {
    return false;
  }

  Math::Vector<AngularVelocityNumberT, 3> rotationAxis;
  if (m_DiscreteOrientationPath.orientations().size() == 0) {
    rotationAxis[0] = AngularVelocityNumberT(0);
    rotationAxis[1] = AngularVelocityNumberT(0);
    rotationAxis[2] = AngularVelocityNumberT(0);
    m_DiscreteOrientationPath.orientations().push_back(
        typename DiscreteOrientationPath<RotT>::BufferT::value_type(
            anglularVelocity.time, RotationT(rotationAxis)));
  } else {
    Time dt = anglularVelocity.time -
              m_DiscreteOrientationPath.orientations().rbegin()->time;
    AngularVelocityNumberT dtSeconds =
        timeToSeconds<AngularVelocityNumberT>(dt);
    rotationAxis = anglularVelocity.velocity * dtSeconds;

    typename DiscreteOrientationPath<RotT>::BufferT::value_type rot(
        anglularVelocity.time,
        RotationT(rotationAxis) *
            m_DiscreteOrientationPath.orientations().rbegin()->orientation);
    rot.orientation.normalize();

    m_DiscreteOrientationPath.orientations().push_back(rot);
  }
  return true;
}

template <typename RotT, typename AngularVelocityNumT>
const DiscreteOrientationPath<RotT>& DiscreteAngularVelocitiesIntegrator<
    RotT, AngularVelocityNumT>::orientations() const {
  return m_DiscreteOrientationPath;
}

template <typename RotT, typename AngularVelocityNumT>
DiscreteOrientationPath<RotT>&
DiscreteAngularVelocitiesIntegrator<RotT, AngularVelocityNumT>::orientations() {
  return m_DiscreteOrientationPath;
}

}  // namespace OpticalFlow
}  // namespace Test
