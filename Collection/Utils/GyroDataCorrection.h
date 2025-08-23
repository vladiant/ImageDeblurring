#pragma once

#include "Common.h"
// #include "CoreCommon.h"
// #include "Time.h"

// keep it a power of two!
#define TEST_CORE_TOO_MANY_SKIPPED_GYRO_SAMPLES 16

namespace Test {
namespace OpticalFlow {

template <typename AngularVelocityNumT>
class GyroDataCorrection {
 public:
  typedef TimeAngularVelocity<AngularVelocityNumT> GyroSample_t;

  GyroDataCorrection(Time maximumToleratedDelay, unsigned samplesBeforehand);
  Time averageGyroPeriod() const;
  // For correct opperation, the object needs several gyro samples before any
  // output is expected. Feed this function gyro data until it returns true.
  bool startup(const GyroSample_t &gyroSample);

  bool sweepForInsertOrCorrect(GyroSample_t &gyroSample);

  GyroSample_t get() const;

 private:
  Time m_avvGyroPeriod;
  Time m_firstGyroTimestamp;
  GyroSample_t m_previousGyroSample;
  long int m_gyroSamplesCount;
  GyroSample_t m_toBeInserted;

  const Time M_maximumToleratedDelay;
  const unsigned M_samplesBeforehand;
};

////////////////////Implementation//////////////////////////////

template <typename AngularVelocityNumT>
GyroDataCorrection<AngularVelocityNumT>::GyroDataCorrection(
    Time maximumToleratedDelay, unsigned samplesBeforehand)
    : M_maximumToleratedDelay(maximumToleratedDelay),
      M_samplesBeforehand(samplesBeforehand) {
  m_gyroSamplesCount = -1;
}

template <typename AngularVelocityNumT>
Time GyroDataCorrection<AngularVelocityNumT>::averageGyroPeriod() const {
  return m_avvGyroPeriod;
}

template <typename AngularVelocityNumT>
bool GyroDataCorrection<AngularVelocityNumT>::startup(
    const GyroSample_t &gyroSample) {
  if (m_gyroSamplesCount == -1) {
    m_firstGyroTimestamp = gyroSample.time;
    m_gyroSamplesCount++;
    m_previousGyroSample = gyroSample;
    return false;
  } else if (m_gyroSamplesCount < M_samplesBeforehand) {
    m_gyroSamplesCount++;
    m_previousGyroSample = gyroSample;
    m_avvGyroPeriod =
        (gyroSample.time - m_firstGyroTimestamp) / m_gyroSamplesCount;
    return false;
  } else {
    return true;
  }
}

/*
 * 12___789
 * Suppose samples 3, 4, 5, 6 are skipped.
 * We are looking at sample 7, 2 is our previous.
 * We detect skipped samples and reconstruct sample 3. It is a valid sample, so
 * it sets itself as our previous. From there we should rescan with our updated
 * previous sample. Inserted samples are returned decentered and uncounted -
 * i.e. undistinguishable from real gyro output.
 *
 * Return: true if a sample should be inserted.
 */
template <typename AngularVelocityNumT>
bool GyroDataCorrection<AngularVelocityNumT>::sweepForInsertOrCorrect(
    GyroSample_t &gyroSample) {
  Time diff = gyroSample.time - m_previousGyroSample.time;
  Time validityTreshold = m_avvGyroPeriod + M_maximumToleratedDelay;
  Time skippedSampleTreshold = 2 * m_avvGyroPeriod - M_maximumToleratedDelay;

  // sample is missing
  // dont count it or set it as previous - it will do this on its own
  if (diff >= skippedSampleTreshold) {
    // protect against deep recursion leading to stack overflow
    if (diff > TEST_CORE_TOO_MANY_SKIPPED_GYRO_SAMPLES * m_avvGyroPeriod) {
      // error, but the best we can do is return without restoring any samples
      return false;
    }

    m_toBeInserted.time = m_previousGyroSample.time + m_avvGyroPeriod;
    m_toBeInserted.velocity =
        (gyroSample.velocity + m_previousGyroSample.velocity) / 2;
    return true;
  }

  // sample is delayed
  if (diff > validityTreshold && diff < skippedSampleTreshold) {
    gyroSample.time = m_previousGyroSample.time + m_avvGyroPeriod;
  }

  m_previousGyroSample = gyroSample;
  m_gyroSamplesCount++;
  m_avvGyroPeriod =
      (gyroSample.time - m_firstGyroTimestamp) / m_gyroSamplesCount;

  return false;
}

template <typename AngularVelocityNumT>
typename GyroDataCorrection<AngularVelocityNumT>::GyroSample_t
GyroDataCorrection<AngularVelocityNumT>::get() const {
  return m_toBeInserted;
}

}  // namespace OpticalFlow
}  // namespace Test
