#pragma once

#include <stdint.h>
#include <time.h>

#include <limits>

#ifdef __unix
#include <sys/time.h>
#endif

namespace Test {

typedef int64_t Nanoseconds;
typedef Nanoseconds Time;
typedef int32_t TimeSmall;

template <typename T>
T timeToSeconds(Time t);
template <typename T>
T timeToMilliseconds(Time t);
template <typename T>
T timeToMicroseconds(Time t);
template <typename T>
Time timeFromSeconds(T t);
template <typename T>
Time timeFromNanoseconds(T t);
Time minTime();
Time getTimeReal();
Time getTimeMonotonic();
#ifdef __unix
Time timeFromTimeval(const timeval& tv);
timeval timeToTimeval(const Time& t);
Time timeFromTimespec(const timespec& ts);
timespec timeToTimespec(const Time& t);
#endif

template <typename T>
T timeToSeconds(Time t) {
  if (std::numeric_limits<T>::is_integer) {
    return T(t / (1000 * 1000 * 1000));
  } else {
    return T(t) / T(1000 * 1000 * 1000);
  }
}

template <typename T>
T timeToMilliseconds(Time t) {
  return T(t) / T(1000 * 1000);
}

template <typename T>
T timeToMicroseconds(Time t) {
  return T(t) / T(1000);
}

template <typename T>
Time timeFromSeconds(T t) {
  return Time(t * (1000 * 1000 * 1000));
}

inline Time timeFromSeconds(short t) {
  return (Time(t) * (1000 * 1000 * 1000));
}

inline Time timeFromSeconds(unsigned short t) {
  return (Time(t) * (1000 * 1000 * 1000));
}

inline Time timeFromSeconds(int t) { return (Time(t) * (1000 * 1000 * 1000)); }

inline Time timeFromSeconds(unsigned t) {
  return (Time(t) * (1000 * 1000 * 1000));
}

inline Time timeFromSeconds(long t) { return (Time(t) * (1000 * 1000 * 1000)); }

inline Time timeFromSeconds(unsigned long t) {
  return (Time(t) * (1000 * 1000 * 1000));
}

inline Time timeFromSeconds(float t) { return Time(t * (1000 * 1000 * 1000)); }

inline Time timeFromSeconds(double t) { return Time(t * (1000 * 1000 * 1000)); }

template <typename T>
inline Time timeFromNanoseconds(T t) {
  return t;
}

inline Time minTime() { return std::numeric_limits<Time>::min(); }

#ifdef WIN32

#elif __unix
inline Time timeFromTimeval(const timeval& tv) {
  return Time(tv.tv_sec) * 1000 * 1000 * 1000 + Time(tv.tv_usec) * 1000;
}

inline timeval timeToTimeval(const Time& t) {
  timeval result;
  result.tv_sec = t / (1000 * 1000 * 1000);
  result.tv_usec = t % (1000 * 1000 * 1000);
  result.tv_usec /= 1000;
  return result;
}

inline Time timeFromTimespec(const timespec& ts) {
  return Time(ts.tv_sec) * 1000 * 1000 * 1000 + Time(ts.tv_nsec);
}

inline timespec timeToTimespec(const Time& t) {
  timespec result;
  result.tv_sec = t / (1000 * 1000 * 1000);
  result.tv_nsec = t % (1000 * 1000 * 1000);
  return result;
}

inline Time getTimeMonotonic() {
  timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return timeFromTimespec(now);
}

inline Time getTimeReal() {
  timespec now;
  clock_gettime(CLOCK_REALTIME, &now);
  return timeFromTimespec(now);

  //        timeval now;
  //        gettimeofday(&now, 0);
  //        return (now.tv_sec*1000*1000 + now.tv_usec);
}
#else
inline Time getTimeReal() {
  time_t now = time(0);
  return timeFromSeconds(now);
}
#endif  // WIN32

}  // namespace Test
