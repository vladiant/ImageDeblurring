#pragma once

#ifdef __cplusplus

#include <stdint.h>
#include <stdlib.h>

#include <climits>
#include <cmath>

// #include "CoreCommon.h"

#ifndef M_E
#define M_E 2.7182818284590452354 /* e */
#endif                            // M_E

#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634074 /* log_2 e */
#endif                                // M_LOG2E

#ifndef M_LOG10E
#define M_LOG10E 0.43429448190325182765 /* log_10 e */
#endif                                  // M_LOG10E

#ifndef M_LN2
#define M_LN2 0.69314718055994530942 /* log_e 2 */
#endif                               // M_LN2

#ifndef M_LN10
#define M_LN10 2.30258509299404568402 /* log_e 10 */
#endif                                // M_LN10

#ifndef M_PI
#define M_PI 3.14159265358979323846 /* pi */
#endif                              // M_PI

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923 /* pi/2 */
#endif                                // M_PI_2

#ifndef M_PI_4
#define M_PI_4 0.78539816339744830962 /* pi/4 */
#endif                                // M_PI_4

#ifndef M_1_PI
#define M_1_PI 0.31830988618379067154 /* 1/pi */
#endif                                // M_1_PI

#ifndef M_2_PI
#define M_2_PI 0.63661977236758134308 /* 2/pi */
#endif                                // M_2_PI

#ifndef M_2_SQRTPI
#define M_2_SQRTPI 1.12837916709551257390 /* 2/sqrt(pi) */
#endif                                    // M_2_SQRTPI

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880 /* sqrt(2) */
#endif                                 // M_SQRT2

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440 /* 1/sqrt(2) */
#endif                                   // M_SQRT1_2

namespace Test {
namespace Math {

template <typename T, unsigned Size>
struct Vector;

template <typename T>
T abs(const T& x);
template <typename T>
T sqrt(const T& x);
template <typename T>
T exp(const T& x);
template <typename T>
T ln(const T& x);
template <typename T>
T sin(const T& x);
template <typename T>
T cos(const T& x);
template <typename T>
T tan(const T& x);
template <typename T>
T acos(const T& x);
template <typename T>
T asin(const T& x);
template <typename T>
T atan(const T& x);
template <typename T>
T sign(const T& x);
template <typename T>
T pow(const T& x, const T& pow);
template <typename T>
T mod(const T& nominator, const T& denominator);
bool equal(float a, float b, uint_fast64_t maxUlps = 100);
bool equal(double a, double b, uint_fast64_t = 200);
bool equal(int a, int b, uint_fast64_t maxUlps = 100);
// Compare two collections element by element. If two corresponding elements are
// not equal then the two collections are not equal
template <typename It1, typename It2>
bool equal(It1 collection1Begin, It1 collection1End, It2 collection2Begin,
           uint_fast64_t maxUlps);

// Note: use only with unsigned types
template <typename T>
unsigned numberOfLeadingZeros(T x);

}  // namespace Math
}  // namespace Test

#include "Vector.h"

namespace Test {
namespace Math {

template <typename T>
T abs(const T& x) {
  return (x >= T(0) ? x : -x);
}

template <typename T>
T sqrt(const T& x) {
  return std::sqrt(x);
}

template <typename T>
T exp(const T& x) {
  return std::exp(x);
}

template <typename T>
T ln(const T& x) {
  return std::log(x);
}

template <typename T>
T sin(const T& x) {
  return std::sin(x);
}

template <typename T>
T cos(const T& x) {
  return std::cos(x);
}

template <typename T>
T tan(const T& x) {
  return std::tan(x);
}

template <typename T>
T acos(const T& x) {
  return std::acos(x);
}

template <typename T>
T asin(const T& x) {
  return std::asin(x);
}

template <typename T>
T atan(const T& x) {
  return std::atan(x);
}

template <typename T>
T sign(const T& x) {
  return x >= T(0) ? T(1) : T(-1);
}

template <typename T>
T pow(const T& x, const T& pow) {
  return std::pow(x, pow);
}

template <typename T>
T mod(const T& nominator, const T& denominator) {
  return std::fmod(nominator, denominator);
}

inline bool equal(float a, float b, uint_fast64_t maxUlps) {
  // Make sure maxUlps is small enough that the
  // default NAN won't compare as equal to anything.
  TEST_ALGORITHMS_CORE_ASSERT(maxUlps < 4 * 1024 * 1024);
  int32_t aInt;
  union CastUnion {
    float src;
    int32_t dst;
  };
  aInt = reinterpret_cast<CastUnion*>(&a)->dst;
  // Make aInt lexicographically ordered as a twos-complement int
  if (aInt < 0) {
    aInt = 0x80000000 - aInt;
  }
  // Make bInt lexicographically ordered as a twos-complement int
  int32_t bInt = reinterpret_cast<CastUnion*>(&b)->dst;
  if (bInt < 0) {
    bInt = 0x80000000 - bInt;
  }
  uint32_t intDiff = abs(aInt - bInt);
  if (intDiff <= uint_fast32_t(maxUlps)) {
    return true;
  }
  return false;
}

inline bool equal(double a, double b, uint_fast64_t maxUlps) {
  // Make sure maxUlps is small enough that the
  // default NAN won't compare as equal to anything.
  TEST_ALGORITHMS_CORE_ASSERT(maxUlps < uint_fast64_t(1) << 50);
  // int64_t aInt = *reinterpret_cast<int64_t*>(reinterpret_cast<char*>(&a));
  int64_t aInt;
  union CastUnion {
    double src;
    int64_t dst;
  };
  aInt = reinterpret_cast<CastUnion*>(&a)->dst;

  // Make aInt lexicographically ordered as a twos-complement int
  if (aInt < 0) {
    aInt = (int64_t(1) << 63) - aInt;
  }
  // Make bInt lexicographically ordered as a twos-complement int
  int64_t bInt = reinterpret_cast<CastUnion*>(&b)->dst;
  if (bInt < 0) {
    bInt = (int64_t(1) << 63) - bInt;
  }
  uint64_t intDiff = llabs(aInt - bInt);
  if (intDiff <= maxUlps) {
    return true;
  }
  return false;
}

inline bool equal(int a, int b, uint_fast64_t maxUlps) {
  unsigned diff = abs(a - b);
  if (diff <= maxUlps) {
    return true;
  }
  return false;
}

template <typename It1, typename It2>
bool equal(It1 collection1Begin, It1 collection1End, It2 collection2Begin,
           uint_fast64_t maxUlps) {
  for (; collection1Begin != collection1End;
       ++collection1Begin, ++collection2Begin) {
    if (!equal(*collection1Begin, *collection2Begin, maxUlps)) {
      return false;
    }
  }

  return true;
}

template <typename T>
unsigned numberOfLeadingZeros(T x) {
  unsigned result = sizeof(T) * CHAR_BIT;

  while (x != 0) {
    x >>= 1;
    result--;
  }

  return result;
}

}  // namespace Math
}  // namespace Test

#endif
