#pragma once

#ifndef TEST_ALGORITHMS_CORE_RELEASE
#if defined(RELEASE) || defined(NDEBUG)
#define TEST_ALGORITHMS_CORE_RELEASE
#endif
#if !defined(NDEBUG) || defined(DEBUG)
#define TEST_ALGORITHMS_CORE_DEBUG
#endif
#endif

#define DO_EXPAND(VAL) VAL##1
#define EXPAND(VAL) DO_EXPAND(VAL)

#ifndef OUT
#define OUT
#elif (EXPAND(OUT) != 1)
#error "Error: preprocessor definition 'OUT' must be empty"
#endif

#ifndef IN
#define IN
#elif (EXPAND(IN) != 1)
#error "Error: preprocessor definition 'IN' must be empty"
#endif

#ifdef __cplusplus

template <typename T>
void safeDelete(T*& x) {
  delete x;
  x = 0;
}

template <typename T>
void safeDeleteArr(T*& arr) {
  delete[] arr;
  arr = 0;
}

#include <cassert>

#ifdef TEST_ALGORITHMS_CORE_DEBUG
#define TEST_ALGORITHMS_CORE_ASSERT(x) assert(x)
#else
#define TEST_ALGORITHMS_CORE_ASSERT(x)
#endif

#ifdef TEST_ALGORITHMS_CORE_USE_BOOST
#include <boost/static_assert.hpp>
#define TEST_ALGORITHMS_CORE_STATIC_ASSERT BOOST_STATIC_ASSERT
#else

namespace Test {

#define TEST_ALGORITHMS_CORE_JOIN(X, Y) TEST_ALGORITHMS_CORE_JOIN2(X, Y)
#define TEST_ALGORITHMS_CORE_JOIN2(X, Y) X##Y

template <bool>
struct STATIC_ASSERT_FAILURE;
template <>
struct STATIC_ASSERT_FAILURE<true> {
  enum { value = 1 };
};

template <int x>
struct staticAssertTest {};
}  // namespace Test

// https://blog.bonggeek.com/2008/10/
// Obsoleted by static_assert
#define TEST_ALGORITHMS_CORE_STATIC_ASSERT(x)  \
  typedef Test::staticAssertTest<sizeof(       \
      Test::STATIC_ASSERT_FAILURE<(bool)(x)>)> \
  TEST_ALGORITHMS_CORE_JOIN(_static_assert_typedef, __LINE__)

#endif

#else  // plain C

#define TEST_ALGORITHMS_CORE_ASSERT(x)
#define TEST_ALGORITHMS_CORE_STATIC_ASSERT(x)

#endif
