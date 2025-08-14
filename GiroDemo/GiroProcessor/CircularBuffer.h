#pragma once

#include <assert.h>
#include <limits.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <new>

#ifndef WIN32
#include <stdbool.h>
#endif  // WIN32

#if !defined(TEST_CB_C_VARIANT) && defined(__cplusplus)

#undef TEST_CB_CPP_VARIANT
#undef TEST_GYRO_VSTAB_TEST_CIRCULAR_BUFFER_H_
#define TEST_CB_CPP_VARIANT

#else
#undef TEST_CB_CPP_VARIANT
#endif

#ifdef TEST_CB_CPP_VARIANT
#include <iterator>
#include <new>
// #include "MathCommon.h"
#endif

#define TEST_CB_ASSERT(x) TEST_ALGORITHMS_CORE_ASSERT(x)

#ifdef TEST_CB_CPP_VARIANT
#define _TEST_CIRCULAR_BUFF_T CircularBuffer
#define TEST_CIRCULAR_BUFF_T CircularBuffer
#define TEST_CB_THIS this

namespace Test {
namespace Container {

template <typename T>
class CircularBuffer;

template <typename PointerT, bool isReverse>
class CircularBufferIterator {
 public:
  typedef typename std::iterator_traits<PointerT>::value_type value_type;
  typedef typename std::iterator_traits<PointerT>::pointer pointer;
  typedef typename std::iterator_traits<PointerT>::reference reference;
  typedef
      typename std::iterator_traits<PointerT>::difference_type difference_type;
  typedef size_t size_type;
  typedef size_t capacity_type;
  typedef std::random_access_iterator_tag iterator_category;

  CircularBufferIterator();
  template <bool isReverseSrc>
  CircularBufferIterator(
      const CircularBufferIterator<const value_type*, isReverseSrc>& it);
  template <bool isReverseSrc>
  CircularBufferIterator(
      const CircularBufferIterator<value_type*, isReverseSrc>& it);
  CircularBufferIterator(const CircularBuffer<value_type>& m_Buff,
                         size_type index);
  template <typename PointerTR, bool isReverseR>
  bool operator==(
      const CircularBufferIterator<PointerTR, isReverseR>& it) const;
  template <typename PointerTR, bool isReverseR>
  bool operator!=(
      const CircularBufferIterator<PointerTR, isReverseR>& it) const;
  template <typename PointerTR, bool isReverseR>
  bool operator<(const CircularBufferIterator<PointerTR, isReverseR>& it) const;
  template <typename PointerTR, bool isReverseR>
  bool operator<=(
      const CircularBufferIterator<PointerTR, isReverseR>& it) const;
  template <typename PointerTR, bool isReverseR>
  bool operator>(const CircularBufferIterator<PointerTR, isReverseR>& it) const;
  template <typename PointerTR, bool isReverseR>
  bool operator>=(
      const CircularBufferIterator<PointerTR, isReverseR>& it) const;
  reference operator*();
  pointer operator->();
  CircularBufferIterator operator++(int);
  CircularBufferIterator& operator++();
  CircularBufferIterator operator--(int);
  CircularBufferIterator& operator--();
  CircularBufferIterator operator+(difference_type n) const;
  CircularBufferIterator operator-(difference_type n) const;
  CircularBufferIterator& operator+=(difference_type n);
  CircularBufferIterator& operator-=(difference_type n);
  template <typename PointerTR>
  difference_type operator-(
      const CircularBufferIterator<PointerTR, isReverse>& rhs) const;
  reference operator[](difference_type n) const;

  const CircularBuffer<value_type>& buff() const;
  size_type index() const;
  size_type relativeIndex() const;

 private:
  const CircularBuffer<value_type>* m_Buff;
  size_type m_Index;
};

#else
#define TEST_CIRCULAR_BUFF_T TestCircularBuffer
#define TEST_CB_THIS pThis
#endif

#ifdef TEST_CB_CPP_VARIANT
template <typename T>
class TEST_CIRCULAR_BUFF_T
#else
struct _TEST_CIRCULAR_BUFF_T
#endif
{
#ifdef TEST_CB_CPP_VARIANT
 public:
  typedef typename CircularBufferIterator<T*, false>::value_type value_type;
  typedef typename CircularBufferIterator<T*, false>::pointer pointer;
  typedef
      typename CircularBufferIterator<const T*, false>::pointer const_pointer;
  typedef typename CircularBufferIterator<T*, false>::reference reference;
  typedef typename CircularBufferIterator<const T*, false>::reference
      const_reference;
  typedef typename CircularBufferIterator<T*, false>::difference_type
      difference_type;
  typedef typename CircularBufferIterator<T*, false>::size_type size_type;
  typedef
      typename CircularBufferIterator<T*, false>::capacity_type capacity_type;
  typedef CircularBufferIterator<pointer, false> iterator;
  typedef CircularBufferIterator<const_pointer, false> const_iterator;
  typedef CircularBufferIterator<pointer, true> reverse_iterator;
  typedef CircularBufferIterator<const_pointer, true> const_reverse_iterator;

  CircularBuffer();
  CircularBuffer(capacity_type capacity);
  CircularBuffer(pointer buffer, capacity_type capacity);
  ~CircularBuffer();
  CircularBuffer& operator=(const CircularBuffer& src);
  size_type size() const;
  capacity_type capacity() const;
  bool empty() const;
  bool full() const;
  void clear();
  void set_capacity(size_type capacity);
  reference operator[](size_type index);
  const_reference operator[](size_type index) const;
  void push_back(const_reference data);
  void push_front(const_reference data);
  void pop_back();
  void pop_front();
  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;
  reverse_iterator rbegin();
  reverse_iterator rend();
  const_reverse_iterator rbegin() const;
  const_reverse_iterator rend() const;
  reference front();
  reference back();
  const_reference front() const;
  const_reference back() const;
  iterator erase(const iterator& first, const iterator& last);
  template <typename Stream>
  void print(Stream& s);

#endif

#ifdef TEST_CB_CPP_VARIANT
 private:
  friend class CircularBufferIterator<pointer, false>;
  friend class CircularBufferIterator<const_pointer, false>;
  friend class CircularBufferIterator<pointer, true>;
  friend class CircularBufferIterator<const_pointer, true>;

  pointer m_Buff;
#else
  unsigned char* m_Buff;
#endif
  size_t m_Begin;  // Index of the first element
  size_t m_End;    // Index of the element after the last
  size_t m_ValueTypeSize;
  // m_Capacity == (1 << shiftCapacity) - 1
  // The element after the last is always dummy to distinguish form a full and
  // empty buffer
  size_t m_Capacity;
  bool m_IsBufferOwner;
};

#if !defined(TEST_CB_CPP_VARIANT)
typedef struct _TEST_CIRCULAR_BUFF_T TEST_CIRCULAR_BUFF_T;

// Capacity is expanded to be a multiple of 2
inline int TestCircularBufferCreate(OUT TestCircularBuffer* TEST_CB_THIS,
                                    size_t valueTypeSize, size_t capacity);
inline void TestCircularBufferCreateExternalBuff(
    OUT TestCircularBuffer* TEST_CB_THIS, void* buffer, size_t valueTypeSize,
    size_t capacity);
inline void TestCircularBufferDestroy(TestCircularBuffer* TEST_CB_THIS);
inline size_t TestCircularBufferSize(TestCircularBuffer* TEST_CB_THIS);
inline size_t TestCircularBufferCapacity(TestCircularBuffer* TEST_CB_THIS);
inline void TestCircularBufferPushBack(TestCircularBuffer* TEST_CB_THIS,
                                       const void* data);
inline void TestCircularBufferPushFront(TestCircularBuffer* TEST_CB_THIS,
                                        const void* data);
inline void TestCircularBufferPopBack(TestCircularBuffer* TEST_CB_THIS);
inline void TestCircularBufferPopFront(TestCircularBuffer* TEST_CB_THIS);
inline void* TestCircularBufferBack(TestCircularBuffer* TEST_CB_THIS);
inline void* TestCircularBufferFront(TestCircularBuffer* TEST_CB_THIS);
inline void* TestCircularBufferElementAt(TestCircularBuffer* TEST_CB_THIS,
                                         size_t index);
#endif  //! defined(TEST_CB_CPP_VARIANT)

#ifdef TEST_CB_CPP_VARIANT
template <typename T>
CircularBuffer<T>::CircularBuffer()
    : m_Buff(NULL),
      m_Begin(0),
      m_End(0),
      m_ValueTypeSize(sizeof(T)),
      m_Capacity(0),
      m_IsBufferOwner(true) {}
#endif

#ifdef TEST_CB_CPP_VARIANT
template <typename T>
CircularBuffer<T>::CircularBuffer(
    typename CircularBuffer<T>::capacity_type capacity)
#else
int TestCircularBufferCreate(TestCircularBuffer* TEST_CB_THIS,
                             size_t valueTypeSize, size_t capacity)
#endif
{
  // Find the smallest number that is a multiple of 2 and is greater than
  // capacity
  unsigned numberOfLeadingZeros = sizeof(size_t) * CHAR_BIT;
  // The element after the last is always dummy to distinguish form a full and
  // empty buffer
  while (capacity != 0) {
    capacity >>= 1;
    numberOfLeadingZeros--;
  }
  capacity = (size_t)1 << (sizeof(capacity) * CHAR_BIT - numberOfLeadingZeros);

#ifdef TEST_CB_CPP_VARIANT
  TEST_CB_THIS->m_Buff = (T*)malloc(capacity * sizeof(T));
  TEST_CB_THIS->m_ValueTypeSize = sizeof(T);

  // Workaround for stupid non-standard Android compiler during OS build.
  // One day, when the Android compiler is not shit, this will be unnecessary.
#ifndef __ANDROID__
  if (!TEST_CB_THIS->m_Buff) {
    throw std::bad_alloc();
  }
#endif

#else
  TEST_CB_THIS->m_Buff = (unsigned char*)malloc(capacity * valueTypeSize);
  if (!TEST_CB_THIS->m_Buff) {
    return 1;
  }
  TEST_CB_THIS->m_ValueTypeSize = valueTypeSize;
#endif
  TEST_CB_THIS->m_Capacity = capacity - 1;
  TEST_CB_THIS->m_Begin = 0;
  TEST_CB_THIS->m_End = 0;
  TEST_CB_THIS->m_IsBufferOwner = true;

#if !defined(TEST_CB_CPP_VARIANT)
  return 0;
#endif
}

#ifdef TEST_CB_CPP_VARIANT
template <typename T>
CircularBuffer<T>::CircularBuffer(
    typename CircularBuffer<T>::pointer buffer,
    typename CircularBuffer<T>::capacity_type capacity)
#else
void TestCircularBufferCreateExternalBuff(OUT TestCircularBuffer* TEST_CB_THIS,
                                          void* buffer, size_t valueTypeSize,
                                          size_t capacity)
#endif
{
#ifdef TEST_CB_CPP_VARIANT
  TEST_CB_THIS->m_ValueTypeSize = sizeof(T);
#else
  TEST_CB_THIS->m_ValueTypeSize = valueTypeSize;
#endif
  // Find the smallest number that is a multiple of 2 and is greater than
  // capacity
  unsigned numberOfLeadingZeros = sizeof(size_t) * CHAR_BIT;
  // The element after the last is always dummy to distinguish form a full and
  // empty buffer
  while (capacity != 0) {
    capacity >>= 1;
    numberOfLeadingZeros--;
  }
  capacity = (size_t)1 << (sizeof(capacity) * CHAR_BIT - numberOfLeadingZeros);

  TEST_CB_THIS->m_Capacity = capacity - 1;
  TEST_CB_THIS->m_Buff = (unsigned char*)buffer;
  TEST_CB_THIS->m_Begin = 0;
  TEST_CB_THIS->m_End = 0;
  TEST_CB_THIS->m_IsBufferOwner = false;
}

#ifdef TEST_CB_CPP_VARIANT
template <typename T>
CircularBuffer<T>::~CircularBuffer()
#else
void TestCircularBufferDestroy(TestCircularBuffer* TEST_CB_THIS)
#endif
{
  if (TEST_CB_THIS->m_IsBufferOwner) {
#ifdef TEST_CB_CPP_VARIANT
    iterator endIt = end();
    for (iterator it = begin(); it != endIt; ++it) {
      it->~T();
    }
#endif
    free(TEST_CB_THIS->m_Buff);
    TEST_CB_THIS->m_Buff = NULL;
    TEST_CB_THIS->m_Begin = 0;
    TEST_CB_THIS->m_End = 0;
  }
}

#ifdef TEST_CB_CPP_VARIANT
template <typename T>
CircularBuffer<T>& CircularBuffer<T>::operator=(const CircularBuffer<T>& src) {
  if (src.m_IsBufferOwner) {
    set_capacity(src.capacity());
    m_Begin = src.m_Begin;
    m_End = src.m_End;
    m_IsBufferOwner = true;
    m_Capacity = src.m_Capacity;
    if (src.m_Buff) {
      memcpy(m_Buff, src.m_Buff, (m_Capacity + 1) * sizeof(T));
    }
  } else {
    if (m_IsBufferOwner && m_Buff) {
      for (iterator it = begin(); it != end(); ++it) {
        it->~T();
      }
    }
    free(m_Buff);
    memcpy(this, &src, sizeof(*this));
  }

  return *this;
}
#endif

#ifdef TEST_CB_CPP_VARIANT
template <typename T>
typename CircularBuffer<T>::size_type CircularBuffer<T>::size() const
#else
size_t TestCircularBufferSize(TestCircularBuffer* TEST_CB_THIS)
#endif
{
  ptrdiff_t s = TEST_CB_THIS->m_End - TEST_CB_THIS->m_Begin;
  return s < 0 ? TEST_CB_THIS->m_Capacity + s + 1 : s;
}

#ifdef TEST_CB_CPP_VARIANT
template <typename T>
typename CircularBuffer<T>::capacity_type CircularBuffer<T>::capacity() const
#else
size_t TestCircularBufferCapacity(TestCircularBuffer* TEST_CB_THIS)
#endif
{
  return TEST_CB_THIS->m_Capacity;
}

#ifdef TEST_CB_CPP_VARIANT
template <typename T>
typename CircularBuffer<T>::reference CircularBuffer<T>::operator[](
    typename CircularBuffer<T>::size_type index)
#else
void* TestCircularBufferElementAt(TestCircularBuffer* TEST_CB_THIS,
                                  size_t index)
#endif
{
#ifdef TEST_CB_CPP_VARIANT
  TEST_CB_ASSERT(index < TEST_CB_THIS->size());
#else
  TEST_CB_ASSERT(index < TestCircularBufferSize(TEST_CB_THIS));
#endif

  index = (TEST_CB_THIS->m_Begin + index) & TEST_CB_THIS->m_Capacity;
#ifdef TEST_CB_CPP_VARIANT
  return *(TEST_CB_THIS->m_Buff + index);
#else
  return TEST_CB_THIS->m_Buff + index * TEST_CB_THIS->m_ValueTypeSize;
#endif
}

#ifdef TEST_CB_CPP_VARIANT
template <typename T>
void CircularBuffer<T>::push_back(
    typename CircularBuffer<T>::const_reference data)
#else
void TestCircularBufferPushBack(TestCircularBuffer* TEST_CB_THIS,
                                const void* data)
#endif
{
#ifdef TEST_CB_CPP_VARIANT
  T* ptrEnd = TEST_CB_THIS->m_Buff + TEST_CB_THIS->m_End;
  new (ptrEnd) T(data);
#else
  void* ptrEnd = TEST_CB_THIS->m_Buff +
                 TEST_CB_THIS->m_End * TEST_CB_THIS->m_ValueTypeSize;
  memcpy(ptrEnd, data, TEST_CB_THIS->m_ValueTypeSize);
#endif
  TEST_CB_THIS->m_End = (TEST_CB_THIS->m_End + 1) & TEST_CB_THIS->m_Capacity;
  // The buffer has overflown. There should be a dummy element at the end of the
  // buffer.
  TEST_CB_ASSERT(TEST_CB_THIS->m_Begin != TEST_CB_THIS->m_End);
}

#ifdef TEST_CB_CPP_VARIANT
template <typename T>
void CircularBuffer<T>::push_front(
    typename CircularBuffer<T>::const_reference data)
#else
void TestCircularBufferPushFront(TestCircularBuffer* TEST_CB_THIS,
                                 const void* data)
#endif
{
  TEST_CB_THIS->m_Begin =
      (TEST_CB_THIS->m_Begin - 1) & TEST_CB_THIS->m_Capacity;
#ifdef TEST_CB_CPP_VARIANT
  T* ptrBegin = TEST_CB_THIS->m_Buff + TEST_CB_THIS->m_Begin;
  new (ptrBegin) T(data);
#else
  void* ptrBegin = TEST_CB_THIS->m_Buff +
                   TEST_CB_THIS->m_Begin * TEST_CB_THIS->m_ValueTypeSize;
  memcpy(ptrBegin, data, TEST_CB_THIS->m_ValueTypeSize);
#endif
  // The buffer has overflown. There should be a dummy element at the end of the
  // buffer.
  TEST_CB_ASSERT(TEST_CB_THIS->m_Begin != TEST_CB_THIS->m_End);
}

#ifdef TEST_CB_CPP_VARIANT
template <typename T>
void CircularBuffer<T>::pop_back()
#else
void TestCircularBufferPopBack(TestCircularBuffer* TEST_CB_THIS)
#endif
{
  TEST_CB_ASSERT(TEST_CB_THIS->m_Begin != TEST_CB_THIS->m_End);
  TEST_CB_THIS->m_End = (TEST_CB_THIS->m_End - 1) & TEST_CB_THIS->m_Capacity;
#ifdef TEST_CB_CPP_VARIANT
  T* ptrEnd = TEST_CB_THIS->m_Buff + TEST_CB_THIS->m_End;
  ptrEnd->~T();
#endif
}

#ifdef TEST_CB_CPP_VARIANT
template <typename T>
void CircularBuffer<T>::pop_front()
#else
void TestCircularBufferPopFront(TestCircularBuffer* TEST_CB_THIS)
#endif
{
  TEST_CB_ASSERT(TEST_CB_THIS->m_Begin != TEST_CB_THIS->m_End);
#ifdef TEST_CB_CPP_VARIANT
  T* ptrBegin = TEST_CB_THIS->m_Buff + TEST_CB_THIS->m_Begin;
  ptrBegin->~T();
#endif
  TEST_CB_THIS->m_Begin =
      (TEST_CB_THIS->m_Begin + 1) & TEST_CB_THIS->m_Capacity;
}

#ifdef TEST_CB_CPP_VARIANT
template <typename T>
typename CircularBuffer<T>::reference CircularBuffer<T>::back()
#else
void* TestCircularBufferBack(TestCircularBuffer* TEST_CB_THIS)
#endif
{
  TEST_CB_ASSERT(TEST_CB_THIS->m_Begin != TEST_CB_THIS->m_End);
#ifdef TEST_CB_CPP_VARIANT
  return *(TEST_CB_THIS->m_Buff +
           ((TEST_CB_THIS->m_End - 1) & TEST_CB_THIS->m_Capacity));
#else
  return TEST_CB_THIS->m_Buff +
         TEST_CB_THIS->m_End * TEST_CB_THIS->m_ValueTypeSize;
#endif
}

#ifdef TEST_CB_CPP_VARIANT
template <typename T>
typename CircularBuffer<T>::reference CircularBuffer<T>::front()
#else
void* TestCircularBufferFront(TestCircularBuffer* TEST_CB_THIS)
#endif
{
  TEST_CB_ASSERT(TEST_CB_THIS->m_Begin != TEST_CB_THIS->m_End);
#ifdef TEST_CB_CPP_VARIANT
  return *(TEST_CB_THIS->m_Buff + TEST_CB_THIS->m_Begin);
#else
  return TEST_CB_THIS->m_Buff +
         TEST_CB_THIS->m_Begin * TEST_CB_THIS->m_ValueTypeSize;
  ;
#endif
}

#ifdef TEST_CB_CPP_VARIANT

template <typename PointerT, bool isReverse>
CircularBufferIterator<PointerT, isReverse>::CircularBufferIterator()
    : m_Buff(NULL) {}

template <typename PointerT, bool isReverse>
template <bool isReverseSrc>
CircularBufferIterator<PointerT, isReverse>::CircularBufferIterator(
    const CircularBufferIterator<
        typename CircularBufferIterator<PointerT, isReverse>::value_type*,
        isReverseSrc>& it)
    : m_Buff(&it.buff()), m_Index(it.index()) {}

template <typename PointerT, bool isReverse>
template <bool isReverseSrc>
CircularBufferIterator<PointerT, isReverse>::CircularBufferIterator(
    const CircularBufferIterator<
        const typename CircularBufferIterator<PointerT, isReverse>::value_type*,
        isReverseSrc>& it)
    : m_Buff(&it.buff()), m_Index(it.index()) {}

template <typename PointerT, bool isReverse>
CircularBufferIterator<PointerT, isReverse>::CircularBufferIterator(
    const CircularBuffer<value_type>& buff, size_type index)
    : m_Buff(&buff), m_Index(index) {}

template <typename PointerT, bool isReverse>
template <typename PointerTR, bool isReverseR>
bool CircularBufferIterator<PointerT, isReverse>::operator==(
    const CircularBufferIterator<PointerTR, isReverseR>& it) const {
  return m_Index == it.index();
}

template <typename PointerT, bool isReverse>
template <typename PointerTR, bool isReverseR>
bool CircularBufferIterator<PointerT, isReverse>::operator!=(
    const CircularBufferIterator<PointerTR, isReverseR>& it) const {
  return !(*this == it);
}

template <typename PointerT, bool isReverse>
template <typename PointerTR, bool isReverseR>
bool CircularBufferIterator<PointerT, isReverse>::operator<(
    const CircularBufferIterator<PointerTR, isReverseR>& it) const {
  static_assert(isReverse == isReverseR);
  size_type index = relativeIndex();
  size_type itIndex = it.relativeIndex();
  if (isReverse) {
    return itIndex < index;
  } else {
    return index < itIndex;
  }
}

template <typename PointerT, bool isReverse>
template <typename PointerTR, bool isReverseR>
bool CircularBufferIterator<PointerT, isReverse>::operator<=(
    const CircularBufferIterator<PointerTR, isReverseR>& it) const {
  static_assert(isReverse == isReverseR);
  size_type index = relativeIndex();
  size_type itIndex = it.relativeIndex();
  if (isReverse) {
    return itIndex <= index;
  } else {
    return index <= itIndex;
  }
}

template <typename PointerT, bool isReverse>
template <typename PointerTR, bool isReverseR>
bool CircularBufferIterator<PointerT, isReverse>::operator>(
    const CircularBufferIterator<PointerTR, isReverseR>& it) const {
  static_assert(isReverse == isReverseR);
  size_type index = relativeIndex();
  size_type itIndex = it.relativeIndex();
  if (isReverse) {
    return itIndex > index;
  } else {
    return index > itIndex;
  }
}

template <typename PointerT, bool isReverse>
template <typename PointerTR, bool isReverseR>
bool CircularBufferIterator<PointerT, isReverse>::operator>=(
    const CircularBufferIterator<PointerTR, isReverseR>& it) const {
  static_assert(isReverse == isReverseR);
  size_type index = relativeIndex();
  size_type itIndex = it.relativeIndex();
  if (isReverse) {
    return itIndex >= index;
  } else {
    return index >= itIndex;
  }
}

template <typename PointerT, bool isReverse>
typename CircularBufferIterator<PointerT, isReverse>::reference
CircularBufferIterator<PointerT, isReverse>::operator*() {
  return *(m_Buff->m_Buff + m_Index);
}

template <typename PointerT, bool isReverse>
typename CircularBufferIterator<PointerT, isReverse>::pointer
CircularBufferIterator<PointerT, isReverse>::operator->() {
  return m_Buff->m_Buff + m_Index;
}

template <typename PointerT, bool isReverse>
CircularBufferIterator<PointerT, isReverse>
CircularBufferIterator<PointerT, isReverse>::operator++(int) {
  CircularBufferIterator<PointerT, isReverse> result =
      CircularBufferIterator<PointerT, isReverse>(*this);
  ++(*this);
  return result;
}

template <typename PointerT, bool isReverse>
CircularBufferIterator<PointerT, isReverse>&
CircularBufferIterator<PointerT, isReverse>::operator++() {
  if (isReverse) {
    m_Index = (m_Index - 1) & m_Buff->m_Capacity;
  } else {
    m_Index = (m_Index + 1) & m_Buff->m_Capacity;
  }
  return *this;
}

template <typename PointerT, bool isReverse>
CircularBufferIterator<PointerT, isReverse>
CircularBufferIterator<PointerT, isReverse>::operator--(int) {
  CircularBufferIterator<PointerT, isReverse> result =
      CircularBufferIterator<PointerT, isReverse>(*this);
  --(*this);
  return result;
}

template <typename PointerT, bool isReverse>
CircularBufferIterator<PointerT, isReverse>&
CircularBufferIterator<PointerT, isReverse>::operator--() {
  if (isReverse) {
    m_Index = (m_Index + 1) & m_Buff->m_Capacity;
  } else {
    m_Index = (m_Index - 1) & m_Buff->m_Capacity;
  }
  return *this;
}

template <typename PointerT, bool isReverse>
CircularBufferIterator<PointerT, isReverse>
CircularBufferIterator<PointerT, isReverse>::operator+(
    difference_type n) const {
  size_type i;
  if (isReverse) {
    i = (m_Index - n) & m_Buff->m_Capacity;
  } else {
    i = (m_Index + n) & m_Buff->m_Capacity;
  }
  return CircularBufferIterator<PointerT, isReverse>(*m_Buff, i);
}

template <typename PointerT, bool isReverse>
CircularBufferIterator<PointerT, isReverse>
CircularBufferIterator<PointerT, isReverse>::operator-(
    difference_type n) const {
  size_type i;
  if (isReverse) {
    i = (m_Index + n) & m_Buff->m_Capacity;
  } else {
    i = (m_Index - n) & m_Buff->m_Capacity;
  }
  return CircularBufferIterator<PointerT, isReverse>(*m_Buff, i);
}

template <typename PointerT, bool isReverse>
CircularBufferIterator<PointerT, isReverse>&
CircularBufferIterator<PointerT, isReverse>::operator+=(difference_type n) {
  if (isReverse) {
    m_Index = (m_Index - n) & m_Buff->m_Capacity;
  } else {
    m_Index = (m_Index + n) & m_Buff->m_Capacity;
  }
  return *this;
}

template <typename PointerT, bool isReverse>
CircularBufferIterator<PointerT, isReverse>&
CircularBufferIterator<PointerT, isReverse>::operator-=(difference_type n) {
  if (isReverse) {
    m_Index = (m_Index + n) & m_Buff->m_Capacity;
  } else {
    m_Index = (m_Index - n) & m_Buff->m_Capacity;
  }
  return *this;
}

template <typename PointerT, bool isReverse>
template <typename PointerTR>
typename CircularBufferIterator<PointerT, isReverse>::difference_type
CircularBufferIterator<PointerT, isReverse>::operator-(
    const CircularBufferIterator<PointerTR, isReverse>& rhs) const {
  TEST_CB_ASSERT(m_Buff == &rhs.buff());
  size_type index = relativeIndex();
  size_type itIndex = rhs.relativeIndex();
  if (isReverse) {
    return difference_type(m_Buff->capacity()) - difference_type(index) +
           difference_type(itIndex) + 1;
  } else {
    return difference_type(index) - difference_type(itIndex);
  }
}

template <typename PointerT, bool isReverse>
typename CircularBufferIterator<PointerT, isReverse>::reference
CircularBufferIterator<PointerT, isReverse>::operator[](
    typename CircularBufferIterator<PointerT, isReverse>::difference_type n)
    const {
  size_type i;
  if (isReverse) {
    TEST_CB_ASSERT(relativeIndex() - n < m_Buff->size());
    i = (m_Index - n) & m_Buff->m_Capacity;
  } else {
    TEST_CB_ASSERT(relativeIndex() + n < m_Buff->size());
    i = (m_Index + n) & m_Buff->m_Capacity;
  }
  return *(m_Buff->m_Buff + i);
}

template <typename PointerT, bool isReverse>
const CircularBuffer<
    typename CircularBufferIterator<PointerT, isReverse>::value_type>&
CircularBufferIterator<PointerT, isReverse>::buff() const {
  return *m_Buff;
}

template <typename PointerT, bool isReverse>
typename CircularBufferIterator<PointerT, isReverse>::size_type
CircularBufferIterator<PointerT, isReverse>::index() const {
  return m_Index;
}

template <typename PointerT, bool isReverse>
typename CircularBufferIterator<PointerT, isReverse>::size_type
CircularBufferIterator<PointerT, isReverse>::relativeIndex() const {
  return (m_Index - m_Buff->m_Begin) & m_Buff->m_Capacity;
}

template <typename T>
bool CircularBuffer<T>::empty() const {
  return m_Begin == m_End;
}

template <typename T>
bool CircularBuffer<T>::full() const {
  return m_Begin == ((m_End + 1) & m_Capacity);
}

template <typename T>
void CircularBuffer<T>::clear() {
  iterator endIt = end();
  for (iterator it = begin(); it != endIt; ++it) {
    it->~T();
  }

  m_Begin = 0;
  m_End = 0;
}

template <typename T>
void CircularBuffer<T>::set_capacity(size_type capacity) {
  // Find the smallest number that is a multiple of 2 and is greater than
  // capacity
  unsigned numberOfLeadingZeros = sizeof(size_t) * CHAR_BIT;
  // The element after the last is always dummy to distinguish form a full and
  // empty buffer
  while (capacity != 0) {
    capacity >>= 1;
    numberOfLeadingZeros--;
  }
  capacity = (size_t)1 << (sizeof(capacity) * CHAR_BIT - numberOfLeadingZeros);
  CircularBuffer<T> newBuff;
  newBuff.m_Buff = reinterpret_cast<T*>(malloc(capacity * sizeof(T)));
  newBuff.m_Capacity = capacity - 1;
  newBuff.m_Begin = 0;
  newBuff.m_End = 0;
  newBuff.m_IsBufferOwner = false;
  newBuff.m_ValueTypeSize = sizeof(T);
  size_type i = 0;
  CircularBuffer<T>::iterator itSrc = begin();
  CircularBuffer<T>::iterator itDest = newBuff.begin();
  size_type nElementsInBuff = size();
  for (; i < newBuff.m_Capacity && i < nElementsInBuff;
       ++i, ++itSrc, ++itDest) {
    new (&(*itDest)) T(*itSrc);
    itDest->~T();
  }
  iterator itEnd = end();
  for (; itSrc < itEnd; ++itSrc) {
    itSrc->~T();
  }
  newBuff.m_End = itDest.index();
  bool isBufferOwner = m_IsBufferOwner;
  if (this->m_Buff != NULL) {
    free(this->m_Buff);
  }
  memcpy(this, &newBuff, sizeof(*this));
  m_IsBufferOwner = isBufferOwner;
}

template <typename T>
typename CircularBuffer<T>::const_reference CircularBuffer<T>::operator[](
    typename CircularBuffer<T>::size_type index) const {
  TEST_CB_ASSERT(index < size());
  return *(m_Buff + ((m_Begin + index) & m_Capacity));
}

template <typename T>
typename CircularBuffer<T>::iterator CircularBuffer<T>::begin() {
  return iterator(*this, m_Begin);
}

template <typename T>
typename CircularBuffer<T>::iterator CircularBuffer<T>::end() {
  return iterator(*this, m_End);
}

template <typename T>
typename CircularBuffer<T>::const_iterator CircularBuffer<T>::begin() const {
  return const_iterator(*this, m_Begin);
}

template <typename T>
typename CircularBuffer<T>::const_iterator CircularBuffer<T>::end() const {
  return const_iterator(*this, m_End);
}

template <typename T>
typename CircularBuffer<T>::reverse_iterator CircularBuffer<T>::rbegin() {
  size_type reverseBegin = (m_End - 1) & m_Capacity;
  return reverse_iterator(*this, reverseBegin);
}

template <typename T>
typename CircularBuffer<T>::reverse_iterator CircularBuffer<T>::rend() {
  size_type reverseEnd = (m_Begin - 1) & m_Capacity;
  return reverse_iterator(*this, reverseEnd);
}

template <typename T>
typename CircularBuffer<T>::const_reverse_iterator CircularBuffer<T>::rbegin()
    const {
  size_type reverseBegin = (m_End - 1) & m_Capacity;
  return const_reverse_iterator(*this, reverseBegin);
}

template <typename T>
typename CircularBuffer<T>::const_reverse_iterator CircularBuffer<T>::rend()
    const {
  size_type reverseEnd = (m_Begin - 1) & m_Capacity;
  return const_reverse_iterator(*this, reverseEnd);
}

template <typename T>
typename CircularBuffer<T>::const_reference CircularBuffer<T>::back() const {
  return *(this->m_Buff + ((this->m_End - 1) & this->m_Capacity));
}

template <typename T>
typename CircularBuffer<T>::const_reference CircularBuffer<T>::front() const {
  return *(this->m_Buff + this->m_Begin);
}

template <typename T>
typename CircularBuffer<T>::iterator CircularBuffer<T>::erase(
    const iterator& first, const iterator& last) {
  TEST_CB_ASSERT(first <= last && begin() <= first && last <= end());
  iterator itEnd = end();
  iterator itBegin = begin();
  size_type sizeOfBlockBefore = first - itBegin;
  size_type sizeOfBlockAfter = itEnd - last;
  if (sizeOfBlockBefore < sizeOfBlockAfter) {
    reverse_iterator rFirst = last - 1;
    reverse_iterator rLast = first - 1;
    reverse_iterator itDest = rFirst;
    reverse_iterator itSrc = rLast;
    reverse_iterator itREnd = rend();
    for (; itDest != rLast; ++itDest) {
      itDest->~T();
      if (itREnd != itSrc) {
        memcpy(&*itDest, &*itSrc, sizeof(T));
        ++itSrc;
      }
    }
    for (; itSrc != itREnd; ++itSrc, ++itDest) {
      memcpy(&*itDest, &*itSrc, sizeof(T));
    }
    size_type nElementToDelete = last - first;
    m_Begin = (m_Begin + nElementToDelete) & m_Capacity;
  } else {
    iterator itDest = first;
    iterator itSrc = last;
    for (; itDest != last; ++itDest) {
      itDest->~T();
      if (itEnd != itSrc) {
        memcpy(&*itDest, &*itSrc, sizeof(T));
        ++itSrc;
      }
    }
    for (; itSrc != itEnd; ++itSrc, ++itDest) {
      memcpy(&*itDest, &*itSrc, sizeof(T));
    }
    size_type nElementToDelete = last - first;
    m_End = (m_End - nElementToDelete) & m_Capacity;
  }
  return first;
}

template <typename T>
template <typename Stream>
void CircularBuffer<T>::print(Stream& s) {
  s << '{';
  if (m_Begin < m_End) {
    for (size_type i = 0; i < m_Capacity; ++i) {
      if (m_Begin <= i && i < m_End) {
        s << m_Buff[i];
      } else {
        s << "X";
      }

      if (i != m_Capacity - 1) {
        s << ", ";
      }
    }
  } else {
    for (size_type i = 0; i < m_Capacity - 1; ++i) {
      if (m_End <= i && i < m_Begin) {
        s << "X";
      } else {
        s << m_Buff[i];
      }

      if (i != m_Capacity - 1) {
        s << ", ";
      }
    }
  }

  s << '}' << '\n';
}

#endif  // TEST_CB_CPP_VARIANT

#ifdef TEST_CB_CPP_VARIANT
}  // namespace Container
}  // namespace Test
#endif

#undef TEST_CIRCULAR_BUFF_T
#undef TEST_CB_THIS
