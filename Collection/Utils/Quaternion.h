#pragma once

#include "Vector.h"

namespace Test {
namespace Math {

template <typename T, unsigned nRows, unsigned nCols>
struct Matrix;

template <typename T>
struct Quaternion : public Vector<T, 4> {
  typedef T value_type;
  typedef T& reference;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef Quaternion<value_type> Self;
  enum { Size = 4 };
  typedef Vector<T, Size - 1> ImaginaryPart;

  Quaternion();
  explicit Quaternion(const value_type& initVal);
  Quaternion(const value_type& w, const value_type& x, const value_type& y,
             const value_type& z);
  template <typename RotAxisNumT>
  explicit Quaternion(const Vector<RotAxisNumT, 3>& rotationAxis);
  static Quaternion identity() { return Quaternion(1, 0, 0, 0); };
  Quaternion operator-() const;
  Quaternion<value_type>& operator+=(const Quaternion<value_type>& rhs);
  Quaternion<value_type>& operator+=(const value_type& rhs);
  Quaternion<value_type>& operator-=(const Quaternion<value_type>& rhs);
  Quaternion<value_type>& operator-=(const value_type& rhs);
  Quaternion<value_type>& operator*=(const Quaternion<value_type>& rhs);
  Quaternion<value_type>& operator*=(const value_type& rhs);
  Quaternion<value_type>& operator/=(const value_type& rhs);
  Quaternion<value_type> operator+(const Quaternion<value_type>& rhs) const;
  Quaternion<value_type> operator+(const value_type& rhs) const;
  Quaternion<value_type> operator-(const Quaternion<value_type>& rhs) const;
  Quaternion<value_type> operator-(const value_type& rhs) const;
  Vector<value_type, 3> operator*(const Vector<value_type, 3>& v) const;
  Quaternion<value_type> operator*(const Quaternion<value_type>& rhs) const;
  Quaternion<value_type> operator*(const value_type& rhs) const;
  Quaternion<value_type> operator/(const value_type& rhs) const;
  value_type& w();
  const value_type& w() const;
  value_type& x();
  const value_type& x() const;
  value_type& y();
  const value_type& y() const;
  value_type& z();
  const value_type& z() const;
  ImaginaryPart& imaginaryVector();
  const ImaginaryPart& imaginaryVector() const;
  Quaternion<value_type> imaginary() const;
  Quaternion<value_type>& conjugate();
  Quaternion<value_type> getConjugate() const;
  using Vector<value_type, Size>::norm;
  Quaternion normalized() const;
  Quaternion<value_type>& invert();
  Quaternion<value_type> inverse() const;
  Quaternion<value_type>& exp();
  Quaternion<value_type>& exp(const value_type& normImaginary);
  Quaternion<value_type> getExp() const;
  Quaternion<value_type> getExp(const value_type& normImaginary) const;
  Quaternion<value_type>& ln();
  Quaternion<value_type> getLn() const;
  Quaternion<value_type>& ln(const value_type& norm,
                             const value_type& normImaginary);
  Quaternion<value_type> getLn(const value_type& norm,
                               const value_type& normImaginary) const;
  Quaternion<value_type>& pow(const value_type& x);
  Quaternion<value_type> getPow(const value_type& x) const;
  template <unsigned nMatrixRows, unsigned nMatrixCols>
  void toRotationMatrix(
      IN OUT Matrix<value_type, nMatrixRows, nMatrixCols>& m) const;
  value_type rotationAngle() const;

 protected:
  //     using Vector<value_type, Size>::operator -;
  //     using Vector<value_type, Size>::operator +=;
  //     using Vector<value_type, Size>::operator -=;
  //     using Vector<value_type, Size>::operator *=;
  //     using Vector<value_type, Size>::operator /=;
  //     using Vector<value_type, Size>::operator +;
  //    using Vector<value_type, Size>::operator *;
  //    using Vector<value_type, Size>::operator /;
  using Vector<value_type, Size>::crossProduct;
  using Vector<value_type, Size>::normalized;

  using Vector<T, Size>::m_Vals;
};

}  // namespace Math
}  // namespace Test

#include "Matrix.h"

namespace Test {
namespace Math {

typedef Quaternion<float> QuaternionF;

template <typename T>
Quaternion<T>::Quaternion() : Vector<T, Size>() {}

template <typename T>
Quaternion<T>::Quaternion(const value_type& initVal)
    : Vector<T, Size>(initVal) {}

template <typename T>
Quaternion<T>::Quaternion(const value_type& w, const value_type& x,
                          const value_type& y, const value_type& z) {
  m_Vals[0] = w;
  m_Vals[1] = x;
  m_Vals[2] = y;
  m_Vals[3] = z;
}

template <typename T>
template <typename RotAxisNumT>
Quaternion<T>::Quaternion(const Vector<RotAxisNumT, 3>& rotationAxis)
    : Vector<T, Size>() {
  value_type angleSize = value_type(rotationAxis.norm());
  if (equal(angleSize, value_type(0))) {
    m_Vals[0] = 1;
    m_Vals[1] = 0;
    m_Vals[2] = 0;
    m_Vals[3] = 0;
  } else {
    value_type angleSizeDiv2 = angleSize / 2;
    value_type sin = sin(angleSizeDiv2);
    value_type cos = cos(angleSizeDiv2);
    m_Vals[0] = cos;
    m_Vals[1] = sin * value_type(rotationAxis[0]) / angleSize;
    m_Vals[2] = sin * value_type(rotationAxis[1]) / angleSize;
    m_Vals[3] = sin * value_type(rotationAxis[2]) / angleSize;
  }
}

template <typename T>
Quaternion<T> Quaternion<T>::operator-() const {
  Quaternion<T> result;
  for (unsigned i = 0; i < Size; ++i) {
    result.m_Vals[i] = -m_Vals[i];
  }
  return result;
}

template <typename T>
Quaternion<T>& Quaternion<T>::operator+=(const Quaternion<T>& rhs) {
  Vector<T, Size>::operator+=(rhs);
  return *this;
}

template <typename T>
Quaternion<T>& Quaternion<T>::operator+=(const T& rhs) {
  w() += rhs;
  return *this;
}

template <typename T>
Quaternion<T>& Quaternion<T>::operator-=(const Quaternion<T>& rhs) {
  Vector<T, Size>::operator-=(rhs);
  return *this;
}

template <typename T>
Quaternion<T>& Quaternion<T>::operator-=(const T& rhs) {
  Vector<T, Size>::operator-=(rhs);
  return *this;
}

template <typename T>
Quaternion<T>& Quaternion<T>::operator*=(const Quaternion<T>& rhs) {
  *this = *this * rhs;
  return *this;
}

template <typename T>
Quaternion<T>& Quaternion<T>::operator*=(const T& rhs) {
  Vector<T, Size>::operator*=(rhs);
  return *this;
}

template <typename T>
Quaternion<T>& Quaternion<T>::operator/=(const T& rhs) {
  Vector<T, Size>::operator/=(rhs);
  return *this;
}

template <typename T>
Quaternion<T> Quaternion<T>::operator+(const Quaternion<T>& rhs) const {
  Quaternion<T> result;
  static_cast<Vector<value_type, Size>&>(result) =
      Vector<T, Size>::operator+(rhs);
  return result;
}

template <typename T>
Quaternion<T> Quaternion<T>::operator+(const T& rhs) const {
  return Quaternion<T>(w() + rhs, x(), y(), z());
}

template <typename T>
Quaternion<T> Quaternion<T>::operator-(const Quaternion<T>& rhs) const {
  Quaternion<T> result;
  static_cast<Vector<value_type, Size>&>(result) =
      Vector<T, Size>::operator-(rhs);
  return result;
}

template <typename T>
Quaternion<T> Quaternion<T>::operator-(const T& rhs) const {
  return Quaternion<T>(w() - rhs, x(), y(), z());
}

template <typename T>
Vector<typename Quaternion<T>::value_type, 3> Quaternion<T>::operator*(
    const Vector<value_type, 3>& v) const {
  Quaternion<T> vAsQuat;
  vAsQuat[0] = 0;
  vAsQuat[1] = v[0];
  vAsQuat[2] = v[1];
  vAsQuat[3] = v[2];
  Quaternion<T> resultAsQuat = *this * vAsQuat * this->inverse();

  Vector<value_type, 3> result;
  result.x() = resultAsQuat.x();
  result.y() = resultAsQuat.y();
  result.z() = resultAsQuat.z();
  return result;
}

template <typename T>
Quaternion<T> Quaternion<T>::operator*(const Quaternion<T>& rhs) const {
  Quaternion<value_type> result;
  result.m_Vals[0] = m_Vals[0] * rhs[0] - m_Vals[1] * rhs[1] -
                     m_Vals[2] * rhs[2] - m_Vals[3] * rhs[3];
  result.m_Vals[1] = m_Vals[0] * rhs[1] + m_Vals[1] * rhs[0] +
                     m_Vals[2] * rhs[3] - m_Vals[3] * rhs[2];
  result.m_Vals[2] = m_Vals[0] * rhs[2] - m_Vals[1] * rhs[3] +
                     m_Vals[2] * rhs[0] + m_Vals[3] * rhs[1];
  result.m_Vals[3] = m_Vals[0] * rhs[3] + m_Vals[1] * rhs[2] -
                     m_Vals[2] * rhs[1] + m_Vals[3] * rhs[0];
  return result;
}

template <typename T>
Quaternion<T> Quaternion<T>::operator*(const T& rhs) const {
  Quaternion<T> result;
  static_cast<Vector<value_type, Size>&>(result) =
      Vector<T, Size>::operator*(rhs);
  return result;
}

template <typename T>
Quaternion<T> Quaternion<T>::operator/(const T& rhs) const {
  Quaternion<T> result;
  static_cast<Vector<value_type, Size>&>(result) =
      Vector<T, Size>::operator/(rhs);
  return result;
}

template <typename T>
T& Quaternion<T>::w() {
  return m_Vals[0];
}

template <typename T>
const T& Quaternion<T>::w() const {
  return m_Vals[0];
}

template <typename T>
T& Quaternion<T>::x() {
  return m_Vals[1];
}

template <typename T>
const T& Quaternion<T>::x() const {
  return m_Vals[1];
}

template <typename T>
T& Quaternion<T>::y() {
  return m_Vals[2];
}

template <typename T>
const T& Quaternion<T>::y() const {
  return m_Vals[2];
}

template <typename T>
T& Quaternion<T>::z() {
  return m_Vals[3];
}

template <typename T>
const T& Quaternion<T>::z() const {
  return m_Vals[3];
}

template <typename T>
typename Quaternion<T>::ImaginaryPart& Quaternion<T>::imaginaryVector() {
  return *reinterpret_cast<ImaginaryPart*>(&m_Vals[1]);
}

template <typename T>
const typename Quaternion<T>::ImaginaryPart& Quaternion<T>::imaginaryVector()
    const {
  return *reinterpret_cast<const ImaginaryPart*>(&m_Vals[1]);
}

template <typename T>
Quaternion<T> Quaternion<T>::imaginary() const {
  Quaternion<value_type> q(*this);
  q[0] = value_type(0);
  return q;
}

template <typename T>
Quaternion<T>& Quaternion<T>::conjugate() {
  m_Vals[1] = -m_Vals[1];
  m_Vals[2] = -m_Vals[2];
  m_Vals[3] = -m_Vals[3];
  return *this;
}

template <typename T>
Quaternion<T> Quaternion<T>::getConjugate() const {
  Quaternion<value_type> result;
  result.m_Vals[0] = m_Vals[0];
  result.m_Vals[1] = -m_Vals[1];
  result.m_Vals[2] = -m_Vals[2];
  result.m_Vals[3] = -m_Vals[3];
  return result;
}

template <typename T>
Quaternion<T> Quaternion<T>::normalized() const {
  Quaternion<T> result;
  static_cast<Vector<value_type, Size>&>(result) =
      Vector<T, Size>::normalized();
  return result;
}

template <typename T>
Quaternion<T>& Quaternion<T>::invert() {
  value_type squaredNorm = Vector<T, Size>::dotProduct();
  conjugate();
  return *this /= squaredNorm;
}

template <typename T>
Quaternion<T> Quaternion<T>::inverse() const {
  value_type squaredNorm = Vector<T, Size>::dotProduct();
  Quaternion<value_type> result = getConjugate();
  return result /= squaredNorm;
}

template <typename T>
Quaternion<T>& Quaternion<T>::exp() {
  ImaginaryPart& v = imaginaryVector();
  return exp(v.norm());
}

template <typename T>
Quaternion<T>& Quaternion<T>::exp(const T& normImaginary) {
  TEST_ALGORITHMS_CORE_ASSERT(equal(normImaginary, imaginaryVector().norm()));

  ImaginaryPart& imaginary = this->imaginaryVector();
  if (Math::equal(normImaginary, T(0))) {
    m_Vals[0] = Math::exp(w());
    m_Vals[1] = 0;
    m_Vals[2] = 0;
    m_Vals[3] = 0;
    return *this;
  } else {
    imaginary *= sin(normImaginary) / normImaginary;
    value_type w = m_Vals[0];
    m_Vals[0] = cos(normImaginary);
    return *this *= Math::exp(w);
  }
}

template <typename T>
Quaternion<T> Quaternion<T>::getExp() const {
  const ImaginaryPart& v = imaginaryVector();
  return getExp(v.norm());
}

template <typename T>
Quaternion<T> Quaternion<T>::getExp(const T& normImaginary) const {
  TEST_ALGORITHMS_CORE_ASSERT(equal(normImaginary, imaginaryVector().norm()));

  Quaternion<value_type> result;
  if (Math::equal(normImaginary, T(0))) {
    result[0] = exp(w());
    result[1] = 0;
    result[2] = 0;
    result[3] = 0;
    return result;
  } else {
    result[0] = cos(normImaginary);
    ImaginaryPart& v = result.imaginaryVector();
    v = imaginaryVector() * (sin(normImaginary) / normImaginary);
    return result *= Math::exp(w());
  }
}

template <typename T>
Quaternion<T>& Quaternion<T>::ln() {
  return ln(norm(), imaginaryVector().norm());
}

template <typename T>
Quaternion<T> Quaternion<T>::getLn() const {
  return getLn(norm(), imaginaryVector().norm());
}

template <typename T>
Quaternion<T>& Quaternion<T>::ln(const value_type& norm,
                                 const value_type& normImaginary) {
  TEST_ALGORITHMS_CORE_ASSERT(equal(norm, this->norm()) &&
                              equal(normImaginary, imaginaryVector().norm()));

  if (Math::equal(normImaginary, T(0))) {
    m_Vals[1] = 0;
    m_Vals[2] = 0;
    m_Vals[3] = 0;
  } else {
    ImaginaryPart& v = imaginaryVector();
    v *= acos(w() / norm) / normImaginary;
  }
  m_Vals[0] = Math::ln(norm);
  return *this;
}

template <typename T>
Quaternion<T> Quaternion<T>::getLn(const value_type& norm,
                                   const value_type& normImaginary) const {
  TEST_ALGORITHMS_CORE_ASSERT(equal(norm, this->norm()) &&
                              equal(normImaginary, imaginaryVector().norm()));

  Quaternion<value_type> result;
  if (Math::equal(normImaginary, T(0))) {
    result[1] = 0;
    result[2] = 0;
    result[3] = 0;
  } else {
    ImaginaryPart& v = result.imaginaryVector();
    v = imaginaryVector() * (acos(w() / norm) / normImaginary);
  }
  result[0] = Math::ln(norm);
  return result;
}

template <typename T>
Quaternion<T>& Quaternion<T>::pow(const T& x) {
  ln() *= x;
  return exp();
}

template <typename T>
Quaternion<T> Quaternion<T>::getPow(const T& x) const {
  Quaternion<T> result = getLn() * x;
  return result.exp();
}

template <typename T>
template <unsigned nMatrixRows, unsigned nMatrixCols>
void Quaternion<T>::toRotationMatrix(
    IN OUT Matrix<T, nMatrixRows, nMatrixCols>& m) const {
  static_assert(nMatrixRows > 2 && nMatrixCols > 2);

  value_type xx = this->x() * this->x();
  value_type yy = this->y() * this->y();
  value_type zz = this->z() * this->z();
  value_type xx2 = xx * 2;
  value_type yy2 = yy * 2;
  value_type zz2 = zz * 2;
  value_type wx = this->w() * this->x();
  value_type wy = this->w() * this->y();
  value_type wz = this->w() * this->z();
  value_type xy = this->x() * this->y();
  value_type xz = this->x() * this->z();
  value_type yz = this->y() * this->z();

  m[0][0] = 1 - yy2 - zz2;
  m[0][1] = 2 * (xy - wz);
  m[0][2] = 2 * (xz + wy);

  m[1][0] = 2 * (xy + wz);
  m[1][1] = 1 - xx2 - zz2;
  m[1][2] = 2 * (yz - wx);

  m[2][0] = 2 * (xz - wy);
  m[2][1] = 2 * (wx + yz);
  m[2][2] = 1 - xx2 - yy2;
}

template <typename T>
T Quaternion<T>::rotationAngle() const {
  return acos(w() / Vector<value_type, Size>::norm()) * 2;
}

}  // namespace Math
}  // namespace Test
