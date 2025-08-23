#pragma once

#include <math.h>

#include "Quaternion.h"

namespace Test {
namespace Math {

template <typename T, unsigned nRows, unsigned nCols>
struct Matrix;

template <typename T>
struct Versor : public Quaternion<T> {
  typedef T value_type;
  typedef T& reference;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef Versor<value_type> Self;
  enum { Size = 4 };
  typedef Vector<T, Size - 1> ImaginaryPart;

  Versor();
  explicit Versor(const value_type& initVal);
  explicit Versor(const Quaternion<value_type>& q);
  Versor(const value_type& w, const value_type& x, const value_type& y,
         const value_type& z);

  /// The rotation angles are in radians.
  template <typename RotAxisNumT>
  explicit Versor(const Vector<RotAxisNumT, 3>& rotationAxis);
  template <typename RotAxisNumT>
  Versor(const Vector<RotAxisNumT, 3>& unitAxis, RotAxisNumT angle);

  template <unsigned nMatrixRows, unsigned nMatrixCols>
  Versor(const Matrix<value_type, nMatrixRows, nMatrixCols>& rotationMatrix);
  static Versor identity() { return Versor(1, 0, 0, 0); };
  using Quaternion<value_type>::operator-;
  Versor operator-() const;
  Versor<value_type>& operator*=(const Versor<value_type>& rhs);
  Versor<value_type> operator*(const Versor<value_type>& rhs) const;
  Vector<value_type, 3> operator*(const Vector<value_type, 3>& v) const;
  using Quaternion<value_type>::operator*;
  using Quaternion<value_type>::operator/;
  using Quaternion<T>::dotProduct;
  value_type dotProduct() const;
  value_type norm() const;
  void normalize();
  Versor normalized() const;
  void forceNormalize();
  Versor forceNormalized() const;
  Quaternion<value_type> imaginary() const;
  using Quaternion<T>::conjugate;
  Versor<value_type> getConjugate() const;
  Versor<value_type>& invert();
  Versor<value_type> inverse() const;
  Quaternion<value_type>& ln();
  Quaternion<value_type> getLn() const;
  Quaternion<value_type>& ln(const value_type& normImaginary);
  Quaternion<value_type> getLn(const value_type& normImaginary) const;
  Versor<value_type>& pow(const value_type& x);
  Versor<value_type> getPow(const value_type& x) const;
  Versor lerp(const Versor& v, value_type t) const;
  Versor slerp(const Versor& v, value_type t) const;
  // Works for any value of t
  Versor slerpGeneral(const Versor& v, value_type t) const;
  value_type rotationAngle() const;
  template <unsigned nMatrixRows, unsigned nMatrixCols>
  void toRotationMatrix(
      IN OUT Matrix<value_type, nMatrixRows, nMatrixCols>& m) const;
  Matrix<value_type, 3, 3> asRotationMatrix() const;
  void toRotationAxis(OUT Vector<value_type, 3>& rotAxis) const;
  Vector<value_type, 3> asRotationAxis() const;
  void axisAndAngle(OUT Vector<value_type, 3>& unitRotAxis,
                    OUT value_type& angle) const;
  template <typename VectorValT>
  void rotateVector(IN OUT Vector<VectorValT, 3>& v);

 protected:
  using Quaternion<value_type>::operator+=;
  using Quaternion<value_type>::operator-=;
  //    using Quaternion<value_type>::operator *=;
  using Quaternion<value_type>::operator/=;

  using Vector<T, Size>::m_Vals;
};

typedef Versor<float> VersorF;

}  // namespace Math
}  // namespace Test

#include "Matrix.h"

namespace Test {
namespace Math {

template <typename T>
Versor<T>::Versor() {}

template <typename T>
Versor<T>::Versor(const value_type& initVal) : Quaternion<T>(initVal) {}

template <typename T>
Versor<T>::Versor(const Quaternion<T>& q) {
  value_type norm = q.norm();
  for (unsigned i = 0; i < Size; ++i) {
    m_Vals[i] = q[i] / norm;
  }
}

template <typename T>
Versor<T>::Versor(const value_type& w, const value_type& x, const value_type& y,
                  const value_type& z) {
  m_Vals[0] = w;
  m_Vals[1] = x;
  m_Vals[2] = y;
  m_Vals[3] = z;
}

template <typename T>
template <typename RotAxisNumT>
Versor<T>::Versor(const Vector<RotAxisNumT, 3>& rotationAxis)
    : Quaternion<T>() {
  value_type angle = value_type(rotationAxis.norm());
  //    TEST_ALGORITHMS_CORE_ASSERT(T(-2*M_PI) < angle && angle < T(2*M_PI));
  angle = Math::mod(angle, T(2 * M_PI));

  if (!equal(angle, value_type(0))) {
    value_type angleSizeDiv2 = angle * T(0.5);
    value_type sin = Math::sin(angleSizeDiv2);
    value_type cos = Math::cos(angleSizeDiv2);
    m_Vals[0] = cos;
    T mult = sin / angle;
    m_Vals[1] = value_type(rotationAxis[0]) * mult;
    m_Vals[2] = value_type(rotationAxis[1]) * mult;
    m_Vals[3] = value_type(rotationAxis[2]) * mult;
  } else {
    m_Vals[1] = rotationAxis[0] * T(0.5);
    m_Vals[2] = rotationAxis[1] * T(0.5);
    m_Vals[3] = rotationAxis[2] * T(0.5);
    m_Vals[0] = Math::sqrt(1 - m_Vals[1] * m_Vals[1] + m_Vals[2] * m_Vals[2] +
                           m_Vals[3] * m_Vals[3]);
    if (angle > M_PI) {
      m_Vals[0] = -m_Vals[0];
    }
  }
}

template <typename T>
template <typename RotAxisNumT>
Versor<T>::Versor(const Vector<RotAxisNumT, 3>& unitAxis, RotAxisNumT angle) {
  //    TEST_ALGORITHMS_CORE_ASSERT(T(-2*M_PI) < angle && angle < T(2*M_PI));
  angle = Math::mod(angle, T(2 * M_PI));

  if (!equal(angle, value_type(0))) {
    value_type angleSizeDiv2 = angle * T(0.5);
    value_type sin = Math::sin(angleSizeDiv2);
    value_type cos = Math::cos(angleSizeDiv2);
    m_Vals[0] = cos;
    m_Vals[1] = sin * value_type(unitAxis[0]);
    m_Vals[2] = sin * value_type(unitAxis[1]);
    m_Vals[3] = sin * value_type(unitAxis[2]);
  } else {
    value_type absAngleSizeDiv2 = std::abs(angle) * T(0.5);
    m_Vals[1] = unitAxis[0] * absAngleSizeDiv2;
    m_Vals[2] = unitAxis[1] * absAngleSizeDiv2;
    m_Vals[3] = unitAxis[2] * absAngleSizeDiv2;
    m_Vals[0] = Math::sqrt(1 - m_Vals[1] * m_Vals[1] + m_Vals[2] * m_Vals[2] +
                           m_Vals[3] * m_Vals[3]);
    if (angle < T(0)) {
      m_Vals[0] = -m_Vals[0];
    }
  }
}

template <typename T>
template <unsigned nMatrixRows, unsigned nMatrixCols>
Versor<T>::Versor(const Matrix<T, nMatrixRows, nMatrixCols>& rotationMatrix) {
  static_assert(nMatrixRows > 2 && nMatrixCols > 2);

  value_type xyz = value_type(1) + rotationMatrix[0][0] - rotationMatrix[1][1] -
                   rotationMatrix[2][2];
  value_type yzx = value_type(1) - rotationMatrix[0][0] + rotationMatrix[1][1] -
                   rotationMatrix[2][2];
  value_type zxy = value_type(1) - rotationMatrix[0][0] - rotationMatrix[1][1] +
                   rotationMatrix[2][2];

  value_type r;
  if (xyz > yzx) {
    if (xyz > zxy)  // xyz is max
    {
      if (equal(xyz, value_type(0))) {
        goto identityQuat;
      }
      r = sqrt(xyz);
      this->x() = r / 2;
      r *= 2;
      this->w() = (rotationMatrix[2][1] - rotationMatrix[1][2]) / r;
      this->y() = (rotationMatrix[0][1] + rotationMatrix[1][0]) / r;
      this->z() = (rotationMatrix[2][0] + rotationMatrix[0][2]) / r;
    } else  // zxy is max
    {
      goto zxyIsMax;
    }
  } else {
    if (yzx > zxy)  // yzx is max
    {
      if (equal(yzx, value_type(0))) {
        goto identityQuat;
      }
      r = sqrt(yzx);
      this->y() = r / 2;
      r *= 2;
      this->w() = (rotationMatrix[0][2] - rotationMatrix[2][0]) / r;
      this->x() = (rotationMatrix[0][1] + rotationMatrix[1][0]) / r;
      this->z() = (rotationMatrix[1][2] + rotationMatrix[2][1]) / r;
    } else  // zxy is max
    {
      if (equal(zxy, value_type(0))) {
      identityQuat:
        this->w() = value_type(1);
        this->x() = value_type(0);
        this->y() = value_type(0);
        this->z() = value_type(0);
      } else {
      zxyIsMax:
        r = sqrt(zxy);
        this->z() = r / 2;
        r *= 2;
        this->w() = (rotationMatrix[1][0] - rotationMatrix[0][1]) / r;
        this->x() = (rotationMatrix[2][0] + rotationMatrix[0][2]) / r;
        this->y() = (rotationMatrix[1][2] + rotationMatrix[2][1]) / r;
      }
    }
  }

  //    value_type max = ww;
  //    if (xx > max) max = xx;
  //    if (yy > max) max = yy;
  //    if (zz > max) max = zz;
  //
  //    if (xx == max) {
  //        value_type w4 = sqrt(ww * 4.0);
  //        this->x() = w4 * 0.25;
  //        this->y() = (rotationMatrix[2][1] - rotationMatrix[1][2]) / w4;
  //        this->z() = (rotationMatrix[0][2] - rotationMatrix[2][0]) / w4;
  //        this->w() = (rotationMatrix[1][0] - rotationMatrix[0][1]) / w4;
  //    } else if (yy == max) {
  //        value_type y4 = sqrt(xx * 4.0);
  //        this->x() = (rotationMatrix[2][1] - rotationMatrix[1][2]) / y4;
  //        this->y() = y4 * 0.25;
  //        this->z() = (rotationMatrix[0][1] + rotationMatrix[1][0]) / y4;
  //        this->w() = (rotationMatrix[0][2] + rotationMatrix[2][0]) / y4;
  //    } else if (z == max) {
  //        value_type z4 = sqrt(yy * 4.0);
  //        this->x() = (rotationMatrix[0][2] - rotationMatrix[2][0]) / z4;
  //        this->y() = (rotationMatrix[0][1] + rotationMatrix[1][0]) / z4;
  //        this->z() =  z4 * 0.25;
  //        this->w() = (rotationMatrix[1][2] + rotationMatrix[2][1]) / z4;
  //    } else {
  //        value_type z4 = sqrt(zz * 4.0);
  //        this->x() = (rotationMatrix[1][0] - rotationMatrix[0][1]) / z4;
  //        this->y() = (rotationMatrix[0][2] + rotationMatrix[2][0]) / z4;
  //        this->z() = (rotationMatrix[1][2] + rotationMatrix[2][1]) / z4;
  //        this->w() =  z4 * 0.25;
  //    }

  //    value_type d0 = rotationMatrix[0][0];
  //    value_type d1 = rotationMatrix[1][1];
  //    value_type d2 = rotationMatrix[2][2];
  //    value_type ww = 1.0f + d0 + d1 + d2;
  //    value_type xx = 1.0f + d0 - d1 - d2;
  //    value_type yy = 1.0f - d0 + d1 - d2;
  //    value_type zz = 1.0f - d0 - d1 + d2;
  //
  //    value_type max = ww;
  //    if (xx > max) max = xx;
  //    if (yy > max) max = yy;
  //    if (zz > max) max = zz;
  //
  //    if (ww == max) {
  //        value_type w4 = sqrt(ww * 4.0);
  //        this->x() = w4 * 0.25;
  //        this->y() = (rotationMatrix[2][1] - rotationMatrix[1][2]) / w4;
  //        this->z() = (rotationMatrix[0][2] - rotationMatrix[2][0]) / w4;
  //        this->w() = (rotationMatrix[1][0] - rotationMatrix[0][1]) / w4;
  //    } else if (xx == max) {
  //        value_type x4 = sqrt(xx * 4.0);
  //        this->x() = (rotationMatrix[2][1] - rotationMatrix[1][2]) / x4;
  //        this->y() = x4 * 0.25;
  //        this->z() = (rotationMatrix[0][1] + rotationMatrix[1][0]) / x4;
  //        this->w() = (rotationMatrix[0][2] + rotationMatrix[2][0]) / x4;
  //    } else if (yy == max) {
  //        value_type y4 = sqrt(yy * 4.0);
  //        this->x() = (rotationMatrix[0][2] - rotationMatrix[2][0]) / y4;
  //        this->y() = (rotationMatrix[0][1] + rotationMatrix[1][0]) / y4;
  //        this->z() =  y4 * 0.25;
  //        this->w() = (rotationMatrix[1][2] + rotationMatrix[2][1]) / y4;
  //    } else {
  //        value_type z4 = sqrt(zz * 4.0);
  //        this->x() = (rotationMatrix[1][0] - rotationMatrix[0][1]) / z4;
  //        this->y() = (rotationMatrix[0][2] + rotationMatrix[2][0]) / z4;
  //        this->z() = (rotationMatrix[1][2] + rotationMatrix[2][1]) / z4;
  //        this->w() =  z4 * 0.25;
  //    }

  //    m_Vals[0] = ( rotationMatrix[0][0] + rotationMatrix[1][1] +
  //    rotationMatrix[2][2] +
  //        value_type(1)) / value_type(4);
  //    m_Vals[1] = ( rotationMatrix[0][0] - rotationMatrix[1][1] -
  //    rotationMatrix[2][2] +
  //        value_type(1)) / value_type(4);
  //    m_Vals[2] = (-rotationMatrix[0][0] + rotationMatrix[1][1] -
  //    rotationMatrix[2][2] +
  //        value_type(1)) / value_type(4);
  //    m_Vals[3] = (-rotationMatrix[0][0] - rotationMatrix[1][1] +
  //    rotationMatrix[2][2] +
  //        value_type(1)) / value_type(4);
  //    if(m_Vals[0] < 0.0f) m_Vals[0] = 0.0f;
  //    if(m_Vals[1] < 0.0f) m_Vals[1] = 0.0f;
  //    if(m_Vals[2] < 0.0f) m_Vals[2] = 0.0f;
  //    if(m_Vals[3] < 0.0f) m_Vals[3] = 0.0f;
  //    m_Vals[0] = sqrt(m_Vals[0]);
  //    m_Vals[1] = sqrt(m_Vals[1]);
  //    m_Vals[2] = sqrt(m_Vals[2]);
  //    m_Vals[3] = sqrt(m_Vals[3]);
  //    if(m_Vals[0] >= m_Vals[1] && m_Vals[0] >= m_Vals[2] && m_Vals[0] >=
  //    m_Vals[3]) {
  //        m_Vals[0] *= +1.0f;
  //        m_Vals[1] *= sign(rotationMatrix[2][1] - rotationMatrix[1][2]);
  //        m_Vals[2] *= sign(rotationMatrix[0][2] - rotationMatrix[2][0]);
  //        m_Vals[3] *= sign(rotationMatrix[1][0] - rotationMatrix[0][1]);
  //    } else if(m_Vals[1] >= m_Vals[0] && m_Vals[1] >= m_Vals[2] && m_Vals[1]
  //    >= m_Vals[3]) {
  //        m_Vals[0] *= sign(rotationMatrix[2][1] - rotationMatrix[1][2]);
  //        m_Vals[1] *= +1.0f;
  //        m_Vals[2] *= sign(rotationMatrix[1][0] + rotationMatrix[0][1]);
  //        m_Vals[3] *= sign(rotationMatrix[0][2] + rotationMatrix[2][0]);
  //    } else if(m_Vals[2] >= m_Vals[0] && m_Vals[2] >= m_Vals[1] && m_Vals[2]
  //    >= m_Vals[3]) {
  //        m_Vals[0] *= sign(rotationMatrix[0][2] - rotationMatrix[2][0]);
  //        m_Vals[1] *= sign(rotationMatrix[1][0] + rotationMatrix[0][1]);
  //        m_Vals[2] *= +1.0f;
  //        m_Vals[3] *= sign(rotationMatrix[2][1] + rotationMatrix[1][2]);
  //    } else if(m_Vals[3] >= m_Vals[0] && m_Vals[3] >= m_Vals[1] && m_Vals[3]
  //    >= m_Vals[2]) {
  //        m_Vals[0] *= sign(rotationMatrix[1][0] - rotationMatrix[0][1]);
  //        m_Vals[1] *= sign(rotationMatrix[2][0] + rotationMatrix[0][2]);
  //        m_Vals[2] *= sign(rotationMatrix[2][1] + rotationMatrix[1][2]);
  //        m_Vals[3] *= +1.0f;
  //    } else {
  //        TEST_ALGORITHMS_CORE_ASSERT(false); //Impossible. Coding error.
  //    }
  //    value_type r = Quaternion<value_type>::norm();
  //    m_Vals[0] /= r;
  //    m_Vals[1] /= r;
  //    m_Vals[2] /= r;
  //    m_Vals[3] /= r;
}

template <typename T>
Versor<T> Versor<T>::operator-() const {
  Versor<T> result;
  for (unsigned i = 0; i < Size; ++i) {
    result.m_Vals[i] = -m_Vals[i];
  }
  return result;
}

template <typename T>
Versor<T>& Versor<T>::operator*=(const Versor<T>& rhs) {
  Quaternion<T>::operator*=(rhs);
  return *this;
}

template <typename T>
Versor<T> Versor<T>::operator*(const Versor<T>& rhs) const {
  Versor<T> result;
  static_cast<Quaternion<T>&>(result) = Quaternion<T>::operator*(rhs);
  return result;
}

template <typename T>
Vector<typename Versor<T>::value_type, 3> Versor<T>::operator*(
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
T Versor<T>::dotProduct() const {
  return Quaternion<T>::dotProduct();
}

template <typename T>
T Versor<T>::norm() const {
  return Quaternion<T>::norm();
}

template <typename T>
void Versor<T>::normalize() {
  Quaternion<T>::normalize();
}

template <typename T>
Versor<T> Versor<T>::normalized() const {
  Versor<T> result(*this);
  result.normalize();
  return result;
}

template <typename T>
void Versor<T>::forceNormalize() {
  Quaternion<T>::normalize();
}

template <typename T>
Versor<T> Versor<T>::forceNormalized() const {
  Versor<T> result(*this);
  result.forceNormalize();
  return result;
}

template <typename T>
Versor<T>& Versor<T>::invert() {
  conjugate();
  return *this;
}

template <typename T>
Versor<T> Versor<T>::inverse() const {
  Versor<T> result;
  static_cast<Quaternion<T>&>(result) = Quaternion<T>::getConjugate();
  return result;
}

template <typename T>
Quaternion<T>& Versor<T>::ln() {
  return ln(Quaternion<T>::imaginary().norm());
}

template <typename T>
Quaternion<T> Versor<T>::getLn() const {
  return getLn(Quaternion<T>::imaginaryVector().norm());
}

template <typename T>
Quaternion<T>& Versor<T>::ln(const value_type& normImaginary) {
  TEST_ALGORITHMS_CORE_ASSERT(
      equal(normImaginary, Quaternion<T>::imaginaryVector().norm()));

  if (Math::equal(normImaginary, T(0))) {
    m_Vals[1] = 0;
    m_Vals[2] = 0;
    m_Vals[3] = 0;
  } else {
    ImaginaryPart& v = Quaternion<T>::imaginaryVector();
    v *= acos(std::max(T(-1), std::min(T(1), Quaternion<T>::w()))) /
         normImaginary;
  }
  m_Vals[0] = 0;
  return *this;
}

template <typename T>
Quaternion<T> Versor<T>::getLn(const value_type& normImaginary) const {
  TEST_ALGORITHMS_CORE_ASSERT(
      equal(normImaginary, Quaternion<T>::imaginaryVector().norm()));

  Quaternion<value_type> result;
  if (Math::equal(normImaginary, T(0))) {
    result[1] = 0;
    result[2] = 0;
    result[3] = 0;
  } else {
    ImaginaryPart& v = result.imaginaryVector();
    v = Quaternion<T>::imaginaryVector() *
        (acos(std::max(T(-1), std::min(T(1), Quaternion<T>::w()))) /
         normImaginary);
  }
  result[0] = 0;
  return result;
}

template <typename T>
Versor<T>& Versor<T>::pow(const T& x) {
  T angleDiv2 = Math::acos(std::max(std::min(m_Vals[0], T(1)), T(-1))) * x;
  T pureQuatPartNorm = Math::sqrt(
      m_Vals[1] * m_Vals[1] + m_Vals[2] * m_Vals[2] + m_Vals[3] * m_Vals[3]);
  if (equal(pureQuatPartNorm, T(0))) {
    m_Vals[0] = 1;
    m_Vals[1] = 0;
    m_Vals[2] = 0;
    m_Vals[3] = 0;
  } else {
    T coeff = Math::sin(angleDiv2) / pureQuatPartNorm;
    m_Vals[0] = Math::cos(angleDiv2);
    m_Vals[1] *= coeff;
    m_Vals[2] *= coeff;
    m_Vals[3] *= coeff;
  }
  return *this;
}

template <typename T>
Versor<T> Versor<T>::getPow(const T& x) const {
  Versor<T> result = *this;
  result.pow(x);
  return result;
}

template <typename T>
Versor<T> Versor<T>::lerp(const Versor& v, value_type t) const {
  return Versor(*this * (1 - t) + v * t);
}

template <typename T>
Versor<T> Versor<T>::slerp(const Versor& v, value_type t) const {
  TEST_ALGORITHMS_CORE_ASSERT(T(0) <= t && t <= T(1));

  Versor<T> result;

  T angle;
  T sinAngle;
  T dot = dotProduct(v);

  //    Versor<T> vv;
  //    //Invert for more than pi difference
  //    if (dot < 0)
  //    {
  //        vv = -v;
  //        dot = -dot;
  //    }
  //    else
  //    {
  //        vv = v;
  //    }

  T absDot = Math::abs(dot);
  if (absDot > T(1) || Math::equal(absDot, T(1))) {
    angle = T(0);
    sinAngle = T(0);
    result = Versor<T>(*this * (T(1) - t) + v * t);
    return result;
  } else {
    angle = Math::acos(dot);
    sinAngle = Math::sin(angle);
  }

  *reinterpret_cast<Math::Quaternion<T>*>(&result) =
      (*this * Math::sin(angle * (1 - t)) + v * Math::sin(angle * t)) /
      sinAngle;

  return result;
}

template <typename T>
Versor<T> Versor<T>::slerpGeneral(const Versor& v, value_type t) const {
  Versor<T> result;
  if (T(0) <= t && t <= T(1)) {
    result = slerp(v, t);
  } else {
    Versor diff = v * this->inverse();
    Vector<T, 3> axis;
    T angle;
    diff.axisAndAngle(axis, angle);
    angle *= t;

    diff = Versor(axis, angle);
    result = diff * *this;
  }
  return result;
}

template <typename T>
T Versor<T>::rotationAngle() const {
  return acos(std::max(std::min(Quaternion<T>::w(), T(1)), T(-1))) * 2;
}

template <typename T>
template <unsigned nMatrixRows, unsigned nMatrixCols>
void Versor<T>::toRotationMatrix(
    IN OUT Matrix<T, nMatrixRows, nMatrixCols>& m) const {
  static_assert(nMatrixRows > 2 && nMatrixCols > 2);

  value_type ww = this->w() * this->w();
  value_type xx = this->x() * this->x();
  value_type yy = this->y() * this->y();
  value_type zz = this->z() * this->z();

  m[0][0] = ww + xx - yy - zz;
  m[1][1] = ww - xx + yy - zz;
  m[2][2] = ww - xx - yy + zz;

  value_type xy = this->x() * this->y();
  value_type wz = this->w() * this->z();
  m[1][0] = T(2) * (xy + wz);
  m[0][1] = T(2) * (xy - wz);

  value_type xz = this->x() * this->z();
  value_type wy = this->w() * this->y();
  m[2][0] = T(2) * (xz - wy);
  m[0][2] = T(2) * (xz + wy);

  value_type yz = this->y() * this->z();
  value_type wx = this->w() * this->x();
  m[2][1] = T(2) * (yz + wx);
  m[1][2] = T(2) * (yz - wx);
}

template <typename T>
Matrix<typename Versor<T>::value_type, 3, 3> Versor<T>::asRotationMatrix()
    const {
  Matrix<value_type, 3, 3> m;
  toRotationMatrix(m);
  return m;
}

template <typename T>
void Versor<T>::toRotationAxis(
    OUT Vector<typename Versor<T>::value_type, 3>& rotAxis) const {
  if (!(Quaternion<T>::w() < T(-1) ||
        equal(Quaternion<T>::w(), value_type(-1)) ||
        T(1) < Quaternion<T>::w() ||
        equal(Quaternion<T>::w(), value_type(1)))) {
    value_type halfedAngle = acos(Quaternion<T>::w());
    value_type angle = halfedAngle * T(2);
    value_type sinHalfedAngle = sin(halfedAngle);
    value_type mult = angle / sinHalfedAngle;
    rotAxis[0] = mult * m_Vals[1];
    rotAxis[1] = mult * m_Vals[2];
    rotAxis[2] = mult * m_Vals[3];
  } else {
    rotAxis[0] = m_Vals[1] * T(0.5);
    rotAxis[1] = m_Vals[2] * T(0.5);
    rotAxis[2] = m_Vals[3] * T(0.5);
    if (this->w() < 0) {
      rotAxis[0] = -rotAxis[0];
      rotAxis[1] = -rotAxis[1];
      rotAxis[2] = -rotAxis[2];
    }
  }
}

template <typename T>
Vector<typename Versor<T>::value_type, 3> Versor<T>::asRotationAxis() const {
  Vector<value_type, 3> r;
  toRotationAxis(r);
  return r;
}

template <typename T>
void Versor<T>::axisAndAngle(OUT Vector<value_type, 3>& unitRotAxis,
                             OUT value_type& angle) const {
  if (!(Quaternion<T>::w() < T(-1) ||
        equal(Quaternion<T>::w(), value_type(-1)) ||
        T(1) < Quaternion<T>::w() ||
        equal(Quaternion<T>::w(), value_type(1)))) {
    value_type halfedAngle = acos(Quaternion<T>::w());
    angle = halfedAngle * 2;
    value_type sinHalfedAngle = sin(halfedAngle);
    unitRotAxis[0] = m_Vals[1] / sinHalfedAngle;
    unitRotAxis[1] = m_Vals[2] / sinHalfedAngle;
    unitRotAxis[2] = m_Vals[3] / sinHalfedAngle;
    unitRotAxis.normalize();
  } else {
    const ImaginaryPart& img = this->imaginaryVector();
    T imgNorm = img.norm();
    angle = imgNorm * T(2);
    if (angle == T(0)) {
      unitRotAxis[0] = T(1);
      unitRotAxis[1] = T(0);
      unitRotAxis[2] = T(0);
    } else {
      unitRotAxis[0] = m_Vals[1] / imgNorm;
      unitRotAxis[1] = m_Vals[2] / imgNorm;
      unitRotAxis[2] = m_Vals[3] / imgNorm;
    }
    if (this->w() < 0) {
      angle = T(2 * M_PI) - angle;
    }
  }
}

template <typename T>
template <typename VectorValT>
void Versor<T>::rotateVector(IN OUT Vector<VectorValT, 3>& v) {
  Quaternion<T> vAsQuat;
  vAsQuat[0] = 0;
  vAsQuat[1] = v[0];
  vAsQuat[2] = v[1];
  vAsQuat[3] = v[2];
  Quaternion<T> resultAsQuat = *this * vAsQuat * this->inverse();
  v.x() = resultAsQuat.x();
  v.y() = resultAsQuat.y();
  v.z() = resultAsQuat.z();
}

}  // namespace Math
}  // namespace Test

#undef QUAT_VERS
#undef QUAT_VERS_BASE
