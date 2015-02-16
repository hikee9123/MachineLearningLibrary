/**
 * @file triangular_kernel.hpp
 * @author Ryan Curtin
 *
 * Definition and implementation of the trivially simple triangular kernel.
 */
#ifndef __MLPACK_CORE_KERNELS_TRIANGULAR_KERNEL_HPP
#define __MLPACK_CORE_KERNELS_TRIANGULAR_KERNEL_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

namespace mlpack {
namespace kernel {

/**
 * The trivially simple triangular kernel, defined by
 *
 * @f[
 * K(x, y) = \max \{ 0, 1 - \frac{|| x - y ||_2}{b} \}
 * @f]
 *
 * where \f$ b \f$ is the bandwidth of the kernel.
 */
class TriangularKernel
{
 public:
  /**
   * Initialize the triangular kernel with the given bandwidth (default 1.0).
   *
   * @param bandwidth Bandwidth of the triangular kernel.
   */
  TriangularKernel(const double bandwidth = 1.0) : bandwidth(bandwidth) { }

  /**
   * Evaluate the triangular kernel for the two given vectors.
   *
   * @param a First vector.
   * @param b Second vector.
   */
  template<typename Vec1Type, typename Vec2Type>
  double Evaluate(const Vec1Type& a, const Vec2Type& b) const
  {
    return std::max(0.0, (1 - metric::EuclideanDistance::Evaluate(a, b) /
        bandwidth));
  }

  /**
   * Evaluate the triangular kernel given that the distance between the two
   * points is known.
   *
   * @param distance The distance between the two points.
   */
  double Evaluate(const double distance) const
  {
    return std::max(0.0, (1 - distance) / bandwidth);
  }
  
  /**
   * Evaluate the gradient of triangular kernel 
   * given that the distance between the two
   * points is known.
   *
   * @param distance The distance between the two points.
   */
  double Gradient(const double distance) const {
    if (distance < 1) {
      return -1.0 / bandwidth;
    } else if (distance > 1) {
      return 0;
    } else {
      return arma::datum::nan;
    }
  }

  //! Get the bandwidth of the kernel.
  double Bandwidth() const { return bandwidth; }
  //! Modify the bandwidth of the kernel.
  double& Bandwidth() { return bandwidth; }

  //! Return a string representation of the kernel.
  std::string ToString() const
  {
    std::ostringstream convert;
    convert << "TriangularKernel [" << this << "]" << std::endl;
    convert << "  Bandwidth: " << bandwidth << std::endl;
    return convert.str();
  }

 private:
  //! The bandwidth of the kernel.
  double bandwidth;
};

//! Kernel traits for the triangular kernel.
template<>
class KernelTraits<TriangularKernel>
{
 public:
  //! The triangular kernel is normalized: K(x, x) = 1 for all x.
  static const bool IsNormalized = true;
  //! The triangular kernel doesn't include a squared distance.
  static const bool UsesSquaredDistance = false;
};

}; // namespace kernel
}; // namespace mlpack

#endif
