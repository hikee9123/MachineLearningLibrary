/**
 * @file linear_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the LinearLayer class also known as fully-connected layer or
 * affine transformation.
 */
#ifndef __MLPACK_METHODS_ANN_LAYER_LINEAR_LAYER_HPP
#define __MLPACK_METHODS_ANN_LAYER_LINEAR_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/optimizer/rmsprop.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the LinearLayer class. The LinearLayer class represents a
 * single layer of a neural network.
 *
 * @tparam OptimizerType Type of the optimizer used to update the weights.
 * @tparam WeightInitRule Rule used to initialize the weight matrix.
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    template<typename> class OptimizerType = mlpack::ann::RMSPROP,
    class WeightInitRule = NguyenWidrowInitialization,
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class LinearLayer
{
 public:
  /**
   * Create the LinearLayer object using the specified number of units.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   * @param WeightInitRule The weight initialization rule used to initialize the
   *        weight matrix.
   */
  LinearLayer(const size_t inSize,
              const size_t outSize,
              WeightInitRule weightInitRule = WeightInitRule()) :
      inSize(inSize),
      outSize(outSize)
  {
    weightInitRule.Initialize(weights, outSize, inSize);
  }
  



  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output)
  {
    output = weights * input;
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Cube<eT>& input, arma::Mat<eT>& output)
  {
    arma::Mat<eT> data(input.n_elem, 1);

    for (size_t s = 0, c = 0; s < input.n_slices / data.n_cols; s++)
    {
      for (size_t i = 0; i < data.n_cols; i++, c++)
      {
        data.col(i).subvec(s * input.n_rows * input.n_cols, (s + 1) *
            input.n_rows * input.n_cols - 1) = arma::vectorise(input.slice(c));
      }
    }

    output = weights * data;
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename InputType, typename eT>
  void Backward(const InputType& /* unused */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g)
  {
    g = weights.t() * gy;
  }

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param d The calculated error.
   * @param g The calculated gradient.
   */
  template<typename eT, typename GradientDataType>
  void Gradient(const arma::Mat<eT>& d, GradientDataType& g)
  {
    GradientDelta(inputParameter, d, g);
  }

  void UpdateOptimizer()
  {
	  optimizer.Update(Gradient());
	  }
  void Optimize()
  {
	  optimizer.Optimize(Weights());
	  }
  void ResetOptimizer()
	  {
	  optimizer.Reset();
  }

  //! Get the weights.
  OutputDataType& Weights() const { return weights; }
  //! Modify the weights.
  OutputDataType& Weights() { return weights; }

  //! Get the input parameter.
  InputDataType& InputParameter() const {return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const {return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType& Delta() const {return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the gradient.
  OutputDataType& Gradient() const {return gradient; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return gradient; }

 private:
   /*
   * Calculate the gradient using the output delta (3rd order tensor) and the
   * input activation (3rd order tensor).
   *
   * @param input The input parameter used for calculating the gradient.
   * @param d The output delta.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void GradientDelta(const arma::Cube<eT>& input,
                     const arma::Mat<eT>& d,
                     arma::Cube<eT>& g)
  {
    g = arma::Cube<eT>(weights.n_rows, weights.n_cols, 1);
    arma::Mat<eT> data = arma::Mat<eT>(d.n_cols,
        input.n_elem / d.n_cols);

    for (size_t s = 0, c = 0; s < input.n_slices /
        data.n_rows; s++)
    {
      for (size_t i = 0; i < data.n_rows; i++, c++)
      {
        data.row(i).subvec(s * input.n_rows *
            input.n_cols, (s + 1) *
            input.n_rows *
            input.n_cols - 1) = arma::vectorise(
                input.slice(c), 1);
      }
    }

    g.slice(0) = d * data / d.n_cols;
  }

  /*
   * Calculate the gradient (3rd order tensor) using the output delta
   * (dense matrix) and the input activation (dense matrix).
   *
   * @param input The input parameter used for calculating the gradient.
   * @param d The output delta.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void GradientDelta(const arma::Mat<eT>& /* input unused */,
                     const arma::Mat<eT>& d,
                     arma::Cube<eT>& g)
  {
    g = arma::Cube<eT>(weights.n_rows, weights.n_cols, 1);
    Gradient(d, g.slice(0));
  }

  /*
   * Calculate the gradient (dense matrix) using the output delta
   * (dense matrix) and the input activation (3rd order tensor).
   *
   * @param input The input parameter used for calculating the gradient.
   * @param d The output delta.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void GradientDelta(const arma::Cube<eT>& /* input unused */,
                     const arma::Mat<eT>& d,
                     arma::Mat<eT>& g)
  {
    arma::Cube<eT> grad = arma::Cube<eT>(weights.n_rows, weights.n_cols, 1);
    Gradient(d, grad);
    g = grad.slice(0);
  }

  /*
   * Calculate the gradient (dense matrix) using the output delta
   * (dense matrix) and the input activation (dense matrix).
   *
   * @param input The input parameter used for calculating the gradient.
   * @param d The output delta.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void GradientDelta(const arma::Mat<eT>& input,
                     const arma::Mat<eT>& d,
                     arma::Mat<eT>& g)
  {
    g = d * input.t();
  }

  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored pointer to the optimzer object.
  OptimizerType<OutputDataType>	  optimizer;

  //! Parameter that indicates if the class owns a optimizer object.

}; // class LinearLayer

/**
 * Linear Mapping layer to map between 3rd order tensors and dense matrices.
 */
template <
    template< typename> class OptimizerType = mlpack::ann::RMSPROP,
    class WeightInitRule = NguyenWidrowInitialization,
    typename InputDataType = arma::cube,
    typename OutputDataType = arma::mat
>
using LinearMappingLayer = LinearLayer<
    OptimizerType, WeightInitRule, InputDataType, OutputDataType>;

//! Layer traits for the linear layer.
template<
    template< typename> class OptimizerType,
    typename WeightInitRule,
    typename InputDataType,
    typename OutputDataType
>
class LayerTraits<LinearLayer<
    OptimizerType, WeightInitRule, InputDataType, OutputDataType> >
{
 public:
  static const bool IsBinary = false;
  static const bool IsOutputLayer = false;
  static const bool IsBiasLayer = false;
  static const bool IsLSTMLayer = false;
  static const bool IsConnection = true;
};

} // namespace ann
} // namespace mlpack

#endif
