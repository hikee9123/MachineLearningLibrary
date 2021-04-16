/**
 * @file methods/rvm_regression_model_impl.hpp
 * @author Clement Mercier
 *
 * A serializable RVM model used by the main program.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RVM_REGRESSION_MODEL_IMPL_HPP
#define MLPACK_METHODS_RVM_REGRESSION_MODEL_IMPL_HPP

#include "rvm_regression_model.hpp"

RVMRegressionModel::RVMRegressionModel(const std::string kernelType,
				       const bool centerData,
				       const bool scaleData,
				       const double bandwidth,
				       const double offset,
				       const double kernel_scale,
				       const double degree)
{
  if (kernelType == "linear")
  {    
    LinearKernel kernel;
    rVariant = new RVMRegression<LinearKernel>(
        kernel, centerData, scaleData, false);
  }

  else if (kernelType == "gaussian")
  {
    GaussianKernel kernel(bandwidth);
    rVariant = new RVMRegression<GaussianKernel>(
        kernel, centerData, scaleData, false);
  }

  else if (kernelType == "laplacian")
  {
    LaplacianKernel kernel(bandwidth);
    rVariant = new RVMRegression<LaplacianKernel>(
        kernel, centerData, scaleData, false);
  }

  else if (kernelType == "epanechnikov")
  {
    EpanechnikovKernel kernel(bandwidth);
    rVariant = new RVMRegression<EpanechnikovKernel>(
        kernel, centerData, scaleData, false);
  }

  else if (kernelType == "spherical")
  {
    SphericalKernel kernel(bandwidth);
    rVariant = new RVMRegression<SphericalKernel>(
        kernel, centerData, scaleData, false);
  }

  else if (kernelType == "hyptan")
  {
    std::cout << "hyptan kernel unavailable as long as its Evualuate "
        "method is not const." << std::endl;
    exit(0);
    // HyperbolicTangentKernel kernel(kernel_scale, offset);
    // rVariant = new RVMRegression<HyperbolicTangentKernel>(
    //     kernel, centerData, scaleData, false);
  }

  else if (kernelType == "polynomial")
  {
    PolynomialKernel kernel(degree, offset);
    rVariant = new RVMRegression<PolynomialKernel>(
        kernel, centerData, scaleData, false);
  }

  else if (kernelType == "cosine")
  {
    CosineDistance kernel;
    rVariant = new RVMRegression<CosineDistance>(
        kernel, centerData, scaleData, false);
  }

  else if (kernelType == "ard")
  {
    LinearKernel kernel;
    rVariant = new RVMRegression<LinearKernel>(
        kernel, centerData, scaleData, true);
  }

  else
  {
    std::cout << "ard, linear or gaussian only." << std::endl;
  }
};

RVMRegressionModel::~RVMRegressionModel()
{
  boost::apply_visitor(DeleteVisitor{}, rVariant);
}

void RVMRegressionModel::Predict(const arma::mat& matX,
				 arma::rowvec& predictions,
				 arma::rowvec& std)
{
  PredictStdVisitor predict(matX, predictions, std);
  boost::apply_visitor(predict, rVariant);
}


void RVMRegressionModel::Train(const arma::mat& matX,
			       const arma::rowvec& responses)
{
  TrainVisitor train(matX, responses);
  boost::apply_visitor(train, rVariant);
}

void RVMRegressionModel::Predict(const arma::mat& matX,
				 arma::rowvec& predictions)
{
  PredictVisitor predict(matX, predictions);
  boost::apply_visitor(predict, rVariant);
}

template <typename KernelType>
const RVMRegression<KernelType>* RVMRegressionModel::RVMPtr() const
{
  void* pointer = boost::apply_visitor(GetValueVisitor(), rVariant);
  return (RVMRegression<KernelType>*) pointer;
}


template<typename Archive>
void RVMRegressionModel::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_VARIANT_POINTER(rVariant));
}

TrainVisitor::TrainVisitor(const arma::mat& matX,
			   const arma::rowvec& responses) :
    matX(matX),
    responses(responses)
{ }

template <typename T>
void TrainVisitor::operator()(T* t) const
{
  t->Train(matX, responses);
}

PredictVisitor::PredictVisitor(const arma::mat& matX,
			       arma::rowvec& predictions) :
  matX(matX),
  predictions(predictions)
{ /* Nothing to do */ }

template <typename T>
void PredictVisitor::operator()(T* t) const
{
    t->Predict(matX, predictions);
}

PredictStdVisitor::PredictStdVisitor(const arma::mat& matX,
				  arma::rowvec& predictions,
				  arma::rowvec& std) :
  matX(matX),
  predictions(predictions),
  std(std)
  { /* Nothing to do */ }

template <typename T>
void PredictStdVisitor::operator()(T* t) const
{
  t->Predict(matX, predictions, std);
}

template <typename T>
void* GetValueVisitor::operator()(T *t) const
{
  if (!t)
    throw std::runtime_error("no rvm model initialized");

  return (void*) t;
}

#endif

