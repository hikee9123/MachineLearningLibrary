/**
 * @file methods/kernel_svm/kernel_svm.hpp
 * @author Himanshu Pathak
 *
 * An implementation of Kernel SVM.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KERNEL_SVM_KERNEL_SVM_HPP
#define MLPACK_METHODS_KERNEL_SVM_KERNEL_SVM_HPP

#include <mlpack/prereqs.hpp>
#include <ensmallen.hpp>
#include <string>

#include "kernel_svm_function.hpp"

namespace mlpack {
namespace svm {

/**
 * The KernelSVM class implements an smo algorithm for support vector machine
 * model, and supports training with multiple non-linear and linear kenel.
 * The class supports different observation types via the MatType template
 * parameter; for instance, support vector classification can be performed
 * on sparse datasets by specifying arma::sp_mat as the MatType parameter.
 *
 *
 * @code
 * @article{Microsoft Research,
 *   author    = {John C. Platt},
 *   title     = {Sequential Minimal Optimization:A Fast 
                  Algorithm for Training Support Vector Machines},
 *   journal   = {Technical Report MSR-TR-98-14},
 *   year      = {1998},
 *   url       = {https://www.microsoft.com/en-us/research
                  /wp-content/uploads/2016/02/tr-98-14.pdf},
 * }
 * @endcode
 *
 *
 * An example on how to use the interface is shown below:
 *
 * @code
 * arma::mat train_data; // Training data matrix.
 * arma::Row<size_t> labels; // Labels associated with the data.
 * const size_t inputSize = 1000; // Size of input feature vector.
 *
 * // Train the model using default options.
 * KernelSVM<> svm(train_data, C, kernel_flag, max_iter, tol,
 *     kernel::Gaussian());
 *
 * arma::mat test_data;
 * arma::Row<size_t> predictions;
 * lsvm.Classify(test_data, predictions);
 * @endcode
 *
 * @tparam MatType Type of data matrix.
 * @tparam KernelType Type of kernel used with svm.
 */
template <typename MatType = arma::mat,
          typename KernelType = kernel::LinearKernel>
class KernelSVM
{
 public:
  /**
   * Construct the Kernel SVM class with the provided data and labels.
   *
   * @param data Input training features. Each column associate with one sample
   * @param labels Labels associated with the feature data.
   * @param regularization standard svm regularization parameter.
   * @param fitIntercept add intercept term or not.
   * @param numClass Number of classes for classification.
   * @param maxIter maximum number of iteration for training.
   * @param tol tolerance value.
   */
  KernelSVM(const MatType& data,
            const arma::Row<size_t>& labels,
            const double regularization = 1.0,
            const bool fitIntercept = false,
            const double numClass = 2,
            const size_t maxIter = 10,
            const double tol = 1e-3);

  /**
   * Initialize the Kernel SVM without performing training.  Default  Be sure 
   * to use Train() before calling
   * Classify() or ComputeAccuracy(), otherwise the results may be meaningless.
   *
   * @param regularization standard svm regularization parameter.
   * @param fitIntercept add intercept term or not.
   * @param numClass Number of classes for classification.
   * @param maxIter maximum number of iteration for training.
   */
  KernelSVM(const double regularization = 1.0,
            const bool fitIntercept = false,
            const double numClass = 2,
            const size_t maxIter = 10);

  /**
   * Classify the given points, returning the predicted labels for each point.
   * The function calculates the probabilities for every class, given a data
   * point. It then chooses the class which has the highest probability among
   * all.
   *
   * @param data Set of points to classify.
   * @param labels Predicted labels for each point.
   */
  void Classify(const MatType& data,
                arma::Row<size_t>& labels) const;

  /**
   * Classify the given points, returning class scores and predicted
   * class label for each point.
   * The function calculates the scores for every class, given a data
   * point. It then chooses the class which has the highest probability among
   * all.
   *
   * @param data Matrix of data points to be classified.
   * @param labels Predicted labels for each point.
   * @param scores Class probabilities for each point.
   */
  void Classify(const MatType& data,
                arma::Row<size_t>& labels,
                arma::mat& scores) const;

  /**
   * Classify the given points, returning class scores for each point.
   *
   * @param data Matrix of data points to be classified.
   * @param scores Class scores for each point.
   */
  void Classify(const MatType& data,
                arma::mat& scores) const;

  /**
   * Classify the given point. The predicted class label is returned.
   * The function calculates the scores for every class, given the point.
   * It then chooses the class which has the highest probability among all.
   *
   * @param point Point to be classified.
   * @return Predicted class label of the point.
   */
  template<typename VecType>
  size_t Classify(const VecType& point) const;

  /**
   * Computes accuracy of the learned model given the feature data and the
   * labels associated with each data point. Predictions are made using the
   * provided data and are compared with the actual labels.
   *
   * @param testData Matrix of data points using which predictions are made.
   * @param testLabels Vector of labels associated with the data.
   * @return Accuracy of the model.
   */
  double ComputeAccuracy(const MatType& testData,
                         const arma::Row<size_t>& testLabels) const;

  /**
   * Train the Kernel SVM with the given training data.
   *
   * @tparam OptimizerType Desired optimizer.
   * @param data Input training features. Each column associate with one sample.
   * @param labels Labels associated with the feature data.
   * @param maxIter maximum number of iteration for training.
   * @param tol tolerance value.
   * @return Objective value of the final point.
   */
  double Train(const MatType& data,
               const arma::Row<size_t>& labels,
               const size_t maxIter = 5,
               const double tol = 1e-3);

  //! Sets the number of classes.
  size_t& NumClasses() { return numClass; }
  //! Gets the number of classes.
  size_t NumClasses() const { return numClass; }

  //! Sets the regularization parameter.
  double& Lambda() { return regularization; }
  //! Gets the regularization parameter.
  double Lambda() const { return regularization; }

  //! Sets the margin between the correct class and all other classes.
  double& MaxIter() { return maxIter; }
  //! Gets the margin between the correct class and all other classes.
  double MaxIter() const { return maxIter; }

  //! Sets the intercept term flag.
  bool& FitIntercept() { return fitIntercept; }

  //! Gets the features size of the training data
  size_t FeatureSize() const
  { return features; }

  /**
   * Serialize the KernelSVM model.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(fitIntercept);
    ar & BOOST_SERIALIZATION_NVP(numClass);
    ar & BOOST_SERIALIZATION_NVP(maxIter);
  }

 private:
  //! Locally saved number of classes.
  double numClass;
  //! Locally stored maximum iteration.
  double maxIter;
  //! Locally saved number of classifier trained.
  double numClassifier;
  //! Locally saved features.
  size_t features;
  //! Locally saved classes of classifiers.
  arma::mat classesClassifier;
  //! Locally saved network of trained svms.
  std::vector<KernelSVMFunction<MatType , KernelType> > network;
  //! Locally saved standard svm regularization parameter.
  double regularization;
  //! Intercept term flag.
  bool fitIntercept;
};

} // namespace svm
} // namespace mlpack

// Include implementation.
#include "kernel_svm_impl.hpp"

#endif // MLPACK_METHODS_KERNEL_SVM_KERNEL_SVM_HPP
