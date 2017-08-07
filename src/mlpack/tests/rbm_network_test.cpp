/**
 * @file rbm_network_test.cpp
 * @author Kris Singh
 *
 * Tests the rbm Network
 *
 * digits dataset source:
 * @misc{Lichman:2013 ,
 * author = "M. Lichman",
 * year = "2013",
 * title = "{UCI} Machine Learning Repository",
 * url = "http://archive.ics.uci.edu/ml",
 * institution = "University of California,
 * Irvine, School of Information and Computer Sciences" } 
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/rbm/binary_rbm_policy.hpp>
#include <mlpack/methods/ann/rbm/spike_slab_rbm_policy.hpp>
#include <mlpack/methods/ann/rbm.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/core/optimizers/minibatch_sgd/minibatch_sgd.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(RbmNetworkTest);

template<typename MatType = arma::mat>
void BuildVanillaNetwork(MatType& trainData,
                         const size_t hiddenLayerSize)
{
  /*
   * Construct a feed forward network with trainData.n_rows input nodes,
   * hiddenLayerSize hidden nodes and trainLabels.n_rows output nodes. The
   * network structure looks like:
   *
   *  Visible       Hidden        
   *  Layer         Layer         
   * +-----+       +-----+       
   * |     |       |     |            
   * |     +<----->|     |
   * |     |       |     | 
   * +-----+       +-----+     
   *        
   */
  arma::mat output;
  BinaryRBMPolicy<> binary_rbm(trainData.n_rows, hiddenLayerSize);
  GaussianInitialization gaussian(0, 0.1);
  RBM<GaussianInitialization, BinaryRBMPolicy<>> model(trainData,
      gaussian, binary_rbm, 1,  true);

  model.Reset();
  // Set the parmaeters from a learned rbm sklearn random state 23
  model.Parameters() = arma::mat(
      "-0.23224054, -0.23000632, -0.25701271, -0.25122418, -0.20716651,"
      "-0.20962217, -0.59922456, -0.60003836, -0.6, -0.625, -0.475;");

  // Check free energy
  arma::vec freeEnergy = arma::mat(
      "-0.87523715, 0.50615066, 0.46923476, 1.21509084;");
  arma::vec calcultedFreeEnergy(4);
  calcultedFreeEnergy.zeros();
  for (size_t i = 0; i < trainData.n_cols; i++)
  {
    calcultedFreeEnergy(i) = model.FreeEnergy(std::move(trainData.col(i)));
  }

  for (size_t i = 0; i < freeEnergy.n_elem; i++)
    BOOST_REQUIRE_CLOSE(calcultedFreeEnergy(i), freeEnergy(i), 1e-5);
}
BOOST_AUTO_TEST_CASE(MiscTest)
{
  /**
   * Train and evaluate a vanilla network with the specified structure.
   */

  arma::mat X = arma::mat("0, 0, 0;"
                          "0, 1, 1;"
                          "1, 0, 1;"
                          "1, 1, 1;");
  X = X.t();
  BuildVanillaNetwork<>(X, 2);
}

BOOST_AUTO_TEST_CASE(ClassificationTest)
{
  // Normalised dataset
  int hiddenLayerSize = 100;
  size_t batchSize = 10;
  size_t numEpoches = 30;
  arma::mat trainData, testData, dataset;
  arma::mat trainLabelsTemp, testLabelsTemp;
  trainData.load("digits_train.arm");
  testData.load("digits_test.arm");
  trainLabelsTemp.load("digits_train_label.arm");
  testLabelsTemp.load("digits_test_label.arm");

  arma::Row<size_t> trainLabels = arma::zeros<arma::Row<size_t>>(1,
      trainLabelsTemp.n_cols);
  arma::Row<size_t> testLabels = arma::zeros<arma::Row<size_t>>(1,
      testLabelsTemp.n_cols);

  for (size_t i = 0; i < trainLabelsTemp.n_cols; ++i)
    trainLabels(i) = arma::as_scalar(trainLabelsTemp.col(i));

  for (size_t i = 0; i < testLabelsTemp.n_cols; ++i)
    testLabels(i) = arma::as_scalar(testLabelsTemp.col(i));

  arma::mat output, XRbm(hiddenLayerSize, trainData.n_cols),
      YRbm(hiddenLayerSize, testLabels.n_cols);

  XRbm.zeros();
  YRbm.zeros();

  BinaryRBMPolicy<> binary_rbm(trainData.n_rows, hiddenLayerSize);
  GaussianInitialization gaussian(0, 0.1);
  RBM<GaussianInitialization, BinaryRBMPolicy<>> model(trainData,
      gaussian, binary_rbm, 1,  1, true, false);

  size_t numRBMIterations = trainData.n_cols * numEpoches;
  numRBMIterations /= batchSize;

  MiniBatchSGD msgd(batchSize, 0.06, numRBMIterations, 0, true);
  model.Reset();
  model.Policy().VisibleBias().ones();
  model.Policy().HiddenBias().ones();
  // test the reset function
  model.Train(trainData, msgd);

  for (size_t i = 0; i < trainData.n_cols; i++)
  {
    model.Policy().HiddenMean(std::move(trainData.col(i)),
        std::move(output));
    XRbm.col(i) = output;
  }

  for (size_t i = 0; i < testData.n_cols; i++)
  {
    model.Policy().HiddenMean(std::move(testData.col(i)),
      std::move(output));
    YRbm.col(i) = output;
  }
  const size_t numClasses = 10; // Number of classes.
  const size_t numBasis = 5; // Parameter required for L-BFGS algorithm.
  const size_t numIterations = 100; // Maximum number of iterations.

  // Use an instantiated optimizer for the training.
  L_BFGS optimizer(numBasis, numIterations);
  SoftmaxRegression regressor2(trainData, trainLabels,
      numClasses, 0.001, false, optimizer);

  arma::Row<size_t> predictions1, predictions2;
  // Vectors to store predictions in.

  double classificationAccuray = regressor2.ComputeAccuracy(testData,
   testLabels);
  std::cout << "Softmax Accuracy = " << classificationAccuray << std::endl;

  L_BFGS optimizer1(numBasis, numIterations);
  SoftmaxRegression regressor1(XRbm, trainLabels, numClasses,
        0.001, false, optimizer1);
  double classificationAccuray1 = regressor1.ComputeAccuracy(YRbm, testLabels);
  std::cout << "RBM Accuracy = " <<classificationAccuray1 << std::endl;
  BOOST_REQUIRE_GE(classificationAccuray1, classificationAccuray);
}

BOOST_AUTO_TEST_CASE(ssRBMClassificationTest)
{
  size_t batchSize = 10;
  size_t numEpoches = 30;
  int hiddenLayerSize = 100;
  double radius = 0;
  double tempRadius = 0;
  arma::mat trainData, testData, dataset;
  arma::mat trainLabelsTemp, testLabelsTemp;
  trainData.load("digits_train.arm");
  testData.load("digits_test.arm");
  trainLabelsTemp.load("digits_train_label.arm");
  testLabelsTemp.load("digits_test_label.arm");
  GaussianInitialization gaussian(0, 0.1);

  arma::Row<size_t> trainLabels = arma::zeros<arma::Row<size_t>>(1,
      trainLabelsTemp.n_cols);
  arma::Row<size_t> testLabels = arma::zeros<arma::Row<size_t>>(1,
      testLabelsTemp.n_cols);

  for (size_t i = 0; i < trainLabelsTemp.n_cols; ++i)
    trainLabels(i) = arma::as_scalar(trainLabelsTemp.col(i));

  for (size_t i = 0; i < testLabelsTemp.n_cols; ++i)
    testLabels(i) = arma::as_scalar(testLabelsTemp.col(i));

  for (size_t i = 0; i < trainData.n_cols; i++)
  {
    tempRadius = arma::norm(trainData.col(i));
    if (radius < tempRadius)
      radius = tempRadius;
  }

  size_t poolSize = 1;

  arma::mat output;
  arma::mat XRbm(hiddenLayerSize + poolSize * hiddenLayerSize,
      trainData.n_cols);
  arma::mat YRbm(hiddenLayerSize + poolSize * hiddenLayerSize,
      testLabels.n_cols);

  XRbm.zeros();
  YRbm.zeros();
  double slabPenalty = 2;

  SpikeSlabRBMPolicy<> ss_rbm(trainData.n_rows, hiddenLayerSize, poolSize,
      slabPenalty, radius);
  RBM<GaussianInitialization, SpikeSlabRBMPolicy<>> modelssRBM(trainData,
      gaussian, ss_rbm, 1, 1, true, false);

  size_t numRBMIterations = trainData.n_cols * numEpoches;
  numRBMIterations /= batchSize;

  MiniBatchSGD msgd(batchSize, 0.06, numRBMIterations, 0, true);
  modelssRBM.Reset();
  modelssRBM.Policy().VisiblePenalty().fill(0);
  modelssRBM.Policy().SpikeBias().fill(15);
  modelssRBM.Train(trainData, msgd);
  for (size_t i = 0; i < trainData.n_cols; i++)
  {
    modelssRBM.Policy().HiddenMean(std::move(trainData.col(i)),
        std::move(output));
    XRbm.col(i) = output;
  }

  for (size_t i = 0; i < testData.n_cols; i++)
  {
    modelssRBM.Policy().HiddenMean(std::move(testData.col(i)),
      std::move(output));
    YRbm.col(i) = output;
  }
  const size_t numClasses = 10; // Number of classes.
  const size_t numBasis = 5; // Parameter required for L-BFGS algorithm.
  const size_t numIterations = 100; // Maximum number of iterations.

  // Use an instantiated optimizer for the training.
  L_BFGS optimizer(numBasis, numIterations);
  SoftmaxRegression regressor2(trainData, trainLabels,
      numClasses, 0.001, false, optimizer);

  arma::Row<size_t> predictions1, predictions2;
  // Vectors to store predictions in.

  double classificationAccuray = regressor2.ComputeAccuracy(testData,
   testLabels);
  std::cout << "Softmax Accuracy = " << classificationAccuray << std::endl;

  L_BFGS optimizer1(numBasis, numIterations);
  SoftmaxRegression regressor1(XRbm, trainLabels, numClasses,
        0.001, false, optimizer1);
  double classificationAccuray1 = regressor1.ComputeAccuracy(YRbm, testLabels);
  std::cout << "ssRBM Accuracy = " <<classificationAccuray1 << std::endl;

  BOOST_REQUIRE_GE(classificationAccuray1, 70);
}

template<typename MatType = arma::mat>
void FloatBuildVanillaNetwork(MatType& trainData,
                         const size_t hiddenLayerSize)
{
  arma::Mat<float> output;
  BinaryRBMPolicy<arma::Mat<float>> binary_rbm(trainData.n_rows, hiddenLayerSize);
  GaussianInitialization gaussian(0, 0.1);
  RBM<GaussianInitialization, BinaryRBMPolicy<arma::Mat<float>>> model(trainData,
      gaussian, binary_rbm, 1,  true);

  model.Reset();
  // Set the parmaeters from a learned rbm sklearn random state 23
  model.Parameters() = arma::Mat<float>(
      "-0.23224054, -0.23000632, -0.25701271, -0.25122418, -0.20716651,"
      "-0.20962217, -0.59922456, -0.60003836, -0.6, -0.625, -0.475;");

  // Check free energy
  arma::Col<float> freeEnergy = arma::Mat<float>(
      "-0.87523715, 0.50615066, 0.46923476, 1.21509084;");
  arma::vec calcultedFreeEnergy(4);
  calcultedFreeEnergy.zeros();
  for (size_t i = 0; i < trainData.n_cols; i++)
  {
    calcultedFreeEnergy(i) = model.FreeEnergy(std::move(trainData.col(i)));
  }

  for (size_t i = 0; i < freeEnergy.n_elem; i++)
    BOOST_REQUIRE_CLOSE(calcultedFreeEnergy(i), freeEnergy(i), 1e-5);

}

BOOST_AUTO_TEST_CASE(FloatTest)
{
  /**
   * Train and evaluate a vanilla network with the specified structure.
   */

  arma::Mat<float> X = arma::Mat<float>("0.0, 0.0, 0.0;"
                          "0.0, 1.0, 1.0;"
                          "1.0, 0.0, 1.0;"
                          "1.0, 1.0, 1.0;");
  X = X.t();
  BuildVanillaNetwork<arma::Mat<float>>(X, 2);
}

BOOST_AUTO_TEST_SUITE_END();
