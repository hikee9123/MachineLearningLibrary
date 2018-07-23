/**
 * @file lmnn_test.cpp
 * @author Marcus Edel
 * @author Ryan Curtin
 * @author Manish Kumar
 *
 * Unit tests for Large Margin Nearest Neighbors and related code (including
 * the constraints class).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/lmnn/lmnn.hpp>
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>
#include <mlpack/core/optimizers/bigbatch_sgd/bigbatch_sgd.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::metric;
using namespace mlpack::lmnn;
using namespace mlpack::optimization;


BOOST_AUTO_TEST_SUITE(LMNNTest);

//
// Tests for the Constraints.
//

/**
 * The target neighbors function should be correct.
 * point.
 */
BOOST_AUTO_TEST_CASE(LMNNTargetNeighborsTest)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset        = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  Constraints<> constraint(dataset, labels, 1);

  //! Store target neighbors of data points.
  arma::Mat<size_t> targetNeighbors =
      arma::Mat<size_t>(1, dataset.n_cols, arma::fill::zeros);

  constraint.TargetNeighbors(targetNeighbors, dataset, labels);

  BOOST_REQUIRE_EQUAL(targetNeighbors(0, 0), 1);
  BOOST_REQUIRE_EQUAL(targetNeighbors(0, 1), 0);
  BOOST_REQUIRE_EQUAL(targetNeighbors(0, 2), 1);
  BOOST_REQUIRE_EQUAL(targetNeighbors(0, 3), 4);
  BOOST_REQUIRE_EQUAL(targetNeighbors(0, 4), 3);
  BOOST_REQUIRE_EQUAL(targetNeighbors(0, 5), 4);
}

/**
 * The impostors function should be correct.
 */
BOOST_AUTO_TEST_CASE(LMNNImpostorsTest)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset        = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  Constraints<> constraint(dataset, labels, 1);

  //! Store impostors of data points.
  arma::Mat<size_t> impostors =
      arma::Mat<size_t>(1, dataset.n_cols, arma::fill::zeros);

  constraint.Impostors(impostors, dataset, labels);

  BOOST_REQUIRE_EQUAL(impostors(0, 0), 3);
  BOOST_REQUIRE_EQUAL(impostors(0, 1), 4);
  BOOST_REQUIRE_EQUAL(impostors(0, 2), 5);
  BOOST_REQUIRE_EQUAL(impostors(0, 3), 0);
  BOOST_REQUIRE_EQUAL(impostors(0, 4), 1);
  BOOST_REQUIRE_EQUAL(impostors(0, 5), 2);
}

//
// Tests for the LMNNFunction
//

/**
 * The LMNN function should return the identity matrix as its initial
 * point.
 */
BOOST_AUTO_TEST_CASE(LMNNInitialPointTest)
{
  // Cheap fake dataset.
  arma::mat dataset = arma::randu(5, 5);
  arma::Row<size_t> labels = "0 1 1 0 0";

  LMNNFunction<> lmnnfn(dataset, labels, 1, 0.5, 1);

  // Verify the initial point is the identity matrix.
  arma::mat initialPoint = lmnnfn.GetInitialPoint();
  for (int row = 0; row < 5; row++)
  {
    for (int col = 0; col < 5; col++)
    {
      if (row == col)
        BOOST_REQUIRE_CLOSE(initialPoint(row, col), 1.0, 1e-5);
      else
        BOOST_REQUIRE_SMALL(initialPoint(row, col), 1e-5);
    }
  }
}

/***
 * Ensure non-seprable objective function is right.
 */
BOOST_AUTO_TEST_CASE(LMNNInitialEvaluationTest)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset        = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  LMNNFunction<> lmnnfn(dataset, labels, 1, 0.6, 1);

  double objective = lmnnfn.Evaluate(arma::eye<arma::mat>(2, 2));

  // Result calculated by hand.
  BOOST_REQUIRE_CLOSE(objective, 9.456, 1e-5);
}

/**
 * Ensure non-seprable gradient function is right.
 */
BOOST_AUTO_TEST_CASE(LMNNInitialGradientTest)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset        = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  LMNNFunction<> lmnnfn(dataset, labels, 1, 0.6, 1);

  arma::mat gradient;
  arma::mat coordinates = arma::eye<arma::mat>(2, 2);
  lmnnfn.Gradient(coordinates, gradient);

  // Result calculated by hand.
  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.288, 1e-5);
  BOOST_REQUIRE_SMALL(gradient(1, 0), 1e-5);
  BOOST_REQUIRE_SMALL(gradient(0, 1), 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 12.0, 1e-5);
}

/***
 * Ensure non-seprable EvaluateWithGradient function is right.
 */
BOOST_AUTO_TEST_CASE(LMNNInitialEvaluateWithGradientTest)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset        = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  LMNNFunction<> lmnnfn(dataset, labels, 1, 0.6, 1);

  arma::mat gradient;
  arma::mat coordinates = arma::eye<arma::mat>(2, 2);
  double objective = lmnnfn.EvaluateWithGradient(coordinates, gradient);

  // Result calculated by hand.
  BOOST_REQUIRE_CLOSE(objective, 9.456, 1e-5);
  // Check Gradient
  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.288, 1e-5);
  BOOST_REQUIRE_SMALL(gradient(1, 0), 1e-5);
  BOOST_REQUIRE_SMALL(gradient(0, 1), 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 12.0, 1e-5);
}

/**
 * Ensure the separable objective function is right.
 */
BOOST_AUTO_TEST_CASE(LMNNSeparableObjectiveTest)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset        = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  LMNNFunction<> lmnnfn(dataset, labels, 1, 0.6, 1);

  // Result calculated by hand.
  arma::mat coordinates = arma::eye<arma::mat>(2, 2);
  BOOST_REQUIRE_CLOSE(lmnnfn.Evaluate(coordinates, 0, 1), 1.576, 1e-5);
  BOOST_REQUIRE_CLOSE(lmnnfn.Evaluate(coordinates, 1, 1), 1.576, 1e-5);
  BOOST_REQUIRE_CLOSE(lmnnfn.Evaluate(coordinates, 2, 1), 1.576, 1e-5);
  BOOST_REQUIRE_CLOSE(lmnnfn.Evaluate(coordinates, 3, 1), 1.576, 1e-5);
  BOOST_REQUIRE_CLOSE(lmnnfn.Evaluate(coordinates, 4, 1), 1.576, 1e-5);
  BOOST_REQUIRE_CLOSE(lmnnfn.Evaluate(coordinates, 5, 1), 1.576, 1e-5);
}

/**
 * Ensure the separable gradient is right.
 */
BOOST_AUTO_TEST_CASE(LMNNSeparableGradientTest)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset           = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  LMNNFunction<> lmnnfn(dataset, labels, 1, 0.6, 1);

  arma::mat coordinates = arma::eye<arma::mat>(2, 2);
  arma::mat gradient(2, 2);

  lmnnfn.Gradient(coordinates, 0, gradient, 1);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);

  lmnnfn.Gradient(coordinates, 1, gradient, 1);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);

  lmnnfn.Gradient(coordinates, 2, gradient, 1);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);

  lmnnfn.Gradient(coordinates, 3, gradient, 1);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);

  lmnnfn.Gradient(coordinates, 4, gradient, 1);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);

  lmnnfn.Gradient(coordinates, 5, gradient, 1);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);
}

/**
 * Ensure the separable EvaluateWithGradient function is right.
 */
BOOST_AUTO_TEST_CASE(LMNNSeparableEvaluateWithGradientTest)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset           = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  LMNNFunction<> lmnnfn(dataset, labels, 1, 0.6, 1);

  arma::mat coordinates = arma::eye<arma::mat>(2, 2);
  arma::mat gradient(2, 2);

  double objective = lmnnfn.EvaluateWithGradient(coordinates, 0, gradient, 1);

  BOOST_REQUIRE_CLOSE(objective, 1.576, 1e-5);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);

  objective = lmnnfn.EvaluateWithGradient(coordinates, 1, gradient, 1);

  BOOST_REQUIRE_CLOSE(objective, 1.576, 1e-5);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);

  objective = lmnnfn.EvaluateWithGradient(coordinates, 2, gradient, 1);

  BOOST_REQUIRE_CLOSE(objective, 1.576, 1e-5);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);

  objective = lmnnfn.EvaluateWithGradient(coordinates, 3, gradient, 1);

  BOOST_REQUIRE_CLOSE(objective, 1.576, 1e-5);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);

  objective = lmnnfn.EvaluateWithGradient(coordinates, 4, gradient, 1);

  BOOST_REQUIRE_CLOSE(objective, 1.576, 1e-5);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);

  objective = lmnnfn.EvaluateWithGradient(coordinates, 5, gradient, 1);

  BOOST_REQUIRE_CLOSE(objective, 1.576, 1e-5);

  BOOST_REQUIRE_CLOSE(gradient(0, 0), -0.048, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 0), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(gradient(1, 1), 2.0, 1e-5);
}

// Check that final objective value using SGD optimizer is optimal.
BOOST_AUTO_TEST_CASE(LMNNSGDSimpleDatasetTest)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset        = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  LMNN<> lmnn(dataset, labels, 1);

  arma::mat outputMatrix;
  lmnn.LearnDistance(outputMatrix);

  // Ensure that the objective function is better now.
  LMNNFunction<> lmnnfn(dataset, labels, 1, 0.6, 1);

  double initObj = lmnnfn.Evaluate(arma::eye<arma::mat>(2, 2));
  double finalObj = lmnnfn.Evaluate(outputMatrix);

  // finalObj must be less than initObj.
  BOOST_REQUIRE_LT(finalObj, initObj);
}

// Check that final objective value using L-BFGS optimizer is optimal.
BOOST_AUTO_TEST_CASE(LMNNLBFGSSimpleDatasetTest)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset        = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  LMNN<SquaredEuclideanDistance, L_BFGS> lmnn(dataset, labels, 1);

  arma::mat outputMatrix;
  lmnn.LearnDistance(outputMatrix);

  // Ensure that the objective function is better now.
  LMNNFunction<> lmnnfn(dataset, labels, 1, 0.6, 1);

  double initObj = lmnnfn.Evaluate(arma::eye<arma::mat>(2, 2));
  double finalObj = lmnnfn.Evaluate(outputMatrix);

  // finalObj must be less than initObj.
  BOOST_REQUIRE_LT(finalObj, initObj);
}

double KnnAccuracy(const arma::mat& dataset,
                   const arma::Row<size_t>& labels,
                   const size_t k)
{
  arma::Row<size_t> uniqueLabels = arma::unique(labels);

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  neighbor::KNN knn;

  knn.Train(dataset);
  knn.Search(k, neighbors, distances);

  // Keep count.
  size_t count = 0.0;

  for (size_t i = 0; i < dataset.n_cols; i++)
  {
    arma::vec Map;
    Map.zeros(uniqueLabels.n_cols);

    for (size_t j = 0; j < k; j++)
      Map(labels(neighbors(j, i))) +=
          1 / std::pow(distances(j, i) + 1, 2);

    size_t index = arma::conv_to<size_t>::from(arma::find(Map
        == arma::max(Map)));

    // Increase count if labels match.
    if (index == labels(i))
        count++;
  }

  // return accuracy.
  return ((double) count / dataset.n_cols) * 100;
}

// Check that final accuracy is greater than initial accuracy on
// simple dataset using AMSGrad optimizer.
BOOST_AUTO_TEST_CASE(LMNNAMSGradAccuracyTest)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset        = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  // Taking k = 3 as the case of k = 1 can be easily observed.
  double initAccuracy = KnnAccuracy(dataset, labels, 3);

  LMNN<> lmnn(dataset, labels, 1);

  arma::mat outputMatrix;
  lmnn.LearnDistance(outputMatrix);

  double finalAccuracy = KnnAccuracy(outputMatrix * dataset, labels, 3);

  // finalObj must be less than or equal to initObj.
  BOOST_REQUIRE_LT(initAccuracy, finalAccuracy);
}

// Check that final accuracy is greater than initial accuracy on
// simple dataset using BigBatchSGD optimizer.
BOOST_AUTO_TEST_CASE(LMNNBBSGDAccuracyTest)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset        = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  // Taking k = 3 as the case of k = 1 can be easily observed.
  double initAccuracy = KnnAccuracy(dataset, labels, 3);

  LMNN<LMetric<2>, BBS_BB> lmnn(dataset, labels, 1);

  arma::mat outputMatrix;
  lmnn.LearnDistance(outputMatrix);

  double finalAccuracy = KnnAccuracy(outputMatrix * dataset, labels, 3);

  // finalObj must be less than or equal to initObj.
  BOOST_REQUIRE(initAccuracy <= finalAccuracy);
}

// Check that final accuracy is greater than initial accuracy on
// simple dataset using L-BFGS optimizer.
BOOST_AUTO_TEST_CASE(LMNNLBFGSAccuracyTest)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset        = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  // Taking k = 3 as the case of k = 1 can be easily observed.
  double initAccuracy = KnnAccuracy(dataset, labels, 3);

  LMNN<LMetric<2>, L_BFGS> lmnn(dataset, labels, 1);

  arma::mat outputMatrix;
  lmnn.LearnDistance(outputMatrix);

  double finalAccuracy = KnnAccuracy(outputMatrix * dataset, labels, 3);

  // finalObj must be less than or equal to initObj.
  BOOST_REQUIRE_LT(initAccuracy, finalAccuracy);
}

// Check that accuracy while learning square distance matrix is same
// as when we are learnig low rank matrix.
BOOST_AUTO_TEST_CASE(LMNNLowRankAccuracyTest)
{
  arma::arma_rng::set_seed(2);

  arma::mat dataPart1;
  dataPart1.randn(5, 50);

  arma::Row<size_t> labelsPart1(50);
  labelsPart1.fill(0);

  arma::arma_rng::set_seed(50);

  arma::mat dataPart2;
  dataPart2.randn(5, 50);

  arma::Row<size_t> labelsPart2(50);
  labelsPart2.fill(1);

  // Generate ordering.
  arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0, 99, 100));

  // Generate datasets.
  arma::mat dataset = join_rows(dataPart1, dataPart2);
  dataset = dataset.cols(ordering);

  // Generate labels.
  arma::Row<size_t> labels = join_rows(labelsPart1, labelsPart2);
  labels = labels.cols(ordering);

  LMNN<> lmnn(dataset, labels, 1);

  // Learn a square matrix.
  arma::mat outputMatrix;
  lmnn.LearnDistance(outputMatrix);

  double Accuracy1 = KnnAccuracy(outputMatrix * dataset, labels, 1);

  // Learn a low rank matrix.
  outputMatrix = arma::randu(4, 5);
  lmnn.LearnDistance(outputMatrix);

  double Accuracy2 = KnnAccuracy(outputMatrix * dataset, labels, 1);

  // Keeping tolerance very high.
  // BOOST_REQUIRE_CLOSE(Accuracy1, Accuracy2, 2.0);
}

// Comprehensive gradient tests by Marcus Edel & Ryan Curtin.

// Simple numerical gradient checker.
template<class FunctionType>
double CheckGradient(FunctionType& function,
                     arma::mat& coordinates,
                     const double eps = 1e-7)
{
  // Get gradients for the current parameters.
  arma::mat orgGradient, gradient, estGradient;
  function.Gradient(coordinates, orgGradient);

  estGradient = arma::zeros(orgGradient.n_rows, orgGradient.n_cols);

  // Compute numeric approximations to gradient.
  for (size_t i = 0; i < orgGradient.n_elem; ++i)
  {
    double tmp = coordinates(i);

    // Perturb parameter with a positive constant and get costs.
    coordinates(i) += eps;
    double costPlus = function.Evaluate(coordinates);

    // Perturb parameter with a negative constant and get costs.
    coordinates(i) -= (2 * eps);
    double costMinus = function.Evaluate(coordinates);

    // Restore the parameter value.
    coordinates(i) = tmp;

    // Compute numerical gradients using the costs calculated above.
    estGradient(i) = (costPlus - costMinus) / (2 * eps);
  }

  // Estimate error of gradient.
  return arma::norm(orgGradient - estGradient) /
      arma::norm(orgGradient + estGradient);
}

BOOST_AUTO_TEST_CASE(LMNNFunctionGradientTest)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset        = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  LMNNFunction<> lmnnfn(dataset, labels, 1, 0.6, 1);

  // 10 trials with random positions.
  for (size_t i = 0; i < 10; ++i)
  {
    arma::mat coordinates(2, 2, arma::fill::randn);
    CheckGradient(lmnnfn, coordinates);
  }
}

BOOST_AUTO_TEST_CASE(LMNNFunctionGradientTest2)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset        = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  LMNNFunction<> lmnnfn(dataset, labels, 1, 0.6, 1);

  // 10 trials with random positions.
  for (size_t i = 0; i < 10; ++i)
  {
    arma::mat coordinates(2, 2, arma::fill::randu);
    CheckGradient(lmnnfn, coordinates);
  }
}

BOOST_AUTO_TEST_CASE(LMNNFunctionGradientTest3)
{
  arma::mat dataset;
  arma::Row<size_t> labels;
  data::Load("iris.csv", dataset);
  data::Load("iris_labels.txt", labels);

  LMNNFunction<> lmnnfn(dataset, labels, 1, 0.6, 1);

  // 10 trials with random positions.
  for (size_t i = 0; i < 10; ++i)
  {
    arma::mat coordinates(dataset.n_rows, dataset.n_rows, arma::fill::randn);
    CheckGradient(lmnnfn, coordinates);
  }
}

BOOST_AUTO_TEST_CASE(LMNNFunctionGradientTest4)
{
  arma::mat dataset;
  arma::Row<size_t> labels;
  data::Load("iris.csv", dataset);
  data::Load("iris_labels.txt", labels);

  LMNNFunction<> lmnnfn(dataset, labels, 1, 0.6, 1);

  // 10 trials with random positions.
  for (size_t i = 0; i < 10; ++i)
  {
    arma::mat coordinates(dataset.n_rows, dataset.n_rows, arma::fill::randu);
    CheckGradient(lmnnfn, coordinates);
  }
}

/**
 * Test the tree update function.
 */
BOOST_AUTO_TEST_CASE(UpdateTreeTest)
{
  // What I want to do is nearest neighbor search on a stretched dataset and on
  // an updated tree.  This takes a little bit of care because depending on
  // usage the NeighborSearch object may remap the indices of points.  So we
  // can't check the trees directly (easily at least) but we can check the
  // results.
  arma::mat dataset(10, 1000, arma::fill::randn);

  neighbor::KNN knn(dataset); // Make a copy of the data and build the tree.

  // Now stretch the dataset randomly.
  arma::mat transformation = arma::eye<arma::mat>(10, 10) + 0.01 *
      arma::randn<arma::mat>(10, 10);
  arma::mat transformedDataset = transformation * dataset;

  neighbor::KNN knn2(transformedDataset);

  // Next, update the reference tree we built with the first object.
  knn.ReferenceTree().Dataset() = transformation *
      knn.ReferenceTree().Dataset();
  UpdateTree(knn.ReferenceTree());

  // Run search for both, and ensure the results are the same.
  arma::mat distances1, distances2;
  arma::Mat<size_t> neighbors1, neighbors2;

  knn.Search(5, neighbors1, distances1);
  knn2.Search(5, neighbors2, distances2);

  CheckMatrices(neighbors1, neighbors2);
  CheckMatrices(distances1, distances2);
}

BOOST_AUTO_TEST_SUITE_END();
