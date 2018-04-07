/**
 * @file init_rules_test.cpp
 * @author Marcus Edel
 *
 * Tests for the various weight initialize methods.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/math/random.hpp>

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include <mlpack/methods/ann/init_rules/kathirvalavakumar_subavathi_init.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/init_rules/oivs_init.hpp>
#include <mlpack/methods/ann/init_rules/orthogonal_init.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/init_rules/const_init.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/init_rules/glorot_init.hpp>
#include <mlpack/methods/ann/init_rules/variance_scaling_init.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(InitRulesTest);

/**
 * Test the RandomInitialization class with a constant value.
 */
BOOST_AUTO_TEST_CASE(ConstantInitTest)
{
  arma::mat weights;
  RandomInitialization constantInit(1, 1);
  constantInit.Initialize(weights, 100, 100);

  bool b = arma::all(arma::vectorise(weights) == 1);
  BOOST_REQUIRE_EQUAL(b, 1);
}

/**
 * Simple test of the OrthogonalInitialization class with two different
 * sizes.
 */
BOOST_AUTO_TEST_CASE(OrthogonalInitTest)
{
  arma::mat weights;
  OrthogonalInitialization orthogonalInit;
  orthogonalInit.Initialize(weights, 100, 200);

  arma::mat orthogonalWeights = arma::eye<arma::mat>(100, 100);
  weights *= weights.t();

  for (size_t i = 0; i < weights.n_rows; i++)
    for (size_t j = 0; j < weights.n_cols; j++)
      BOOST_REQUIRE_SMALL(weights.at(i, j) - orthogonalWeights.at(i, j), 1e-3);

  orthogonalInit.Initialize(weights, 200, 100);
  weights = weights.t() * weights;

  for (size_t i = 0; i < weights.n_rows; i++)
    for (size_t j = 0; j < weights.n_cols; j++)
      BOOST_REQUIRE_SMALL(weights.at(i, j) - orthogonalWeights.at(i, j), 1e-3);
}

/**
 * Test the OrthogonalInitialization class with a non default gain.
 */
BOOST_AUTO_TEST_CASE(OrthogonalInitGainTest)
{
  arma::mat weights;

  const double gain = 2;
  OrthogonalInitialization orthogonalInit(gain);
  orthogonalInit.Initialize(weights, 100, 200);

  arma::mat orthogonalWeights = arma::eye<arma::mat>(100, 100);
  orthogonalWeights *= (gain * gain);
  weights *= weights.t();

  for (size_t i = 0; i < weights.n_rows; i++)
    for (size_t j = 0; j < weights.n_cols; j++)
      BOOST_REQUIRE_SMALL(weights.at(i, j) - orthogonalWeights.at(i, j), 1e-3);
}

/**
 * Test the ConstInitialization class. If you think about it, it's kind of
 * ridiculous to test the const init rule. But at least we make sure it
 * builds without any problems.
 */
BOOST_AUTO_TEST_CASE(ConstInitTest)
{
  arma::mat weights;
  ConstInitialization zeroInit(0);
  zeroInit.Initialize(weights, 100, 100);

  bool b = arma::all(arma::vectorise(weights) == 0);
  BOOST_REQUIRE_EQUAL(b, 1);
}

/*
 * Simple test of the KathirvalavakumarSubavathiInitialization class with
 * two different sizes.
 */
BOOST_AUTO_TEST_CASE(KathirvalavakumarSubavathiInitTest)
{
  arma::mat data = arma::randu<arma::mat>(100, 1);

  arma::mat weights;
  arma::cube weights3d;

  KathirvalavakumarSubavathiInitialization kathirvalavakumarSubavathiInit(
      data, 1.5);

  kathirvalavakumarSubavathiInit.Initialize(weights, 100, 100);
  kathirvalavakumarSubavathiInit.Initialize(weights3d, 100, 100, 2);

  BOOST_REQUIRE_EQUAL(weights.n_rows, 100);
  BOOST_REQUIRE_EQUAL(weights.n_cols, 100);

  BOOST_REQUIRE_EQUAL(weights3d.n_rows, 100);
  BOOST_REQUIRE_EQUAL(weights3d.n_cols, 100);
  BOOST_REQUIRE_EQUAL(weights3d.n_slices, 2);
}

/**
 * Simple test of the NguyenWidrowInitialization class.
 */
BOOST_AUTO_TEST_CASE(NguyenWidrowInitTest)
{
  arma::mat weights;
  arma::cube weights3d;

  NguyenWidrowInitialization nguyenWidrowInit;

  nguyenWidrowInit.Initialize(weights, 100, 100);
  nguyenWidrowInit.Initialize(weights3d, 100, 100, 2);

  BOOST_REQUIRE_EQUAL(weights.n_rows, 100);
  BOOST_REQUIRE_EQUAL(weights.n_cols, 100);

  BOOST_REQUIRE_EQUAL(weights3d.n_rows, 100);
  BOOST_REQUIRE_EQUAL(weights3d.n_cols, 100);
  BOOST_REQUIRE_EQUAL(weights3d.n_slices, 2);
}

/**
 * Simple test of the OivsInitialization class with two different sizes.
 */
BOOST_AUTO_TEST_CASE(OivsInitTest)
{
  arma::mat weights;
  arma::cube weights3d;

  OivsInitialization<> oivsInit;

  oivsInit.Initialize(weights, 100, 100);
  oivsInit.Initialize(weights3d, 100, 100, 2);

  BOOST_REQUIRE_EQUAL(weights.n_rows, 100);
  BOOST_REQUIRE_EQUAL(weights.n_cols, 100);

  BOOST_REQUIRE_EQUAL(weights3d.n_rows, 100);
  BOOST_REQUIRE_EQUAL(weights3d.n_cols, 100);
  BOOST_REQUIRE_EQUAL(weights3d.n_slices, 2);
}

/**
 * Simple test of the GaussianInitialization class.
 */
BOOST_AUTO_TEST_CASE(GaussianInitTest)
{
  const size_t rows = 7;
  const size_t cols = 8;
  const size_t slices = 2;

  arma::mat weights;
  arma::cube weights3d;

  GaussianInitialization t(0, 0.2);

  t.Initialize(weights, rows, cols);
  t.Initialize(weights3d, rows, cols, slices);

  BOOST_REQUIRE_EQUAL(weights.n_rows, rows);
  BOOST_REQUIRE_EQUAL(weights.n_cols, cols);

  BOOST_REQUIRE_EQUAL(weights3d.n_rows, rows);
  BOOST_REQUIRE_EQUAL(weights3d.n_cols, cols);
  BOOST_REQUIRE_EQUAL(weights3d.n_slices, slices);
}

/**
 * Simple test of the VarianceScalingNormalInit class.
 * The test checks if the size of initialized weight matrices is correct.
 * Further it checks the size of initialized weight cubes is correct.
 * It also checks whether the values initialized make sense by verifying if
 * values lie within 3 standard deviations of the mean of normal distribution.
 */
BOOST_AUTO_TEST_CASE(VarianceScalingNormalInitTest)
{
  const size_t rows = 7;
  const size_t cols = 8;
  const size_t slices = 2;

  arma::mat weights;
  arma::cube weights3d;

  VarianceScalingNormalInit<> vsn;

  vsn.Initialize(weights, rows, cols);
  vsn.Initialize(weights3d, rows, cols, slices);


  // Check that size of initialized weights is equal to the input weights size

  BOOST_REQUIRE_EQUAL(weights.n_rows, rows);
  BOOST_REQUIRE_EQUAL(weights.n_cols, cols);

  BOOST_REQUIRE_EQUAL(weights3d.n_rows, rows);
  BOOST_REQUIRE_EQUAL(weights3d.n_cols, cols);
  BOOST_REQUIRE_EQUAL(weights3d.n_slices, slices);

  // Check that the initialized values lie in the required range

  double variance = (1.0 / ((rows + cols) / 2.0));
  double stddev = sqrt(variance);

  bool ge = arma::all(arma::vectorise(weights) >= (-3 * stddev));
  bool le = arma::all(arma::vectorise(weights) <= (3 * stddev));
  BOOST_REQUIRE_EQUAL(ge, 1);
  BOOST_REQUIRE_EQUAL(le, 1);

  bool ge3d = arma::all(arma::vectorise(weights3d) >= (-3 * stddev));
  bool le3d = arma::all(arma::vectorise(weights3d) <= (3 * stddev));
  BOOST_REQUIRE_EQUAL(ge3d, 1);
  BOOST_REQUIRE_EQUAL(le3d, 1);
}

/**
 * Simple test of the VarianceScalingUniformInit class.
 * The test checks if the size of initialized weight matrices is correct.
 * Further it checks the size of initialized weight cubes is correct.
 * It also checks whether the values initialized make sense by verifying if
 * values lie within the limits of the uniform initialization.
 */
BOOST_AUTO_TEST_CASE(VarianceScalingUniformInitTest)
{
  const size_t rows = 7;
  const size_t cols = 8;
  const size_t slices = 2;

  arma::mat weights;
  arma::cube weights3d;

  VarianceScalingUniformInit<> vsu;

  vsu.Initialize(weights, rows, cols);
  vsu.Initialize(weights3d, rows, cols, slices);

  // Check that size of initialized weights is equal to the input weights size

  BOOST_REQUIRE_EQUAL(weights.n_rows, rows);
  BOOST_REQUIRE_EQUAL(weights.n_cols, cols);

  BOOST_REQUIRE_EQUAL(weights3d.n_rows, rows);
  BOOST_REQUIRE_EQUAL(weights3d.n_cols, cols);
  BOOST_REQUIRE_EQUAL(weights3d.n_slices, slices);

  // Check that the initialized values lie in the required range

  double limit = sqrt(3.0 / ((rows + cols) / 2.0));

  bool ge = arma::all(arma::vectorise(weights) >= -limit);
  bool le = arma::all(arma::vectorise(weights) <= limit);
  BOOST_REQUIRE_EQUAL(ge, 1);
  BOOST_REQUIRE_EQUAL(le, 1);

  bool ge3d = arma::all(arma::vectorise(weights3d) >= -limit);
  bool le3d = arma::all(arma::vectorise(weights3d) <= limit);
  BOOST_REQUIRE_EQUAL(ge3d, 1);
  BOOST_REQUIRE_EQUAL(le3d, 1);
}

/**
 * Simple test of the NetworkInitialization class, we test it with every
 * implemented initialization rule and make sure the output is reasonable.
 */
BOOST_AUTO_TEST_CASE(NetworkInitTest)
{
  arma::mat input = arma::ones(5, 1);
  arma::mat response;
  NegativeLogLikelihood<> outputLayer;

  // Create a simple network and use the RandomInitialization rule to
  // initialize the network parameters.
  RandomInitialization randomInit(0.5, 0.5);

  FFN<NegativeLogLikelihood<>, RandomInitialization> randomModel(
      std::move(outputLayer), randomInit);
  randomModel.Add<IdentityLayer<> >();
  randomModel.Add<Linear<> >(5, 5);
  randomModel.Add<Linear<> >(5, 2);
  randomModel.Add<LogSoftMax<> >();
  randomModel.Predict(input, response);

  bool b = arma::all(arma::vectorise(randomModel.Parameters()) == 0.5);
  BOOST_REQUIRE_EQUAL(b, 1);
  BOOST_REQUIRE_EQUAL(randomModel.Parameters().n_elem, 42);

  // Create a simple network and use the OrthogonalInitialization rule to
  // initialize the network parameters.
  FFN<NegativeLogLikelihood<>, OrthogonalInitialization> orthogonalModel;
  orthogonalModel.Add<IdentityLayer<> >();
  orthogonalModel.Add<Linear<> >(5, 5);
  orthogonalModel.Add<Linear<> >(5, 2);
  orthogonalModel.Add<LogSoftMax<> >();
  orthogonalModel.Predict(input, response);

  BOOST_REQUIRE_EQUAL(orthogonalModel.Parameters().n_elem, 42);

  // Create a simple network and use the ZeroInitialization rule to
  // initialize the network parameters.
  FFN<NegativeLogLikelihood<>, ConstInitialization>
    zeroModel(NegativeLogLikelihood<>(), ConstInitialization(0));
  zeroModel.Add<IdentityLayer<> >();
  zeroModel.Add<Linear<> >(5, 5);
  zeroModel.Add<Linear<> >(5, 2);
  zeroModel.Add<LogSoftMax<> >();
  zeroModel.Predict(input, response);

  BOOST_REQUIRE_EQUAL(arma::accu(zeroModel.Parameters()), 0);
  BOOST_REQUIRE_EQUAL(zeroModel.Parameters().n_elem, 42);

  // Create a simple network and use the
  // KathirvalavakumarSubavathiInitialization rule to initialize the network
  // parameters.
  KathirvalavakumarSubavathiInitialization kathirvalavakumarSubavathiInit(
      input, 1.5);
  FFN<NegativeLogLikelihood<>, KathirvalavakumarSubavathiInitialization>
      ksModel(std::move(outputLayer), kathirvalavakumarSubavathiInit);
  ksModel.Add<IdentityLayer<> >();
  ksModel.Add<Linear<> >(5, 5);
  ksModel.Add<Linear<> >(5, 2);
  ksModel.Add<LogSoftMax<> >();
  ksModel.Predict(input, response);

  BOOST_REQUIRE_EQUAL(ksModel.Parameters().n_elem, 42);

  // Create a simple network and use the OivsInitialization rule to
  // initialize the network parameters.
  FFN<NegativeLogLikelihood<>, OivsInitialization<> > oivsModel;
  oivsModel.Add<IdentityLayer<> >();
  oivsModel.Add<Linear<> >(5, 5);
  oivsModel.Add<Linear<> >(5, 2);
  oivsModel.Add<LogSoftMax<> >();
  oivsModel.Predict(input, response);

  BOOST_REQUIRE_EQUAL(oivsModel.Parameters().n_elem, 42);

  // Create a simple network and use the GaussianInitialization rule to
  // initialize the network parameters.
  FFN<NegativeLogLikelihood<>, GaussianInitialization> gaussianModel;
  gaussianModel.Add<IdentityLayer<> >();
  gaussianModel.Add<Linear<> >(5, 5);
  gaussianModel.Add<Linear<> >(5, 2);
  gaussianModel.Add<LogSoftMax<> >();
  gaussianModel.Predict(input, response);

  BOOST_REQUIRE_EQUAL(gaussianModel.Parameters().n_elem, 42);

  // Create a simple network and use the Normal Variance Scaling rule to
  // initialize the network parameters.
  FFN<NegativeLogLikelihood<>, VarianceScalingNormalInit<> > vsnModel;
  vsnModel.Add<IdentityLayer<> >();
  vsnModel.Add<Linear<> >(5, 5);
  vsnModel.Add<Linear<> >(5, 2);
  vsnModel.Add<LogSoftMax<> >();
  vsnModel.Predict(input, response);

  BOOST_REQUIRE_EQUAL(vsnModel.Parameters().n_elem, 42);

  // Create a simple network and use the Uniform Variance Scaling rule to
  // initialize the network parameters.
  FFN<NegativeLogLikelihood<>, VarianceScalingUniformInit<> > vsuModel;
  vsuModel.Add<IdentityLayer<> >();
  vsuModel.Add<Linear<> >(5, 5);
  vsuModel.Add<Linear<> >(5, 2);
  vsuModel.Add<LogSoftMax<> >();
  vsuModel.Predict(input, response);

  BOOST_REQUIRE_EQUAL(vsuModel.Parameters().n_elem, 42);
}

/**
 * Simple test of the GlorotInitialization class for uniform distribution.
 */
BOOST_AUTO_TEST_CASE(GlorotInitUniformTest)
{
  arma::mat weights;
  arma::cube weights3d;

  XavierInitialization glorotInit;

  glorotInit.Initialize(weights, 100, 100);
  glorotInit.Initialize(weights3d, 100, 100, 2);

  BOOST_REQUIRE_EQUAL(weights.n_rows, 100);
  BOOST_REQUIRE_EQUAL(weights.n_cols, 100);

  BOOST_REQUIRE_EQUAL(weights3d.n_rows, 100);
  BOOST_REQUIRE_EQUAL(weights3d.n_cols, 100);
  BOOST_REQUIRE_EQUAL(weights3d.n_slices, 2);
}

/**
 * Simple test of the GlorotInitialization class for normal distribution.
 */
BOOST_AUTO_TEST_CASE(GlorotInitNormalTest)
{
  arma::mat weights;
  arma::cube weights3d;

  GlorotInitialization glorotInit;

  glorotInit.Initialize(weights, 100, 100);
  glorotInit.Initialize(weights3d, 100, 100, 2);

  BOOST_REQUIRE_EQUAL(weights.n_rows, 100);
  BOOST_REQUIRE_EQUAL(weights.n_cols, 100);

  BOOST_REQUIRE_EQUAL(weights3d.n_rows, 100);
  BOOST_REQUIRE_EQUAL(weights3d.n_cols, 100);
  BOOST_REQUIRE_EQUAL(weights3d.n_slices, 2);
}


BOOST_AUTO_TEST_SUITE_END();
