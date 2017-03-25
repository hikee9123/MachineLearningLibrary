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
#include <mlpack/core/math/shapiro_wilk_test.hpp>
#include <mlpack/core/math/random.hpp>

#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/init_rules/kathirvalavakumar_subavathi_init.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/init_rules/oivs_init.hpp>
#include <mlpack/methods/ann/init_rules/orthogonal_init.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/init_rules/zero_init.hpp>
#include <mlpack/methods/ann/init_rules/xavier_init.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::math;

BOOST_AUTO_TEST_SUITE(InitRulesTest);

// Test the RandomInitialization class with a constant value.
BOOST_AUTO_TEST_CASE(ConstantInitTest)
{
  arma::mat weights;
  RandomInitialization constantInit(1, 1);
  constantInit.Initialize(weights, 100, 100);

  bool b = arma::all(arma::vectorise(weights) == 1);
  BOOST_REQUIRE_EQUAL(b, 1);
}

// Test the OrthogonalInitialization class.
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

// Test the OrthogonalInitialization class with a non default gain.
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

// Test the ZeroInitialization class. If you think about it, it's kind of
// ridiculous to test the zero init rule. But at least we make sure it
// builds without any problems.
BOOST_AUTO_TEST_CASE(ZeroInitTest)
{
  arma::mat weights;
  ZeroInitialization zeroInit;
  zeroInit.Initialize(weights, 100, 100);

  bool b = arma::all(arma::vectorise(weights) == 0);
  BOOST_REQUIRE_EQUAL(b, 1);
}

// Test the KathirvalavakumarSubavathiInitialization class.
BOOST_AUTO_TEST_CASE(KathirvalavakumarSubavathiInitTest)
{
  arma::mat data = arma::randu<arma::mat>(100, 1);

  arma::mat weights;
  KathirvalavakumarSubavathiInitialization kathirvalavakumarSubavathiInit(
      data, 1.5);
  kathirvalavakumarSubavathiInit.Initialize(weights, 100, 100);

  BOOST_REQUIRE_EQUAL(1, 1);
}

// Test the NguyenWidrowInitialization class.
BOOST_AUTO_TEST_CASE(NguyenWidrowInitTest)
{
  arma::mat weights;
  NguyenWidrowInitialization nguyenWidrowInit;
  nguyenWidrowInit.Initialize(weights, 100, 100);

  BOOST_REQUIRE_EQUAL(1, 1);
}

// Test the OivsInitialization class.
BOOST_AUTO_TEST_CASE(OivsInitTest)
{
  arma::mat weights;
  OivsInitialization<> oivsInit;
  oivsInit.Initialize(weights, 100, 100);

  BOOST_REQUIRE_EQUAL(1, 1);
}

BOOST_AUTO_TEST_CASE(XavierInitTest)
{
  arma::mat weights;
  RandomSeed(21);
  XavierInit<XavierUniform> xavierUniform;
  XavierInit<XavierNormal> xavierNormal;
  xavierUniform.Initialize(weights, 100, 100);
  bool b = arma::all(arma::vectorise(weights)>=-(1e-2) || arma::vectorise(weights)<=(1e-2));
  BOOST_REQUIRE_EQUAL(b, 1);
  xavierNormal.Initialize(weights, 100, 100);
  //std::cout << weights << std::endl;
  auto ret = Shapiro(weights, 0.05);
  BOOST_REQUIRE_EQUAL(ret.accept, true);
}

BOOST_AUTO_TEST_CASE(HeInitTest)
{
  arma::mat weights;
  RandomSeed(21);
  HeInit<HeUniform> heInit;
  HeInit<HeNormal> heNormal;
  heInit.Initialize(weights, 100, 100);
  bool b = arma::all(arma::vectorise(weights)>=-(1e-2) || arma::vectorise(weights)<=(1e-2));
  BOOST_REQUIRE_EQUAL(b, 1);
  heNormal.Initialize(weights, 100, 100);
  auto ret = Shapiro(weights, 0.05);
  BOOST_REQUIRE_EQUAL(ret.accept, true);
}


// Test the GaussianInitialization class.
BOOST_AUTO_TEST_CASE(GaussianInitTest)
{
  arma::mat weights;
  arma::cube weights3d;
  size_t counter = 0;
  const size_t row = 7;
  const size_t col = 7;
  const size_t slice = 2;
  double mean, mean3d;
  double var,var3d;
  mean=mean3d=1;
  var=var3d=1;
  GaussianInitialization t(0, 0.2);
  //RandomSeed(21);
  for(size_t j=0; j<5; j++)
  {
    for(size_t i =0; i<10; i++)
    {
      t.Initialize(weights, row, col);
      t.Initialize(weights3d, row, col, slice);
      mean += arma::accu(weights) / weights.n_elem;
      var += arma::accu(pow((weights.t() - mean), 2)) / weights.n_elem -1 ;
      mean3d += arma::accu(weights3d.slice(0)) / weights3d.slice(0).n_elem;
      var3d += arma::accu(pow((weights3d.slice(0) - mean), 2)) / weights3d.slice(0).n_elem -1;
    }
    mean /= 10;
    var /= 10;
    mean3d /= 10;
    var3d /= 10;
    if((mean > 0 && mean < 0.4) && (var > 0 && var < 0.6) && 
      (mean3d > 0 && mean3d < 0.4) && (var3d >0 && var3d<0.6))
      counter++;
  }

  BOOST_REQUIRE(counter >= 1);
}

BOOST_AUTO_TEST_SUITE_END();
