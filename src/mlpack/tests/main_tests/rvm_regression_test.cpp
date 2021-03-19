/**
 * @file tests/main_tests/rvm_regression_test.cpp
 * @author Clement Mercier
 *
 * Test mlpackMain() of rvm_regression_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "RVMRegression";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/rvm_regression/rvm_regression_main.cpp>
#include <mlpack/core/kernels/linear_kernel.hpp>
#include "test_helper.hpp"


#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

struct RVMRegressionTestFixture
{
 public:
  RVMRegressionTestFixture()
  {
    // Cache in the options for this program.
    IO::RestoreSettings(testName);
  }

  ~RVMRegressionTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    IO::ClearSettings();
  }
};

/**
 * Check the center and scale options.
 */
TEST_CASE_METHOD(RVMRegressionTestFixture,
                 "RVMRegressionCenter0Scale0",
                 "[RVMRegressionMainTest][BindingTests]")
{
  int n = 50, m = 4;
  arma::mat matX = arma::randu<arma::mat>(m, n);
  arma::rowvec omega = arma::randu<arma::rowvec>(m);
  arma::rowvec y =  omega * matX;

  SetInputParam("input", std::move(matX));
  SetInputParam("responses", std::move(y));
  SetInputParam("center", false);
  SetInputParam("scale", false);
  SetInputParam("kernel", std::string("linear"));

  mlpackMain();

  const RVMRegressionModel* estimator =
      IO::GetParam<RVMRegressionModel*>("output_model");

  REQUIRE(estimator->template
          RVMPtr<mlpack::kernel::LinearKernel>()->DataOffset().n_elem == 0);
  REQUIRE(estimator->template
	  RVMPtr<mlpack::kernel::LinearKernel>()->DataScale().n_elem == 0);
}


// Check predictions of saved model and in code model are equal.
TEST_CASE_METHOD(RVMRegressionTestFixture,
                 "RVMRegressionSavedEqualCode",
                 "[RVMRegressionMainTest][BindingTests]")
{
  int n = 10, m = 4;
  arma::mat matX = arma::randu<arma::mat>(m, n);
  arma::mat matXtest = arma::randu<arma::mat>(m, 2 * n);
  const arma::rowvec omega = arma::randu<arma::rowvec>(m);
  arma::rowvec y =  omega * matX;

  mlpack::kernel::LinearKernel kernel;  
  RVMRegression<mlpack::kernel::LinearKernel> model(kernel, true, true, true);
  model.Train(matX, y);

  arma::rowvec responses, uncertainties;
  model.Predict(matXtest, responses, uncertainties);

  SetInputParam("input", std::move(matX));
  SetInputParam("responses", std::move(y));
  SetInputParam("center", true);
  SetInputParam("scale", true);

  mlpackMain();

  IO::GetSingleton().Parameters()["input"].wasPassed = false;
  IO::GetSingleton().Parameters()["responses"].wasPassed = false;

  SetInputParam("input_model",
                IO::GetParam<RVMRegressionModel*>("output_model"));
  SetInputParam("test", std::move(matXtest));

  mlpackMain();

  arma::mat ytest = std::move(responses);
  arma::mat uncertaintiesTest = std::move(uncertainties);

  // Check that initial output and output using saved model are same.
  CheckMatrices(ytest, IO::GetParam<arma::mat>("predictions"));
  CheckMatrices(uncertaintiesTest, IO::GetParam<arma::mat>("stds"));
}

/**
 * Check a crash happens if neither input or input_model are specified.
 * Check a crash happens if both input and input_model are specified.
 */
TEST_CASE_METHOD(RVMRegressionTestFixture,
                 "RVMCheckParamsPassed",
                 "[RVMRegressionMainTest][BindingTests]")
{
  int n = 10, m = 4;
  arma::mat matX = arma::randu<arma::mat>(m, n);
  arma::mat matXtest = arma::randu<arma::mat>(m, 2 * n);
  const arma::rowvec omega = arma::randu<arma::rowvec>(m);
  arma::rowvec y =  omega * matX;


  // Check that std::runtime_error is thrown if neither input or input_model
  // is specified.
  SetInputParam("responses", std::move(y));

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Continue only with input passed.
  SetInputParam("input", std::move(matX));
  mlpackMain();

  // Now pass the previous trained model and one input matrix at the same time.
  // An error should occur.
  SetInputParam("input", std::move(matX));
  SetInputParam("input_model",
                IO::GetParam<RVMRegressionModel*>("output_model"));
  SetInputParam("test", std::move(matXtest));

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}
