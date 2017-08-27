/**
 * @file gan_network_test.cpp
 * @author Kris Singh
 *
 * Tests the gan Network
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/gan.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/core/optimizers/minibatch_sgd/minibatch_sgd.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::math;
using namespace mlpack::optimization;
using namespace mlpack::regression;
using namespace std::placeholders;

BOOST_AUTO_TEST_SUITE(GANNetworkTest);

/*
 * Load pre trained network values
 * for generating distribution that
 * is close to N(4, 0.5)
 */
BOOST_AUTO_TEST_CASE(GanTest)
{
  size_t generatorHiddenLayerSize = 8;
  size_t discriminatorHiddenLayerSize = 8;
  size_t generatorOutputSize = 1;
  size_t discriminatorOutputSize = 1;
  size_t discriminatorPreTrain = 0;
  size_t batchSize = 8;
  size_t noiseDim = 1;
  size_t generatorUpdateStep = 1;
  size_t numSamples = 10000;
  double multiplier = 1;

  arma::mat trainData(1, 10000);
  trainData.imbue( [&]() { return arma::as_scalar(RandNormal(4, 0.5));});
  trainData = arma::sort(trainData);

  // Create the Discrminator network
  FFN<CrossEntropyError<>> discriminator;
  discriminator.Add<Linear<>> (
      generatorOutputSize, discriminatorHiddenLayerSize * 2);
  discriminator.Add<ReLULayer<>>();
  discriminator.Add<Linear<>> (
      discriminatorHiddenLayerSize * 2, discriminatorHiddenLayerSize * 2);
  discriminator.Add<ReLULayer<>>();
  discriminator.Add<Linear<>> (
      discriminatorHiddenLayerSize * 2, discriminatorHiddenLayerSize * 2);
  discriminator.Add<ReLULayer<>>();
  discriminator.Add<Linear<>> (
      discriminatorHiddenLayerSize * 2, discriminatorOutputSize);
  discriminator.Add<SigmoidLayer<>>();
  // Create the Generator network
  FFN<CrossEntropyError<>> generator;
  generator.Add<Linear<>>(noiseDim, generatorHiddenLayerSize);
  generator.Add<SoftPlusLayer<>>();
  generator.Add<Linear<>>(generatorHiddenLayerSize, generatorOutputSize);

  // Create Gan
  GaussianInitialization gaussian(0, 0.1);
  std::function<double ()> noiseFunction = [](){ return math::Random(-8, 8) +
      math::RandNormal(0, 1) * 0.01;};
  GAN<FFN<CrossEntropyError<>>,
      GaussianInitialization,
      std::function<double()>>
  gan(trainData, generator, discriminator, gaussian, noiseFunction,
      noiseDim, batchSize, generatorUpdateStep, discriminatorPreTrain,
      multiplier);
  gan.Reset();

  std::cout << "Loading Parameters" << std::endl;
  arma::mat parameters, generatorParameters;
  parameters.load("preTrainedGAN.arm");
  gan.Parameters() = parameters;

  // Generate samples
  Log::Info << "Sampling..." << std::endl;
  arma::mat noise(noiseDim, 1);

  size_t dim = std::sqrt(trainData.n_rows);
  arma::mat generatedData(2 * dim, dim * numSamples);

  for (size_t i = 0; i < numSamples; i++)
  {
    arma::mat samples;
    noise.imbue( [&]() { return noiseFunction(); } );

    generator.Forward(noise, samples);
    samples.reshape(dim, dim);
    samples = samples.t();

    generatedData.submat(0, i * dim, dim - 1, i * dim + dim - 1) = samples;


    samples = trainData.col(math::RandInt(0, trainData.n_cols));
    samples.reshape(dim, dim);
    samples = samples.t();

    generatedData.submat(dim,
        i * dim, 2 * dim - 1, i * dim + dim - 1) = samples;
  }

  double generatedMean = arma::as_scalar(arma::mean(
      generatedData.rows(0, dim - 1), 1));
  double originalMean = arma::as_scalar(arma::mean(
      generatedData.rows(dim, 2 * dim - 1), 1));
  double generatedStd = arma::as_scalar(arma::stddev(
      generatedData.rows(0, dim - 1), 0, 1));
  double originalStd = arma::as_scalar(arma::stddev(
      generatedData.rows(dim, 2 * dim - 1), 0, 1));

  BOOST_REQUIRE_LE(generatedMean - originalMean, 0.2);
  BOOST_REQUIRE_LE(generatedStd - originalStd, 0.2);
}
BOOST_AUTO_TEST_SUITE_END();
