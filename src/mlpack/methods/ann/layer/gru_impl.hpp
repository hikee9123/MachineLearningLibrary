/**
 * @file methods/ann/layer/gru_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of the GRU class, which implements a gru network
 * layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_GRU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_GRU_IMPL_HPP

// In case it hasn't yet been included.
#include "gru.hpp"

#include "../visitor/forward_visitor.hpp"
#include "../visitor/backward_visitor.hpp"
#include "../visitor/gradient_visitor.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
GRU<InputDataType, OutputDataType>::GRU()
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
GRU<InputDataType, OutputDataType>::GRU(
    const size_t inSize,
    const size_t outSize,
    const size_t rho) :
    inSize(inSize),
    outSize(outSize),
    rho(rho),
    batchSize(1),
    forwardStep(0),
    backwardStep(0),
    gradientStep(0),
    deterministic(false)
{
  // Input specific linear layers(for zt, rt, ot).
  input2GateModule = new Linear<>(inSize, 3 * outSize);

  // Previous output gates (for zt and rt).
  output2GateModule = new LinearNoBias<>(outSize, 2 * outSize);

  // Previous output gate for ot.
  outputHidden2GateModule = new LinearNoBias<>(outSize, outSize);

  network.push_back(input2GateModule);
  network.push_back(output2GateModule);
  network.push_back(outputHidden2GateModule);

  inputGateModule = new SigmoidLayer<>();
  forgetGateModule = new SigmoidLayer<>();
  hiddenStateModule = new TanHLayer<>();

  network.push_back(inputGateModule);
  network.push_back(hiddenStateModule);
  network.push_back(forgetGateModule);

  prevError = arma::zeros<arma::mat>(3 * outSize, batchSize);

  allZeros = arma::zeros<arma::mat>(outSize, batchSize);

  outParameter.emplace_back(allZeros.memptr(),
      allZeros.n_rows, allZeros.n_cols, false, true);

  prevOutput = outParameter.begin();
  backIterator = outParameter.end();
  gradIterator = outParameter.end();
}

// Copy Constructor definition.
template <typename InputDataType, typename OutputDataType>
GRU<InputDataType, OutputDataType>::GRU(const GRU& layer) :
    inSize(layer.inSize),
    outSize(layer.outSize),
    rho(layer.rho),
    batchSize(layer.batchSize),
    forwardStep(layer.forwardStep),
    backwardStep(layer.backwardStep),
    gradientStep(layer.gradientStep),
    weights(layer.weights),
    deterministic(layer.deterministic)    

{
  // Nothing to do here.
}

// Move Constructor definition
template <typename InputDataType, typename OutputDataType>
GRU<InputDataType, OutputDataType>::GRU(GRU&& layer) :
    inSize(std::move(layer.inSize)),
    outSize(std::move(layer.outSize)),
    rho(std::move(layer.rho)),
    batchSize(std::move(layer.batchSize)),
    forwardStep(std::move(layer.forwardStep)),
    backwardStep(std::move(layer.backwardStep)),
    gradientStep(std::move(layer.gradientStep)),
    weights(std::move(layer.weights)),
    deterministic(std::move(layer.deterministic))

{
  // Nothing to do here.
}

// Copy operator definition.
template <typename InputDataType, typename OutputDataType>
GRU<InputDataType, OutputDataType>&
  GRU<InputDataType, OutputDataType>::operator=(const GRU& layer)
{
  if (this != &layer)
  {
    inSize = layer.inSize;
    outSize = layer.outSize;
    rho = layer.rho;
    batchSize = layer.batchSize;
    forwardStep = layer.forwardStep;
    backwardStep = layer.backwardStep;
    gradientStep = layer.gradientStep;
    weights = layer.weights;
    deterministic = layer.deterministic;

  }
  return *this;
}

// Move operator definition.
template <typename InputDataType, typename OutputDataType>
GRU<InputDataType, OutputDataType>&
  GRU<InputDataType, OutputDataType>::operator=(GRU&& layer)
{
  if (this != &layer)
  {
    inSize = std::move(layer.inSize);
    outSize = std::move(layer.outSize);
    rho = std::move(layer.rho);
    batchSize = std::move(layer.batchSize);
    forwardStep = std::move(layer.forwardStep);
    backwardStep = std::move(layer.backwardStep);
    gradientStep = std::move(layer.gradientStep);
    weights = std::move(layer.weights);
    deterministic = std::move(layer.deterministic);

  }
  return *this;
}


template<typename InputDataType, typename OutputDataType>
template<typename eT>
void GRU<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  if (input.n_cols != batchSize)
  {
    batchSize = input.n_cols;
    prevError.resize(3 * outSize, batchSize);
    allZeros.zeros(outSize, batchSize);
    // Batch size better not change during an iteration...
    if (outParameter.size() > 1)
    {
      Log::Fatal << "GRU<>::Forward(): batch size cannot change during a "
          << "forward pass!" << std::endl;
    }

    outParameter.clear();
    outParameter.emplace_back(allZeros.memptr(),
        allZeros.n_rows, allZeros.n_cols, false, true);

    prevOutput = outParameter.begin();
    backIterator = outParameter.end();
    gradIterator = outParameter.end();
  }

  // Process the input linearly(zt, rt, ot).
  boost::apply_visitor(ForwardVisitor(input,
      boost::apply_visitor(outputParameterVisitor, input2GateModule)),
      input2GateModule);

  // Process the output(zt, rt) linearly.
  boost::apply_visitor(ForwardVisitor(*prevOutput,
      boost::apply_visitor(outputParameterVisitor, output2GateModule)),
      output2GateModule);

  // Merge the outputs(zt and rt).
  output = (boost::apply_visitor(outputParameterVisitor,
      input2GateModule).submat(0, 0, 2 * outSize - 1, batchSize - 1) +
      boost::apply_visitor(outputParameterVisitor, output2GateModule));

  // Pass the first outSize through inputGate(it).
  boost::apply_visitor(ForwardVisitor(output.submat(
      0, 0, 1 * outSize - 1, batchSize - 1), boost::apply_visitor(
      outputParameterVisitor, inputGateModule)), inputGateModule);

  // Pass the second through forgetGate.
  boost::apply_visitor(ForwardVisitor(output.submat(
      1 * outSize, 0, 2 * outSize - 1, batchSize - 1),
      boost::apply_visitor(outputParameterVisitor, forgetGateModule)),
      forgetGateModule);

  arma::mat modInput = (boost::apply_visitor(outputParameterVisitor,
      forgetGateModule) % *prevOutput);

  // Pass that through the outputHidden2GateModule.
  boost::apply_visitor(ForwardVisitor(modInput,
      boost::apply_visitor(outputParameterVisitor, outputHidden2GateModule)),
      outputHidden2GateModule);

  // Merge for ot.
  arma::mat outputH = boost::apply_visitor(outputParameterVisitor,
      input2GateModule).submat(2 * outSize, 0, 3 * outSize - 1, batchSize - 1) +
      boost::apply_visitor(outputParameterVisitor, outputHidden2GateModule);

  // Pass it through hiddenGate.
  boost::apply_visitor(ForwardVisitor(outputH,
      boost::apply_visitor(outputParameterVisitor, hiddenStateModule)),
      hiddenStateModule);

  // Update the output (nextOutput): cmul1 + cmul2
  // Where cmul1 is input gate * prevOutput and
  // cmul2 is (1 - input gate) * hidden gate.
  output = (boost::apply_visitor(outputParameterVisitor, inputGateModule)
      % (*prevOutput - boost::apply_visitor(outputParameterVisitor,
      hiddenStateModule))) + boost::apply_visitor(outputParameterVisitor,
      hiddenStateModule);

  forwardStep++;
  if (forwardStep == rho)
  {
    forwardStep = 0;
    if (!deterministic)
    {
      outParameter.emplace_back(allZeros.memptr(),
          allZeros.n_rows, allZeros.n_cols, false, true);
      prevOutput = --outParameter.end();
    }
    else
    {
      *prevOutput = arma::mat(allZeros.memptr(),
          allZeros.n_rows, allZeros.n_cols, false, true);
    }
  }
  else if (!deterministic)
  {
    outParameter.push_back(output);
    prevOutput = --outParameter.end();
  }
  else
  {
    if (forwardStep == 1)
    {
      outParameter.clear();
      outParameter.push_back(output);

      prevOutput = outParameter.begin();
    }
    else
    {
      *prevOutput = output;
    }
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void GRU<InputDataType, OutputDataType>::Backward(
  const arma::Mat<eT>& input, const arma::Mat<eT>& gy, arma::Mat<eT>& g)
{
  if (input.n_cols != batchSize)
  {
    batchSize = input.n_cols;
    prevError.resize(3 * outSize, batchSize);
    allZeros.zeros(outSize, batchSize);
    // Batch size better not change during an iteration...
    if (outParameter.size() > 1)
    {
      Log::Fatal << "GRU<>::Forward(): batch size cannot change during a "
          << "forward pass!" << std::endl;
    }

    outParameter.clear();
    outParameter.emplace_back(allZeros.memptr(),
        allZeros.n_rows, allZeros.n_cols, false, true);

    prevOutput = outParameter.begin();
    backIterator = outParameter.end();
    gradIterator = outParameter.end();
  }

  arma::Mat<eT> gyLocal;
  if ((outParameter.size() - backwardStep  - 1) % rho != 0 && backwardStep != 0)
  {
    gyLocal = gy + boost::apply_visitor(deltaVisitor, output2GateModule);
  }
  else
  {
    gyLocal = arma::Mat<eT>(((arma::Mat<eT>&) gy).memptr(), gy.n_rows,
        gy.n_cols, false, false);
  }

  if (backIterator == outParameter.end())
  {
    backIterator = --(--outParameter.end());
  }

  // Delta zt.
  arma::mat dZt = gyLocal % (*backIterator -
      boost::apply_visitor(outputParameterVisitor,
      hiddenStateModule));

  // Delta ot.
  arma::mat dOt = gyLocal % (arma::ones<arma::mat>(outSize, batchSize) -
      boost::apply_visitor(outputParameterVisitor, inputGateModule));

  // Delta of input gate.
  boost::apply_visitor(BackwardVisitor(boost::apply_visitor(
      outputParameterVisitor, inputGateModule), dZt,
      boost::apply_visitor(deltaVisitor, inputGateModule)),
      inputGateModule);

  // Delta of hidden gate.
  boost::apply_visitor(BackwardVisitor(boost::apply_visitor(
      outputParameterVisitor, hiddenStateModule), dOt,
      boost::apply_visitor(deltaVisitor, hiddenStateModule)),
      hiddenStateModule);

  // Delta of outputHidden2GateModule.
  boost::apply_visitor(BackwardVisitor(boost::apply_visitor(
      outputParameterVisitor, outputHidden2GateModule),
      boost::apply_visitor(deltaVisitor, hiddenStateModule),
      boost::apply_visitor(deltaVisitor, outputHidden2GateModule)),
      outputHidden2GateModule);

  // Delta rt.
  arma::mat dRt = boost::apply_visitor(deltaVisitor, outputHidden2GateModule) %
      *backIterator;

  // Delta of forget gate.
  boost::apply_visitor(BackwardVisitor(boost::apply_visitor(
      outputParameterVisitor, forgetGateModule), dRt,
      boost::apply_visitor(deltaVisitor, forgetGateModule)),
      forgetGateModule);

  // Put delta zt.
  prevError.submat(0, 0, 1 * outSize - 1, batchSize - 1) = boost::apply_visitor(
      deltaVisitor, inputGateModule);

  // Put delta rt.
  prevError.submat(1 * outSize, 0, 2 * outSize - 1, batchSize - 1) =
      boost::apply_visitor(deltaVisitor, forgetGateModule);

  // Put delta ot.
  prevError.submat(2 * outSize, 0, 3 * outSize - 1, batchSize - 1) =
      boost::apply_visitor(deltaVisitor, hiddenStateModule);

  // Get delta ht - 1 for input gate and forget gate.
  arma::mat prevErrorSubview = prevError.submat(0, 0, 2 * outSize - 1,
      batchSize - 1);
  boost::apply_visitor(BackwardVisitor(boost::apply_visitor(
      outputParameterVisitor, input2GateModule),
      prevErrorSubview,
      boost::apply_visitor(deltaVisitor, output2GateModule)),
      output2GateModule);

  // Add delta ht - 1 from hidden state.
  boost::apply_visitor(deltaVisitor, output2GateModule) +=
      boost::apply_visitor(deltaVisitor, outputHidden2GateModule) %
      boost::apply_visitor(outputParameterVisitor, forgetGateModule);

  // Add delta ht - 1 from ht.
  boost::apply_visitor(deltaVisitor, output2GateModule) += gyLocal %
      boost::apply_visitor(outputParameterVisitor, inputGateModule);

  // Get delta input.
  boost::apply_visitor(BackwardVisitor(boost::apply_visitor(
      outputParameterVisitor, input2GateModule), prevError,
      boost::apply_visitor(deltaVisitor, input2GateModule)),
      input2GateModule);

  backwardStep++;
  backIterator--;

  g = boost::apply_visitor(deltaVisitor, input2GateModule);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void GRU<InputDataType, OutputDataType>::Gradient(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& /* error */,
    arma::Mat<eT>& /* gradient */)
{
  if (input.n_cols != batchSize)
  {
    batchSize = input.n_cols;
    prevError.resize(3 * outSize, batchSize);
    allZeros.zeros(outSize, batchSize);
    // Batch size better not change during an iteration...
    if (outParameter.size() > 1)
    {
      Log::Fatal << "GRU<>::Forward(): batch size cannot change during a "
          << "forward pass!" << std::endl;
    }

    outParameter.clear();
    outParameter.emplace_back(allZeros.memptr(),
        allZeros.n_rows, allZeros.n_cols, false, true);

    prevOutput = outParameter.begin();
    backIterator = outParameter.end();
    gradIterator = outParameter.end();
  }

  if (gradIterator == outParameter.end())
  {
    gradIterator = --(--outParameter.end());
  }

  boost::apply_visitor(GradientVisitor(input, prevError), input2GateModule);

  boost::apply_visitor(GradientVisitor(
      *gradIterator,
      prevError.submat(0, 0, 2 * outSize - 1, batchSize - 1)),
      output2GateModule);

  boost::apply_visitor(GradientVisitor(
      *gradIterator % boost::apply_visitor(outputParameterVisitor,
      forgetGateModule),
      prevError.submat(2 * outSize, 0, 3 * outSize - 1, batchSize - 1)),
      outputHidden2GateModule);

  gradIterator--;
}

template<typename InputDataType, typename OutputDataType>
void GRU<InputDataType, OutputDataType>::ResetCell(const size_t /* size */)
{
  outParameter.clear();
  outParameter.emplace_back(allZeros.memptr(),
    allZeros.n_rows, allZeros.n_cols, false, true);

  prevOutput = outParameter.begin();
  backIterator = outParameter.end();
  gradIterator = outParameter.end();

  forwardStep = 0;
  backwardStep = 0;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void GRU<InputDataType, OutputDataType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  // If necessary, clean memory from the old model.
  if (cereal::is_loading<Archive>())
  {
    boost::apply_visitor(deleteVisitor, input2GateModule);
    boost::apply_visitor(deleteVisitor, output2GateModule);
    boost::apply_visitor(deleteVisitor, outputHidden2GateModule);
    boost::apply_visitor(deleteVisitor, inputGateModule);
    boost::apply_visitor(deleteVisitor, forgetGateModule);
    boost::apply_visitor(deleteVisitor, hiddenStateModule);
  }

  ar(CEREAL_NVP(inSize));
  ar(CEREAL_NVP(outSize));
  ar(CEREAL_NVP(rho));

  ar(CEREAL_VARIANT_POINTER(input2GateModule));
  ar(CEREAL_VARIANT_POINTER(output2GateModule));
  ar(CEREAL_VARIANT_POINTER(outputHidden2GateModule));
  ar(CEREAL_VARIANT_POINTER(inputGateModule));
  ar(CEREAL_VARIANT_POINTER(forgetGateModule));
  ar(CEREAL_VARIANT_POINTER(hiddenStateModule));
}

} // namespace ann
} // namespace mlpack

#endif
