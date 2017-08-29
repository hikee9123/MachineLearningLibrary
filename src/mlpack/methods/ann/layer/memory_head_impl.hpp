/**
 * @file memory_unit_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of Memory Head layer, used in Neural Turing Machine.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MEMORY_HEAD_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_MEMORY_HEAD_IMPL_HPP

// In case it hasn't yet been included.
#include "memory_head.hpp"

#include <algorithm>

#include "../visitor/forward_visitor.hpp"
#include "../visitor/backward_visitor.hpp"
#include "../visitor/gradient_visitor.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <typename InputDataType, typename OutputDataType>
MemoryHead<InputDataType, OutputDataType>::MemoryHead(
    const size_t inSize,
    const size_t outSize,
    const size_t memSize,
    const size_t shiftRange) :
    inSize(inSize),
    outSize(outSize),
    memSize(memSize),
    shiftSize(shiftRange),
    deterministic(false)
{
  // Build linear for kT + bT + gT + sT + gammaT
  inputLinear = new Linear<>(inSize, (memSize) + (1) + (1) +
      (2 * shiftSize + 1) + (1));

  // kT non linearity.
  kTNonLinear = new TanHLayer<>();

  network.push_back(inputLinear);
  network.push_back(kTNonLinear);

  prevWeights.push_back(arma::zeros<arma::mat>(outSize, 1));
  weightsBackwardIterator = prevWeights.end();

  prevError = arma::zeros<arma::mat>((memSize) + (1) + (1) +
      (2 * shiftSize + 1) + (1), 1);

  bWdash = lWDash.end();
  bGammaT = lGammaT.end();
  bWTilde = lWTilde.end();
  bShiftMatrix = lShiftMatrix.end();
  bWg = lWg.end();
  bSt = lSt.end();
  bGt = lGt.end();
  bWe = lWe.end();
  bWc = lWc.end();
  bBt = lBt.end();
  bCosineT = lConsineT.end();
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void MemoryHead<InputDataType, OutputDataType>::ForwardWithMemory(
    arma::Mat<eT>&& input, const arma::Mat<eT>&& memory, arma::Mat<eT>&& output)
{
  // Pass the input through linear layer.
  boost::apply_visitor(ForwardVisitor(std::move(input), std::move(
      boost::apply_visitor(outputParameterVisitor, inputLinear))),
      inputLinear);

  arma::mat& lOutput = boost::apply_visitor(outputParameterVisitor,
    inputLinear);

  // Build kT with non linearity.
  boost::apply_visitor(ForwardVisitor(std::move(lOutput.submat(0, 0,
      memSize - 1, 0)), std::move(boost::apply_visitor(outputParameterVisitor,
      kTNonLinear))), kTNonLinear);
  const arma::mat& kT = boost::apply_visitor(outputParameterVisitor,
      kTNonLinear);

  // Build bT with non linearity
  lBt.push_back(bTNonLinear.Fn(lOutput(memSize, 0)));
  const double& bT = lBt.back();

  // Build gT with non linearity
  lGt.push_back(gTNonLinear.Fn(lOutput(memSize + 1, 0)));
  const double& gT = lGt.back();

  // Build sT with non linearity
  arma::vec temp = arma::exp(lOutput.submat(memSize + 2, 0,
    memSize + 2 + 2 * shiftSize, 0));
  temp = temp / arma::as_scalar(arma::sum(temp));
  lSt.push_back(std::move(temp));
  const arma::vec& sT = lSt.back();

  // Build gammaT with non linearity
  lGammaT.push_back(gammaTNonLinear.Fn(arma::as_scalar(
    lOutput.submat(memSize + 2 + 2 * shiftSize + 1, 0,
    memSize + 2 + 2 * shiftSize + 1, 0))));
  const double& gammaT = lGammaT.back();

  // Perform cosine similarity with memory content
  lConsineT.push_back(arma::normalise(memory, 2, 1) * arma::normalise(kT));
  const arma::vec& cosSimilarity = lConsineT.back();

  // Build wC with bT and softmax
  lWe.push_back(arma::exp(bT * cosSimilarity));
  const arma::vec& wE = lWe.back();

  lWc.push_back(wE / arma::as_scalar(arma::sum(wE)));
  const arma::vec& wC = lWc.back();

  // Build wG with gT
  lWg.push_back(prevWeights.back() + arma::as_scalar(gT) *
    (wC - prevWeights.back()));
  const arma::vec& wG = lWg.back();

  // Perform circular convolution with sT
  arma::mat shiftVec = arma::shift(arma::flipud(sT), 1);
  size_t numRep = std::ceil(((double)outSize) / (2 * shiftSize + 1));

  if (numRep > 1)
  {
    shiftVec = arma::repmat(shiftVec, numRep, 1);
  }

  lShiftMatrix.push_back(arma::mat(outSize, outSize, arma::fill::none));
  arma::mat& shiftMatrix = lShiftMatrix.back();

  shiftMatrix.each_col([&](arma::vec& a)
  {
    a = shiftVec.submat(0, 0, wG.n_rows - 1, 0);
    shiftVec = arma::shift(std::move(shiftVec), 1);
  });

  lWTilde.push_back(arma::trans(arma::trans(wG) * shiftMatrix));
  const arma::vec& wTilde = lWTilde.back();

  // Sharpening
  lWDash.push_back(arma::pow(wTilde, gammaT + 1));
  const arma::vec& wDash = lWDash.back();

  output = wDash / arma::as_scalar(arma::sum(wDash));

  if (!deterministic)
  {
    prevWeights.push_back(output);
  }
  else
  {
    prevWeights.back() = output;
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void MemoryHead<InputDataType, OutputDataType>::BackwardWithMemory(
  const arma::Mat<eT>&& /* output */,
  const arma::Mat<eT>&& memory,
  arma::Mat<eT>&& gy,
  arma::Mat<eT>&& g,
  arma::Mat<eT>&& gM)
{
  if (bBt == lBt.end())
  {
    bBt = (--lBt.end());
    bCosineT = (--lConsineT.end());
    bWe = (--lWe.end());
    bWc = (--lWc.end());
    bGt = (--lGt.end());
    bShiftMatrix = (--lShiftMatrix.end());
    bWg = (--lWg.end());
    bWTilde = (--lWTilde.end());
    bSt = (--lSt.end());
    bWdash = (--lWDash.end());
    bGammaT = (--lGammaT.end());

    weightsBackwardIterator = --(--prevWeights.end());
  }
  else
  {
    gy += prevDW;
  }

  // Load parameters.
  const double& bT = *bBt;
  const arma::vec& consineT = *bCosineT;
  const arma::vec& wE = *bWe;
  const arma::vec& wC = *bWc;
  const double& gT = *bGt;
  const arma::mat wG = *bWg;
  const arma::mat& shiftMatrix = *bShiftMatrix;
  const arma::vec& wTilde = *bWTilde;
  const arma::vec& sT = *bSt;
  const double& gammaT = *bGammaT;
  const arma::mat& wDash = *bWdash;

  // Error of output
  arma::mat& dW = gy;

  // Error of wDash
  double sum1 = arma::as_scalar(arma::sum(wDash));
  double sum2 = arma::as_scalar(arma::sum(dW % wDash)) / (sum1 * sum1);
  dW.for_each([&] (double& val)
  {
    val = (val / sum1) - sum2;
  });

  // Error of gammaT
  prevError(memSize + 2 + 2 * shiftSize + 1, 0) = gammaTNonLinear.Deriv(
    boost::apply_visitor(outputParameterVisitor, inputLinear)
    (memSize + 2 + 2 * shiftSize + 1, 0)) *
    arma::as_scalar(arma::sum(dW % wDash % arma::log(wTilde)));

  // Error of Wtilde
  dW %= (gammaT + 1) * arma::pow(wTilde, gammaT);

  // Error of shiftMatrix
  arma::mat dShiftMatrix = wG * arma::trans(dW);

  // Compress shiftMatrix error
  size_t rowIndex = 2 * shiftSize + 1;
  while (rowIndex < shiftMatrix.n_rows)
  {
    const arma::mat& toAdd = dShiftMatrix.submat(rowIndex, 0,
      std::min(rowIndex + 2 * shiftSize, (size_t)shiftMatrix.n_rows - 1),
      shiftMatrix.n_cols - 1);

    dShiftMatrix.submat(0, 0, toAdd.n_rows - 1, shiftMatrix.n_cols - 1) +=
      toAdd;

    rowIndex += 2 * shiftSize + 1;
  }

  size_t colIndex = 2 * shiftSize + 1;

  while (colIndex < shiftMatrix.n_cols)
  {
    const arma::mat& toAdd = dShiftMatrix.submat(0, colIndex, 2 * shiftSize,
      std::min(colIndex + 2 * shiftSize, (size_t)shiftMatrix.n_cols - 1));
    dShiftMatrix.submat(0, 0, 2 * shiftSize, toAdd.n_cols - 1) += toAdd;

    colIndex += 2 * shiftSize + 1;
  }

  arma::mat sDShiftMatrix = dShiftMatrix.submat(0, 0, 2 * shiftSize,
    2 * shiftSize);

  arma::vec dSt = arma::zeros(2 * shiftSize + 1);

  colIndex = 0;
  dShiftMatrix.each_col([&](arma::vec& v)
  {
    if (colIndex < 2 * shiftSize + 1)
    {
      dSt = std::move(arma::shift(std::move(dSt), 1));
      dSt += v.submat(0, 0, 2 * shiftSize, 0);
    }

    colIndex++;
  });

  // Error of St
  dSt = arma::flipud(dSt) % sT;
  sum1 = arma::as_scalar(arma::sum(dSt));
  dSt -= sum1 * sT;
  prevError.submat(memSize + 2, 0, memSize + 2 + 2 * shiftSize, 0) = dSt;

  // Error of Wg.
  dW = shiftMatrix * dW;

  // Error of previously computed weights.
  prevDW = (1 - gT) * dW;

  // Error of Gt.
  prevError(memSize + 1, 0) = gTNonLinear.Deriv(gT) *
    arma::as_scalar(arma::sum(dW % (wC - *weightsBackwardIterator)));

  // Error of wC.
  dW *= gT;

  // Error of We
  sum1 = arma::as_scalar(arma::sum(wE));
  sum2 = arma::as_scalar(arma::sum(wE % dW));
  dW.for_each([&] (double& val)
  {
    val = (val / sum1) - (sum2 / (sum1 * sum1));
  });

  // Error of We without exponential.
  dW %= wE;

  // Error of Bt.
  prevError(memSize, 0) = bTNonLinear.Deriv(
    boost::apply_visitor(outputParameterVisitor, inputLinear)(memSize, 0)) *
    arma::as_scalar(arma::sum(dW % consineT));

  // Error of cosine
  dW *= bT;

  // Normalised memory, will be normalised in the upcoming loop.
  arma::mat nMemory = memory;

  const arma::mat& kT = boost::apply_visitor(outputParameterVisitor,
      kTNonLinear);

  double kTNorm = arma::norm(kT);

  arma::mat nKt = kT / kTNorm;

  // Error of memory with normalization.
  gM = dW * arma::trans(nKt);

  // Error of memory without normalization.
  rowIndex = 0;
  gM.each_row([&] (arma::rowvec& v)
  {
    double n = arma::norm(memory.row(rowIndex));
    nMemory.row(rowIndex) /= n;
    v = (v - (nMemory.row(rowIndex) * arma::as_scalar(arma::sum(
        nMemory.row(rowIndex) % v)))) / n;

    rowIndex++;
  });

  // Error of Kt with normalization
  arma::mat dKt = arma::trans(nMemory) * dW;

  // // Error of Kt without normalization.
  dKt = (dKt - (nKt * arma::as_scalar(arma::sum(nKt % dKt)))) / kTNorm;

  // Backward pass through Kt gate.
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, kTNonLinear)), std::move(dKt),
      std::move(boost::apply_visitor(deltaVisitor, kTNonLinear))),
      kTNonLinear);

  prevError.submat(0, 0, memSize - 1, 0) = boost::apply_visitor(deltaVisitor,
      kTNonLinear);

  // Backward pass through linear gate.
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, inputLinear)), std::move(prevError),
      std::move(boost::apply_visitor(deltaVisitor, inputLinear))),
      inputLinear);

  g = boost::apply_visitor(deltaVisitor, inputLinear);

  bBt--;
  bCosineT--;
  bWe--;
  bWc--;
  bGt--;
  bShiftMatrix--;
  bWg--;
  bWTilde--;
  bSt--;
  bGammaT--;
  bWdash--;

  weightsBackwardIterator--;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void MemoryHead<InputDataType, OutputDataType>::Gradient(
    arma::Mat<eT>&& input,
    arma::Mat<eT>&& /* error */,
    arma::Mat<eT>&& /* gradient */)
{
  boost::apply_visitor(GradientVisitor(std::move(input), std::move(prevError)),
      inputLinear);
}

template<typename InputDataType, typename OutputDataType>
void MemoryHead<InputDataType, OutputDataType>::ResetCell()
{
  prevWeights.clear();
  prevWeights.push_back(arma::zeros<arma::mat>(outSize, 1));
  weightsBackwardIterator = prevWeights.end();

  prevError = arma::zeros<arma::mat>((memSize) + (1) + (1) +
      (2 * shiftSize + 1) + (1), 1);

  lWDash.clear();
  lGammaT.clear();
  lWTilde.clear();
  lShiftMatrix.clear();
  lWg.clear();
  lSt.clear();
  lGt.clear();
  lWe.clear();
  lWc.clear();
  lBt.clear();
  lConsineT.clear();

  bWdash = lWDash.end();
  bGammaT = lGammaT.end();
  bWTilde = lWTilde.end();
  bShiftMatrix = lShiftMatrix.end();
  bWg = lWg.end();
  bSt = lSt.end();
  bGt = lGt.end();
  bWe = lWe.end();
  bWc = lWc.end();
  bBt = lBt.end();
  bCosineT = lConsineT.end();
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void MemoryHead<InputDataType, OutputDataType>::Serialize(
    Archive& /* ar */, const unsigned int /* version */)
{
}

} // namespace ann
} // namespace mlpack

#endif
