/**
 * @file    elm.hpp
 * @author  Siddharth Agrawal
 * @mail    siddharthcore@gmail.com
 *
 * Basic Extreme Learning Machine
 * Extreme Learning Machine(ELM) is a single-hidden layer feedforward neural networks(SLFNs) which randomly chooses hidden nodes and  
 * analytically determines the output weights of SLFNs. 
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_Elm_HPP
#define MLPACK_METHODS_Elm_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core.hpp>

namespace mlpack {
namespace elm {

class Elm
{
 public: 
  Elm(const arma::mat& predictors,
      const arma::mat& responses,
      const size_t act=0,
      const size_t Nh=0,	    //Number of Hidden Neurons
      const size_t N=0,          //Number of data points
      const size_t D=0,         //Data Dimension
      const double lambda = 0,
      const double alpha = 0);

  void Train(const arma::mat& predictors,
             const arma::mat& responses,
             const size_t act);
 
  void Predict(const arma::mat& points,
               const arma::mat& predictions);

  void InitWeightbias();     //Initialise Weights and Biases randomly
	
  double Lambda() const { return lambda; }
  double& Lambda() { return lambda; }

  double Alpha() const { return alpha; }
  double& Alpha() { return alpha; }


  //Serialize the model

  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(lambda, "lambda");
    ar & data::CreateNVP(alpha, "alpha");
  }

 private:
  double lambda;
  double alpha;

};

} // namespace elm
} // namespace mlpack

#endif
