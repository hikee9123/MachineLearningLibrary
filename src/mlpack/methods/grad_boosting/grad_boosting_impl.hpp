/*
 * @file methods/grad_boosting/grad_boosting_impl.hpp
 * @author Abhimanyu Dayal
 *
 * Implementation of the Gradient Boosting class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

// Include guard - Used to prevent including double copies of the same code
#ifndef MLPACK_METHODS_GRADBOOSTING_GRADBOOSTING_IMPL_HPP
#define MLPACK_METHODS_GRADBOOSTING_GRADBOOSTING_IMPL_HPP

// Base definition of the GradBoostingModel class.
#include <grad_boosting.hpp>
#include <mlpack/core.hpp>

// Defined within the mlpack namespace.
namespace mlpack {

// Empty constructor.
template<typename WeakLearnerType, typename MatType>
GradBoosting<WeakLearnerType, MatType>::GradBoosting() :
    numClasses(0),
    num_models(0)
{
// Nothing to do.
}


// In case the user has already initialised the weak learner
// Weak learner type "WeakLearnerInType" defined by the template
/**
 * Constructor.
 *
 * @param data Input data
 * @param labels Corresponding labels
 * @param num_models Number of weak learners
 * @param numClasses Number of classes
 * @param other Weak Learner, which has been initialized already.
 */
template<typename WeakLearnerType, typename MatType>
template<typename WeakLearnerInType>
GradBoosting<WeakLearnerType, MatType>::GradBoosting(
    
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const size_t num_models,

    // Pre-initiated weak learner
    const WeakLearnerInType& other,

    // This ensures that this constructor is only enabled if the WeakLearnerType 
    // and WeakLearnerInType are the same type
    const typename std::enable_if<
        std::is_same<WeakLearnerType, WeakLearnerInType>::value>::type*) :
    
    numClasses(numClasses),
    num_models(num_models)

{
  (void) TrainInternal<true>(data, labels, numClasses, other);
}

// In case the user inputs the arguments for the Weak Learner
/**
 * Constructor.
 *
 * @param data Input data
 * @param labels Corresponding labels
 * @param num_models Number of weak learners
 * @param numClasses Number of classes
 * @param other Weak Learner, which has been initialized already.
 */
template<typename WeakLearnerType, typename MatType>

// Variadic template to the Weak Learner arguments
template<typename... WeakLearnerArgs>
GradBoosting<WeakLearnerType, MatType>::GradBoosting(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const size_t num_models,
    WeakLearnerArgs&&... weakLearnerArgs) :
    num_models(num_models)
{
  WeakLearnerType other; // Will not be used.
  (void) TrainInternal<false>(data, labels, numClasses, other,
      weakLearnerArgs...);
}

// Train GradBoosting with a given weak learner.

template<typename WeakLearnerType, typename MatType>
template<typename WeakLearnerInType>
// typename MatType::elem_type GradBoosting<WeakLearnerType, MatType>::Train(
void GradBoosting<WeakLearnerType, MatType>::Train(
        const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const WeakLearnerInType& learner,
        const size_t num_models
    )
{
    return TrainInternal<true>(data, labels, numClasses, learner);
}

template<typename WeakLearnerType, typename MatType>
template<typename WeakLearnerInType>
// typename MatType::elem_type GradBoosting<WeakLearnerType, MatType>::Train(
void GradBoosting<WeakLearnerType, MatType>::Train(
        const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const size_t num_models,
        const typename std::enable_if<
            std::is_same<WeakLearnerType, WeakLearnerInType>::value>::type* = 0
    )
{
    WeakLearnerType other; // Will not be used.
    return TrainInternal<false>(data, labels, numClasses, other);
}



template<typename WeakLearnerType, typename MatType>
template<typename... WeakLearnerArgs>
// typename MatType::elem_type GradBoosting<WeakLearnerType, MatType>::Train(
void GradBoosting<WeakLearnerType, MatType>::Train(
        const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const size_t num_models,
        WeakLearnerArgs&&... weakLearnerArgs
    )
{
    WeakLearnerType other; // Will not be used.
    return TrainInternal<false>(data, labels, numClasses, other,
        weakLearnerArgs...);
}

// Classify the given test point.
template<typename WeakLearnerType, typename MatType>
template<typename VecType>
void GradBoosting<WeakLearnerType, MatType>::Classify(const VecType& point) 
{
    size_t prediction;
    Classify(point, prediction);

}

template<typename WeakLearnerType, typename MatType>
template<typename VecType>
void GradBoosting<WeakLearnerType, MatType>::Classify(
    const VecType& point,
    size_t& prediction)
{

    for(size_t i=0; i<weaklearners.size(); i++) 
    {
        size_t temp_pred = weaklearners[i].Classify(point);
        prediction += temp_pred;
    }

}


template<typename WeakLearnerType, typename MatType>
void Classify(const MatType& test,
                    arma::Row<size_t>& predictedLabels) 
{
    for(size_t i=0; i<test.size(); i++) {
        size_t prediction;
        Classify(test[i], prediction);
        predictedLabels[i] = prediction;
    }
}



template<
    bool UseExistingWeakLearner,
    typename MatType,
    typename WeakLearnerType,
    typename... WeakLearnerArgs
>
struct WeakLearnerTrainer
{
  static WeakLearnerType Train(
      const MatType& data,
      const arma::Row<size_t>& labels,
      const size_t numClasses,
      const WeakLearnerType& wl,
      WeakLearnerArgs&&... /* weakLearnerArgs */)
  {
    // Use the existing weak learner to train a new one with new weights.
    // API requirement: there is a constructor with this signature:
    //
    //    WeakLearnerType(const WeakLearnerType&,
    //                    MatType& data,
    //                    LabelsType& labels,
    //                    const size_t numClasses)
    //
    // This trains the new WeakLearnerType using the hyperparameters from the
    // given WeakLearnerType.
    return WeakLearnerType(wl, data, labels, numClasses);
  }
};


template<
    typename MatType,
    typename WeakLearnerType,
    typename... WeakLearnerArgs
>
struct WeakLearnerTrainer<
    false, MatType, WeakLearnerType, WeakLearnerArgs...
>
{
  static WeakLearnerType Train(
      const MatType& data,
      const arma::Row<size_t>& labels,
      const size_t numClasses,
      const WeakLearnerType& /* wl */,
      WeakLearnerArgs&&... weakLearnerArgs)
  {
    // When UseExistingWeakLearner is false, we use the given hyperparameters.
    // (This is the preferred approach that supports more types of weak
    // learners.)
    return WeakLearnerType(data, labels, numClasses,
        weakLearnerArgs...);
  }
};







// Template for GradBoosting template as a whole
template<typename WeakLearnerType, typename MatType>

// Template for TrainInternal 
// UseExistingWeakLearner determines whether to define a weak learner anew or 
// use an existing weak learner
// WeakLearnerArgs are the arguments for the weak learner
template<bool UseExistingWeakLearner, typename... WeakLearnerArgs>

// TrainInternal is a private function within GradBoosting class
// It has return type ElemType
// typename MatType::elem_type GradBoosting<WeakLearnerType, MatType>:: TrainInternal(
void GradBoosting<WeakLearnerType, MatType>:: TrainInternal(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t num_models,
    const size_t numClasses,
    const WeakLearnerType& wl,
    WeakLearnerArgs&&... weakLearnerArgs) 
{

    weaklearners.clear();
    
    arma::Row<double> residue = labels; 

    for(size_t model = 0; model < num_models; model++) 
    {
        // weak learner is trained at this point idk how yet

        WeakLearnerType w = WeakLearnerTrainer<
            UseExistingWeakLearner, MatType, WeakLearnerType,
            WeakLearnerArgs...
        >::Train(data, predictions, numClasses, wl, weakLearnerArgs...);

        weaklearners.push_back(w);

        arma::Row<double> predictions = residue;
        w.Classify(data, predictions);

        residue = residue - predictions;
        
    }
}

}

#endif

