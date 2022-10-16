/**
 * @file methods/reinforcement_learning/environment/bit_flipping.hpp
 * @author Eshaan Agarwal
 *
 * This file is an implementation of Bit Flipping toy task:
 * https://www.gymlibrary.ml/environments/classic_control/cart_pole
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_ENVIRONMENT_BIT_FLIPPING_HPP
#define MLPACK_METHODS_RL_ENVIRONMENT_BIT_FLIPPING_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Implementation of Cart Pole task.
 */
class BitFlipping
{
 public:
  /**
   * Implementation of the state of Bit Flipping. Each state is a 
   * (position, velocity, angle, angular velocity).
   */
  class State
  {
   public:
    /**
     * Construct a state instance.
     */
    State() : data(dimension)
    { /* Nothing to do here. */ }

    /**
     * Construct a state instance from given data.
     *
     * @param data Data 
     */
    State(const arma::colvec& data) : data(data)
    { /* Nothing to do here */ }

    //! Modify the internal representation of the state.
    arma::colvec& Data() { return data; }

    //! Get the internal representation of the state.
    arma::colvec Data() const { return data; }

    //! Encode the state to a column vector.
    const arma::colvec& Encode() const { return data; }

    //! Dimension of the encoded state.
    static constexpr size_t dimension = 1;

   private:
    //! Locally-stored n bit integer.
    arma::colvec data;
  };

  /**
   * Implementation of action of Cart Pole.
   */
  class Action
  {
   public:
    
    // To store the action ( index of n bit number to be flipped )
    size_t action;

    // Track the size of the action space.
    static const size_t size = 1;
  };

  /**
   * Construct a binary vector instance using the given constants.
   *
   * @param maxSteps The number of steps after which the episode
   *    terminates. If the value is 0, there is no limit.
   * @param length Length of the binary vector for state and goal
   */
  BitFlipping(const size_t maxSteps = 200,
              const size_t length = 10) :
      maxSteps(maxSteps),
      length(length),
      stepsPerformed(0)
  { 
    goal = arma::randi<arma::colvec>(length, arma::distr_param(0, 1));
  }

  /**
   * Get reward and next state based on current
   * state and current action.
   *
   * @param state The current state.
   * @param action The current action.
   * @param nextState The next state.
   * @return reward, it's always 1.0.
   */
  double Sample(const State& state,
                const Action& action,
                State& nextState)
  {
    // Update the number of steps performed.
    stepsPerformed++;

    arma::colvec modifiedState = state.Data();
    modifiedState(action.action) = 1 - modifiedState(action.action);
    nextState.Data()= modifiedState;


    // Check if the episode has terminated.
    bool done = IsTerminal(nextState);

    // Do not reward agent if it failed.
    if (done)
      return 1.0;

    if ( sum(nextState.Data()) == sum(goal))
    {
      return 1.0;
    }

    return 0.0;
  }

  /**
   * Get reward based on current state and current
   * action.
   *
   * @param state The current state.
   * @param action The current action.
   * @return reward, it's always 1.0.
   */
  double Sample(const State& state, const Action& action)
  {
    State nextState;
    return Sample(state, action, nextState);
  }

  /**
   * Initial state representation is randomly generated within [-0.05, 0.05].
   *
   * @return Initial state for each episode.
   */
  State InitialSample()
  {
    stepsPerformed = 0;
    arma::colvec initialState = arma::randi<arma::colvec>(length, arma::distr_param(0, 1));
    return State(initialState);
  }

  /**
   * This function checks if the cart has reached the terminal state.
   *
   * @param state The desired state.
   * @return true if state is a terminal state, otherwise false.
   */
  bool IsTerminal(const State& state) const
  { 
    arma::colvec currentState = state.Data();
    if (maxSteps != 0 && stepsPerformed >= maxSteps)
    {
      Log::Info << "Episode terminated due to the maximum number of steps"
          "being taken.";
      return true;
    }
    else if (sum(currentState - goal) == 0)
    {
      Log::Info << "Episode terminated as agent has reached desired goal.";
      return true;
    }
    return false;
  }

  //! Get the number of steps performed.
  size_t StepsPerformed() const { return stepsPerformed; }

  //! Get the maximum number of steps allowed.
  size_t MaxSteps() const { return maxSteps; }
  //! Set the maximum number of steps allowed.
  size_t& MaxSteps() { return maxSteps; }

  //! Get the size of binary vector.
  size_t Length() const { return length; }
  //! Set the size of binary vector
  size_t& Length() { return length; }

  //! Get the goal for the episode
  arma::colvec Goal() const { return goal; }
  //! Set the goal for the episode
  arma::colvec& Goal() { return goal; }

 private:
  //! Locally-stored maximum number of steps.
  size_t maxSteps;

  //! Locally-stored done reward.
  double doneReward;

  //! Locally-stored number of steps performed.
  size_t stepsPerformed;

  //! Locally stored goal for the epsiode
  arma::colvec goal;

  //! Locally stored size of binary vector
  size_t length;
};

} // namespace mlpack

#endif
