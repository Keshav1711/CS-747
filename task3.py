"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the FaultyBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon, fault): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)
"""

# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

import numpy as np
import math

class FaultyBanditsAlgo:
    def __init__(self, num_arms, horizon, fault):
        self.num_arms = num_arms
        self.horizon = horizon
        self.fault = fault
        self.alpha = np.ones(num_arms)  #hyperparameter for alpha distribution
        self.beta = np.ones(num_arms)   #hyperparameter for alpha distribution
        self.fault_estimates = np.zeros(num_arms)
    
    def give_pull(self):
        exploration_bonus = np.sqrt(2 * np.log(self.horizon) / (self.alpha + self.beta))
        #estimated_means = self.alpha / (self.alpha + self.beta)
        sampled_means = np.random.beta(self.alpha + exploration_bonus, self.beta + exploration_bonus)
        combined_values = sampled_means * (1 - self.fault_estimates)

        return np.argmax(combined_values)
    
    def get_reward(self, arm_index, reward):
        is_faulty_pull = np.random.random() < self.fault_estimates[arm_index]
        if is_faulty_pull:
            reward = np.random.randint(2)  
        
        self.alpha[arm_index] += reward
        self.beta[arm_index] += 1 - reward
        
        self.fault_estimates[arm_index] = (self.fault_estimates[arm_index] * 0.9) + (0.1 * is_faulty_pull)