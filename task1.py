"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)

        # END EDITING HERE
    
    
    def give_pull(self):
        # START EDITING HERE
        total_counts = np.sum(self.counts)
        if total_counts == 0:
            return np.random.randint(self.num_arms)
        
        ucb_scores = self.values + np.sqrt(2 * np.log(total_counts) / (self.counts + 1e-6))
        return np.argmax(ucb_scores)
        #raise NotImplementedError
        # END EDITING HERE  
        
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value
        #raise NotImplementedError
        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.num_pulls = 0
        self.pulls = np.zeros(num_arms)
        self.rewards = np.zeros(num_arms)

    
    def kl_divergence(self, p, q):
        if p == 0:
            return math.log(1 / (1 - q))
        elif p == 1:
            return math.log(1 / q)
        else:
            return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
    
    def kl_ucb_score(self, p, n):
        left = p
        right = 1.0
        bound = (math.log(self.horizon) + 0 * math.log(math.log(self.horizon))) / (n + 1e-8)
        while right - left > 1e-6:
            mid = (left + right) / 2
            kl = self.kl_divergence(p, mid)
            if kl < bound:
                left = mid
            else:
                right = mid
        return (left + right) / 2
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        self.num_pulls += 1
        emp_mean = self.rewards / (self.pulls + 1e-8)
        kl_ucb = np.zeros(self.num_arms)
        for arm_index in range(self.num_arms):
            kl_ucb[arm_index] = self.kl_ucb_score(emp_mean[arm_index], self.pulls[arm_index])
        return np.argmax(kl_ucb)
        #raise NotImplementedError
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.pulls[arm_index] += 1
        self.rewards[arm_index] += reward
        #raise NotImplementedError
        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.alpha = np.ones(num_arms)
        self.beta = np.ones(num_arms)

        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
        #raise NotImplementedError
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if reward == 1:
            self.alpha[arm_index] += 1
        else:
            self.beta[arm_index] += 1
        #raise NotImplementedError
        # END EDITING HERE
