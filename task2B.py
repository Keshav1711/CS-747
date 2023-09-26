"""
You need to write code to plot the graphs as required in task2 of the problem statement:
    - You can edit any code in this file but be careful when modifying the simulation specific code. 
    - The simulation framework as well as the BernoulliBandit implementation for this task have been separated from the rest of the assignment code and is contained solely in this file. This will be useful in case you would like to collect more information from runs rather than just regret.
"""

import numpy as np
from multiprocessing import Pool
from task1 import Eps_Greedy, UCB, KL_UCB
import matplotlib.pyplot as plt
# START EDITING HERE
# You can use this space to define any helper functions that you need.
HORIZON = 30000
DELTA = 0.1
P1_values = np.arange(0.1, 1.0, 0.05)  # Vary p1 from 0.1 to 0.95 in steps of 0.05
P2_values = P1_values - DELTA
P1 = 0
P2 = 0
# END EDITING HERE

class BernoulliArmTask2:
    def __init__(self, p):
        self.p = p

    def pull(self, num_pulls=None):
        return np.random.binomial(1, self.p, num_pulls)

class BernoulliBanditTask2:
    def __init__(self, probs=[0.3, 0.5, 0.7],):
        self.__arms = [BernoulliArmTask2(p) for p in probs]
        self.__max_p = max(probs)
        self.__regret = 0

    def pull(self, index):
        reward = self.__arms[index].pull()
        self.__regret += self.__max_p - reward
        return reward

    def regret(self):
        return self.__regret
  
    def num_arms(self):
        return len(self.__arms)


def single_sim_task2(seed=0, ALGO=UCB, P1=P1, P2=P2, HORIZON=HORIZON):
    np.random.seed(seed)
    bandit = BernoulliBanditTask2(probs=[P1, P2])
    algo_inst = ALGO(num_arms=2, horizon=HORIZON)
    for t in range(HORIZON):
        arm_to_be_pulled = algo_inst.give_pull()
        reward = bandit.pull(arm_to_be_pulled)
        algo_inst.get_reward(arm_index=arm_to_be_pulled, reward=reward)
    return bandit.regret()



def simulate_task2(algorithm, P1_values, P2_values, horizon, num_sims=50):
    regrets = []
    for P1, P2 in zip(P1_values, P2_values):
        regret_sum = 0
        for _ in range(num_sims):
            regret_sum += single_sim_task2(ALGO=algorithm, P1=P1, P2=P2, HORIZON=horizon)
        regrets.append(regret_sum / num_sims)
    return regrets
     

def task2(algorithm, horizon, p1s, p2s, num_sims=50):
    """generates the data for task2
    """
    probs = [[p1s[i], p2s[i]] for i in range(len(p1s))]

    regrets = []
    for prob in probs:
        regrets.append(simulate_task2(algorithm, prob, horizon, num_sims))

    return regrets

if __name__ == '__main__':
  # EXAMPLE CODE
    algorithm_ucb = UCB
    regrets_ucb = simulate_task2(algorithm_ucb, P1_values, P2_values, HORIZON)
    
    # Run the simulation with KL-UCB algorithm
    algorithm_kl_ucb = KL_UCB
    regrets_kl_ucb = simulate_task2(algorithm_kl_ucb, P1_values, P2_values, HORIZON)
    
    #Plot the results for UCB
    plt.figure(figsize=(10, 6))
    plt.plot(P2_values, regrets_ucb, marker='o', linestyle='-', color='b', label='UCB')
    plt.title('Regret vs. p2 for UCB Algorithm')
    plt.xlabel('p2')
    plt.ylabel('Regret')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot the results for KL-UCB
    plt.figure(figsize=(10, 6))
    plt.plot(P2_values, regrets_kl_ucb, marker='o', linestyle='-', color='r', label='KL-UCB')
    plt.title('Regret vs. p2 for KL-UCB Algorithm')
    plt.xlabel('p2')
    plt.ylabel('Regret')
    plt.grid(True)
    plt.legend()
    plt.show()