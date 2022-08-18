import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import sys
import os

class epsilonGreedy:
    def __init__(self, numarms, horizon, epsilon, arr, file1, path):
        self.numarms = numarms
        self.epsilon = epsilon
        self.horizon = horizon
        self.n = 0
         
        self.cumreward = 0          
        # self.arr = np.array(arr)
        self.maxarr = []
        # sel 
        # np.random.see)
        vals = []
        self.numarms = 0
        lines = file1.readlines()
        for l in lines:
            self.numarms += 1
            vals.append(float(l.strip()))
            # print(l.strip())
        self.arr = np.array(vals)
        self.arm_count = np.zeros(self.numarms)
        # self.total_mean_reward = 0
        # self.reward = np.zeros(horizon)
        self.armrewards = np.zeros(self.numarms)
        self.arm_mean_reward = np.ones(self.numarms)
        
    def pullarm(self, p):
        if (np.random.rand() <= p):
            return 1.
        else:
            return 0.
    
    def calculate(self):
        
        # for i in range(self.numarms):
        #     reward = self.pullarm(self.arr[i])
        #     self.arm_mean_reward[i] = reward

        for i in range(self.horizon):
            decision = np.random.choice([0,1], p=[self.epsilon, 1-self.epsilon])
            if decision == 0:
                armnum = np.random.choice(np.arange(0,self.numarms))
                # print(armnum)
            else:
                for i in range(self.numarms):
                    if self.arm_mean_reward[i]==max(self.arm_mean_reward):
                        self.maxarr.append(i)
                armnum = np.random.choice(self.maxarr)
            currentreward = self.pullarm(self.arr[armnum]) #np.random.choice(np.arange(0,2), p=[1-self.arr[armnum], self.arr[armnum]])
            self.n += 1
            self.arm_count[armnum] += 1
            self.armrewards[armnum] += currentreward
            # self.total_mean_reward = self.total_mean_reward + self.armrewards[armnum] / self.arm_count[armnum]
            self.arm_mean_reward[armnum] = self.armrewards[armnum]/self.arm_count[armnum] #self.arm_mean_reward[armnum]*self.arm_count[armnum] + currentreward) / (self.arm_count[armnum]+1)
            self.maxmean = max(self.arr)
            self.cumreward += currentreward
            # self.reward[i-1] = self.total_mean_reward  
              

class ucb:
    def __init__(self, numarms, horizon, arr, file1, path):
        # Number of arms
        self.numarms = numarms
        # Number of iterations
        self.horizon = horizon
        # Step count
        self.n = 0
        # scale
        # self.scale = scale
        # Step count for each arm
                  
        
        # np.random.see)
        vals = []
        self.numarms = 0
        lines = file1.readlines()
        for l in lines:
            self.numarms += 1
            vals.append(float(l.strip()))
            # print(l.strip())
        self.arr = np.array(vals)
        self.arm_count = np.ones(self.numarms)
        # Total mean reward
        self.total_mean_reward = 0
        # Reward at each step
        self.armrewards = np.zeros(self.numarms)
        self.cumreward = 0
        # Mean reward for each arm
        self.arm_mean_reward = np.zeros(self.numarms) 
        
    def pullarm(self, p):
        if (np.random.rand() <= p):
            return 1.
        else:
            return 0.
    
    def calculate(self):
        for i in range(self.numarms):
            reward = self.pullarm(self.arr[i])
            self.arm_mean_reward[i] = reward

        for i in range(self.numarms, self.horizon):
            # Select action according to UCB Criteria
            # maxes = []
            # for p in range(self.numarms):
            #         if self.betas[p]==max(self.ucb_list):
            #             maxes.append(p)
            # armnum = np.random.choice(maxes)
            armnum = np.argmax(self.arm_mean_reward + np.sqrt((2*np.log(i+1)) / self.arm_count))
            currentreward = self.pullarm(self.arr[armnum]) #np.random.choice(np.arange(0,2), p=[1-self.arr[armnum], self.arr[armnum]])
            # Update counts
            # self.n += 1
            self.arm_count[armnum] += 1
            self.armrewards[armnum] += currentreward
            self.cumreward += currentreward
            # Update total
            # self.total_mean_reward = self.total_mean_reward + self.armrewards[armnum] / self.arm_count[armnum]
            # Update results for arm_mean_reward
            self.arm_mean_reward[armnum] = self.armrewards[armnum] / (self.arm_count[armnum]) #self.arm_mean_reward[armnum] + (currentreward - self.arm_mean_reward[armnum]) / self.arm_count[armnum]
            self.maxmean = max(self.arr)
            # self.reward[i] = self.total_mean_reward
        

class kl_ucb:
    def __init__(self, numarms, horizon, arr, file1, path):
        # Number of arms
        self.numarms = numarms
        # Number of iterations
        self.horizon = horizon
        # Step count
        self.n = 0
        # Step count for each arm
        self.arm_count = np.ones(numarms)
        # Total mean reward
        self.total_mean_reward = 0
        self.reward = np.zeros(horizon)
        self.cumreward = 0
        self.armrewards = np.zeros(numarms)
        # Mean reward for each arm
        self.arm_mean_reward = np.zeros(numarms)           
        # self.arr = np.array(arr)
        self.ucb_list = np.zeros(numarms)
        # np.random.see)
        vals = []
        self.numarms = 0
        lines = file1.readlines()
        for l in lines:
            self.numarms += 1
            vals.append(float(l.strip()))
            # print(l.strip())
        self.arr = np.array(vals)
        self.arm_count = np.ones(self.numarms)
        # Total mean reward
        self.total_mean_reward = 0
        self.reward = np.zeros(horizon)
        self.cumreward = 0
        self.armrewards = np.zeros(self.numarms)
        # Mean reward for each arm
        self.arm_mean_reward = np.zeros(self.numarms)           
        # self.arr = np.array(arr)
        self.ucb_list = np.zeros(self.numarms)

    def pullarm(self, p):
        if (np.random.rand() <= p):
            return 1.
        else:
            return 0.
    
    def KLDivergence(self, x, y):
        if (x == 0):
            x = 0.0000001
        if (x == 1):
            x = 0.9999999
        return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))

    def find_max_q(self, minv, maxv, u, t):
        # q = 0
        # np.seterr(divide = 'ignore')
        RHS = (np.log(t+1) + 3 * np.log(np.log(t+1))) / u
        i = 0.1
        ret = []
        while (i < 1):
            checkval = (minv+maxv)/2
            LHS = self.KLDivergence(minv, i)
            if (LHS <= RHS):
                minv = checkval
            else:
                maxv = checkval
            if checkval == minv:
                break
            i += 0.05
        return minv

    def calculate(self):
        # Select action according to KL-UCB Criteria
        
        for i in range(self.numarms):
            reward = self.pullarm(self.arr[i])
            self.arm_mean_reward[i] = reward 
        
        for i in range(self.numarms, self.horizon):
            if self.n < self.numarms:
                armnum = self.n
            else:
                for i in range(1,self.numarms):
                    minval = self.arm_mean_reward[i]
                    maxval = 1
                    maxes = []
                    self.ucb_list[i] = self.find_max_q(minval, maxval, self.arm_count[i], i)
                    if self.ucb_list[p]==max(self.ucb_list):
                        maxes.append(p)
                armnum = np.random.choice(maxes)
                # picking the arm with maximum KL-UCB
                # armnum = np.argmax(self.ucb_list)
                
            currentreward = self.pullarm(self.arr[armnum]) #np.random.choice(np.arange(0,2), p=[1-self.arr[armnum], self.arr[armnum]])
            self.n += 1
            self.arm_count[armnum] += 1
            self.armrewards[armnum] += currentreward
            # self.total_mean_reward = self.total_mean_reward + (self.armrewards[armnum]) / self.n
            self.arm_mean_reward[armnum] = self.armrewards[armnum]/self.arm_count[armnum] #self.arm_mean_reward[armnum] = self.arm_mean_reward[armnum] + (self.armrewards[armnum] - self.arm_mean_reward[armnum]) / self.arm_count[armnum]
            self.maxmean = max(self.arr)
            self.cumreward += currentreward
            # self.reward[i] = self.total_mean_reward
            

class thompson:
    def __init__(self, numarms, horizon, arr, file1, path):
        self.numarms = numarms
        self.horizon = horizon
        self.n = 0
        # np.random.see)

        vals = []
        self.numarms = 0
        lines = file1.readlines()
        for l in lines:
            self.numarms += 1
            vals.append(float(l.strip()))
            # print(l.strip())
        self.arr = np.array(vals)

        self.arm_count = np.ones(self.numarms)
        self.total_mean_reward = 0
        self.reward = np.zeros(horizon)
        self.cumreward = 0
        self.arm_mean_reward = np.zeros(self.numarms)           
        # self.arr = np.array(arr)
        self.armrewards = np.zeros(self.numarms)
        self.success = np.zeros(self.numarms)
        self.failures = np.zeros(self.numarms)
        self.betas = np.zeros(self.numarms)

    def pullarm(self, p):
        if (np.random.rand() <= p):
            return 1.
        else:
            return 0.
    
    def calculate(self):
        for i in range(self.numarms):
            reward = self.pullarm(self.arr[i])
            self.arm_mean_reward[i] = reward 
        
        for i in range(self.numarms, self.horizon):
            if self.n < self.numarms:
                armnum = self.n
            else : 
                for i in range(self.numarms):
                    self.betas[i] = np.random.beta(self.success[i]+1, self.failures[i]+1) 
                maxes = []
                for p in range(self.numarms):
                        if self.betas[p]==max(self.betas):
                            maxes.append(p)
                armnum = np.random.choice(maxes)
                # armnum = np.argmax([np.random.beta(self.success[i]+1, self.failures[i]+1) for i in range(self.numarms)])
            
            currentreward = self.pullarm(self.arr[armnum]) #np.random.choice(np.arange(0,2), p=[1-self.arr[armnum], self.arr[armnum]])
            self.success[armnum] += currentreward
            if (currentreward == 0):
                self.failures[armnum] += 1
            else:
                # success occurs with reward 1
                self.success[armnum] += 1
            self.n += 1
            self.arm_count[armnum] += 1
            self.armrewards[armnum] += currentreward
            self.cumreward += currentreward
            # self.total_mean_reward = self.total_mean_reward + (self.armrewards[armnum]) / self.n
            self.arm_mean_reward[armnum] = self.armrewards[armnum]/self.arm_count[armnum] #self.arm_mean_reward[armnum] = self.arm_mean_reward[armnum] + (self.armrewards[armnum] - self.arm_mean_reward[armnum]) / self.arm_count[armnum]
            self.maxmean = max(self.arr)
            # self.reward[i] = self.total_mean_reward   


class ucb_t2:
    def __init__(self, numarms, randomSeed, horizon, arr, scale, file1, path):
        # Number of arms
        self.numarms = numarms
        # Number of iterations
        self.horizon = horizon
        # Step count
        self.n = 0
        # scale
        self.scale = scale
        # Step count for each arm          
        # self.arr = np.array(arr)

        vals = []
        self.numarms = 0
        lines = file1.readlines()
        for l in lines:
            self.numarms += 1
            vals.append(float(l.strip()))
            # print(l.strip())
        self.arr = np.array(vals)
        self.arm_count = np.ones(self.numarms)
        # Total mean reward
        self.total_mean_reward = 0
        # Reward at each step
        self.armrewards = np.zeros(self.numarms)
        self.cumreward = 0
        # Mean reward for each arm
        self.arm_mean_reward = np.zeros(self.numarms) 
        np.random.seed(randomSeed)
        
    def pullarm(self, p):
        if (np.random.rand() <= p):
            return 1.
        else:
            return 0.
    
    def calculate(self):
        for i in range(self.numarms):
            reward = self.pullarm(self.arr[i])
            self.arm_mean_reward[i] = reward

        for i in range(self.numarms, self.horizon):
            # Select action according to UCB Criteria
            # maxes = []
            # for p in range(self.numarms):
            #         if self.betas[p]==max(self.ucb_list):
            #             maxes.append(p)
            # armnum = np.random.choice(maxes)
            armnum = np.argmax(self.arm_mean_reward + np.sqrt((self.scale*np.log(i+1)) / self.arm_count))
            currentreward = self.pullarm(self.arr[armnum]) #np.random.choice(np.arange(0,2), p=[1-self.arr[armnum], self.arr[armnum]])
            # Update counts
            # self.n += 1
            self.arm_count[armnum] += 1
            self.armrewards[armnum] += currentreward
            self.cumreward += currentreward
            # Update total
            # self.total_mean_reward = self.total_mean_reward + self.armrewards[armnum] / self.arm_count[armnum]
            # Update results for arm_mean_reward
            self.arm_mean_reward[armnum] = self.armrewards[armnum] / (self.arm_count[armnum]) #self.arm_mean_reward[armnum] + (currentreward - self.arm_mean_reward[armnum]) / self.arm_count[armnum]
            self.maxmean = max(self.arr)
            # self.reward[i] = self.total_mean_reward


class thompson_modified:
    def __init__(self, numarms, horizon, arr, file1, path):
        self.numarms = numarms
        self.horizon = horizon
        self.n = 0
        

        self.arr = []
        self.numarms = 0
        with open(path) as file1:
            lines = file1.readlines()
        # print(lines)
        self.arr = [[float(i) for i in data2.split()] for data2 in lines]
        self.arr = np.array(self.arr)
        self.numarms = len(self.arr[1:])
        self.arm_count = np.ones(self.numarms)
        self.total_mean_reward = 0
        self.reward = np.zeros(horizon)
        self.cumreward = self.numarms
        self.arm_mean_reward = np.ones(self.numarms)           
        # self.arr = np.array(arr[0])
        # print([self.arr])
        self.probs = np.array(self.arr[1:])
        self.armrewards = np.ones(self.numarms)
        self.success = np.ones(self.numarms)
        self.failures = np.zeros(self.numarms)
        self.betas = np.zeros(self.numarms)
        # np.random.see)

    def pullarm(self, arm):
        # numvals = len(p[0])
        return np.random.choice(self.arr[0], p=arm)

    
    def calculate(self):
        # for i in range(self.numarms):
        #     reward = self.pullarm(self.probs[i])
        #     self.arm_mean_reward[i] = reward 
        
        for i in range(self.horizon):
            for i in range(self.numarms):
                self.betas[i] = np.random.beta(self.success[i]+1, self.failures[i]+1) 
            maxes = []
            for p in range(self.numarms):
                    if self.betas[p]==max(self.betas):
                        maxes.append(p)
            armnum = np.random.choice(maxes)
            # armnum = np.argmax([np.random.beta(self.success[i]+1, self.failures[i]+1) for i in range(self.numarms)])
        
            currentreward = self.pullarm(self.probs[armnum]) #np.random.choice(np.arange(0,2), p=[1-self.arr[armnum], self.arr[armnum]])
            self.success[armnum] += currentreward
            self.failures[armnum] += 1 - currentreward
            # self.n += 1
            self.arm_count[armnum] += 1
            self.armrewards[armnum] += currentreward
            self.cumreward += currentreward
            # self.total_mean_reward = self.total_mean_reward + (self.armrewards[armnum]) / self.n
            self.arm_mean_reward[armnum] = self.armrewards[armnum]/self.arm_count[armnum] #self.arm_mean_reward[armnum] = self.arm_mean_reward[armnum] + (self.armrewards[armnum] - self.arm_mean_reward[armnum]) / self.arm_count[armnum]
            self.maxmean = max(self.probs*(self.arr))
            # self.reward[i] = self.total_mean_reward   


class alg_t4:

    def __init__(self, numarms, horizon, arr, threshold, file1, path):
        self.numarms = numarms
        self.horizon = horizon
        self.n = 0
        
        self.prob = 0
        self.probarr = []

        self.arr = []
        numarms = 0
        with open(path) as file1:
            lines = file1.readlines()
        # print(lines)
        self.arr = [[float(i) for i in data2.split()] for data2 in lines]
        self.arr = np.array(self.arr)
        self.numarms = len(self.arr[1:])
        # np.random.see)
        self.arm_count = np.ones(self.numarms)
        self.total_mean_reward = 0
        self.reward = np.zeros(horizon)
        self.cumreward = 0
        self.arm_mean_reward = np.ones(self.numarms)           
        # self.arr = np.array(arr[0])
        self.threshold = threshold
        # print([self.arr])
        self.probs = np.array(self.arr[1:])
        # print(self.probs)
        self.armrewards = np.zeros(self.numarms)
        self.success = np.zeros(self.numarms)
        self.failures = np.zeros(self.numarms)
        self.betas = np.zeros(self.numarms)

    def pullarm(self, arm):
        # numvals = len(p[0])
        if np.random.choice(self.arr[0], p=arm) > self.threshold:
            return 1
        else :
            return 0
        # return 1 if np.random.choice(self.arr, p=arm) > self.threshold else 0

    
    def calculate(self):
        # for i in range(self.numarms):
        #     reward = self.pullarm(self.probs[i])
        #     self.arm_mean_reward[i] = reward 
        
        for i in range(self.horizon):
            for i in range(self.numarms):
                self.betas[i] = np.random.beta(self.success[i]+1, self.failures[i]+1) 
            maxes = []
            for p in range(self.numarms):
                    if self.betas[p]==max(self.betas):
                        maxes.append(p)
            armnum = np.random.choice(maxes)
            # armnum = np.argmax([np.random.beta(self.success[i]+1, self.failures[i]+1) for i in range(self.numarms)])

            currentreward = self.pullarm(self.probs[armnum]) #np.random.choice(np.arange(0,2), p=[1-self.arr[armnum], self.arr[armnum]])
            self.success[armnum] += currentreward
            self.failures[armnum] += 1 - currentreward
            # self.n += 1
            for j in range(self.numarms):
                self.prob = 0
                for i in range(len(self.arr)):
                    if self.probs[j][i]>self.threshold:
                        self.prob += self.probs[j][i]
                self.probarr.append(self.prob)
            # print(self.probarr[0])
            self.arm_count[armnum] += 1
            self.armrewards[armnum] += currentreward
            # self.total_mean_reward = self.total_mean_reward + (self.armrewards[armnum]) / self.n
            self.arm_mean_reward[armnum] = self.armrewards[armnum]/self.arm_count[armnum] #self.arm_mean_reward[armnum] = self.arm_mean_reward[armnum] + (self.armrewards[armnum] - self.arm_mean_reward[armnum]) / self.arm_count[armnum]
            self.maxmean = max(self.probarr)
            # self.reward[i] = self.total_mean_reward


def run_algorithm(vals, algorithm, epsilon, horizon, numarms, scale, threshold, file1, path):
    # np.random.see)
    if algorithm=="epsilon-greedy-t1":
        epg = epsilonGreedy(numarms, horizon, epsilon, vals, file1, path)
        epg.calculate()
        return epg.horizon*epg.maxmean - (epg.cumreward), 0
    elif algorithm=="ucb-t1":
        ucbt1 = ucb(numarms, horizon, vals, file1, path)
        ucbt1.calculate()
        return ucbt1.horizon*ucbt1.maxmean - (ucbt1.cumreward), 0
    elif algorithm=="kl-ucb-t1":
        klucbt1 = kl_ucb(numarms, horizon, vals, file1, path)
        klucbt1.calculate()
        return klucbt1.horizon*klucbt1.maxmean - (klucbt1.cumreward), 0
    elif algorithm=="thompson-sampling-t1":
        thompsont1 = thompson(numarms, horizon, vals, file1, path)
        thompsont1.calculate()
        return thompsont1.horizon*thompsont1.maxmean - (thompsont1.cumreward), 0
    elif algorithm=="ucb-t2":
        ucbt2 = ucb_t2(numarms, randomSeed, horizon, vals, scale, file1, path)
        ucbt2.calculate()
        return ucbt2.horizon*ucbt2.maxmean - (ucbt2.cumreward), 0
    elif algorithm=="alg-t3":
        thompsont3 = thompson_modified(numarms, horizon, vals, file1, path)
        thompsont3.calculate()
        return thompsont3.horizon*thompsont3.maxmean - (thompsont3.cumreward), 0
    elif algorithm=="alg-t4":
        thompsont4 = alg_t4(numarms, horizon, vals, threshold, file1, path)
        thompsont4.calculate()
        return thompsont4.horizon*thompsont4.maxmean - (thompsont4.cumreward), sum(thompsont4.success)

def run_file(instance, algorithm, epsilon, horizon, seed, scale, threshold):
    instance_path = instance[3:]
    # print(instance_path)
    abs_path = os.path.abspath(__file__)  
    present_dir = os.path.dirname(abs_path)
    parent_dir = os.path.dirname(present_dir)
    path = os.path.join(parent_dir, instance_path)

    file1 = open(path,"r")
    file2 = open("/home/harshil/cs747/cs747-pa1-v1/submission/Output.txt", "a")
    vals = 0
    numarms = 0
    
    regret, highs = run_algorithm(vals, algorithm, epsilon, horizon, numarms, scale, threshold, file1, path)
    # file2.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(instance, algorithm, seed, epsilon, scale, threshold, horizon, regret, highs))
    print("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(instance, algorithm, seed, epsilon, scale, threshold, horizon, regret, highs))
    return regret


if __name__=='__main__':
    
    instance = sys.argv[2]
    algorithm = sys.argv[4]
    randomSeed = int(sys.argv[6])
    epsilon = float(sys.argv[8])
    scale = float(sys.argv[10])
    threshold = float(sys.argv[12])
    horizon = int(sys.argv[14])
    regret = run_file(instance, algorithm, epsilon, horizon, randomSeed, scale, threshold)