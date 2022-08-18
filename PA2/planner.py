import numpy as np
import os
import sys
from pulp import *

def mdpdata(mdp):
    file1 = open(mdp,"r")
    with open(mdp) as file1:
        line = file1.readlines()
        for lines in line:    
            if lines.split()[0] == "numStates":
                S = int(lines.split()[1])
            elif lines.split()[0] == "numActions":
                A = int(lines.split()[1])
                rewards = np.zeros((S, A, S), dtype=np.float64)
                probs = np.zeros((S, A, S), dtype=np.float64)
            elif lines.split()[0] == "end":
                end = lines.split()[1:]
            elif lines.split()[0] == "transition":
                rewards[int(lines.split()[1])][int(lines.split()[2])][int(lines.split()[3])] = float(lines.split()[4])
                probs[int(lines.split()[1])][int(lines.split()[2])][int(lines.split()[3])] = float(lines.split()[5])
            elif lines.split()[0] == "mdptype":
                mdptype = lines.split()[1]
            elif lines.split()[0] == "discount":
                gamma = float(lines.split()[1])
    return S, A, end, rewards, probs, gamma, mdptype

def ActionValueFunction(rewards, probs, gamma, V):
    q = (np.sum(probs*rewards, axis=2) + gamma*probs.dot(V))
    return q

def ValueFunction(rewards, probs, gamma, policy):
    numStates = rewards.shape[0]
    Rp = []
    Tp = []
    for i in range(numStates):
        Tp.append(probs[i, policy[i], :])
    for i in range(numStates):
        Rp.append(rewards[i, policy[i], :])
    Tp = np.array(Tp)
    Rp = np.array(Rp)
    A = np.identity(numStates) - gamma*Tp
    B = np.sum(Tp*Rp, axis = 1)
    V = np.linalg.pinv(A).dot(B)
    return V

def ValueIteration(rewards, probs, gamma):
    V = np.zeros(probs.shape[0])
    delta = 1
    while True:
        if delta < 1e-12:
            break
        q = (np.sum(probs*rewards, axis=2) + gamma*probs.dot(V))
        newV = np.amax(q, axis=1)
        delta = np.max(newV - V)
        V = newV.copy()
        policy = np.argmax(q, axis = 1)
    return V, policy


def PolicyIteration(rewards, probs, gamma):
    pi = np.zeros((probs.shape[0]), dtype = int)
    while True: 
        v = ValueFunction(rewards, probs, gamma, pi)
        q = np.sum(probs*rewards, axis=2) + gamma*probs.dot(v)
        newV = np.max(q, axis = 1)
        if np.linalg.norm(newV - v) < 1e-12:
            break
        else:
            newpi = np.argmax(q, axis = 1) 
        if (newpi==pi).all():
            break
        pi = newpi.copy()
    return v, pi


def LP(probs, rewards, gamma):
    prob = LpProblem("MDP", LpMinimize)
    Vi = pulp.LpVariable.dicts("Vi", range(probs.shape[0]))
    prob += pulp.lpSum([Vi[i] for i in range(probs.shape[0])])
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            prob += pulp.lpSum([probs[i, j, k]*rewards[i, j, k] + gamma*probs[i, j, k]*Vi[k] for k in range(probs.shape[0])]) <= Vi[i]
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    v = np.zeros(probs.shape[0])
    for i in range(probs.shape[0]):
    	v[i] = pulp.value(Vi[i])
    return v

if __name__ == "__main__":
    global mdp, algorithm
    mdp = sys.argv[2]
    if len(sys.argv) >= 4:
        algorithm = sys.argv[4]
    elif len(sys.argv) < 4:
        algorithm = "vi"
    S, A, end, rewards, probs, gamma, mdptype = mdpdata(mdp)
    if algorithm == 'vi':
        v, policy = ValueIteration(rewards, probs, gamma)
    if algorithm == 'hpi':
        v, policy = PolicyIteration(rewards, probs, gamma)
    if algorithm == 'lp':
        v = LP(probs, rewards, gamma)
        q = (np.sum(probs*rewards, axis=2) + gamma*probs.dot(v))
        policy = np.argmax(q, axis = 1)
    for i in range(probs.shape[0]):
        print('{} {}'.format(v[i].round(6), int(policy[i])))