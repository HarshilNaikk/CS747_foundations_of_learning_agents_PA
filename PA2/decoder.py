import numpy as np
import os
import sys

def statedata(statefile):
    with open(statefile) as file1:
        line = file1.readlines()
        playerstatearr = []
        for lines in line:
            playerstatearr.append((lines.split()[0])) 
    return playerstatearr

def valpolicydata(valpolicyfile):
    with open(valpolicyfile) as file1:
        line = file1.readlines()
        pi = []
        for lines in line:
            pi.append((lines.split()[1])) 
    return pi

if __name__ == "__main__":
    global policyfile
    valpolicyfile = sys.argv[2]
    statefile = sys.argv[4]
    player = sys.argv[6]

    statearr = statedata(statefile)
    pi = valpolicydata(valpolicyfile)
    li = [0.0 for i in range(9)]

    print(player)
    for i in range(len(statearr)):
        print(statearr[i], end=' ')
        li[pi[i]] = 1.0
        for i in li:
            print(float(i), end=' ')
    statefile = sys.argv[4]
    statearr = statedata(statefile)
    initialstates = len(statearr)