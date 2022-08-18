import numpy as np
import os
import sys

statefile1 = 
statefile2 = 

with open(statefile1) as file1:
        line = file1.readlines()
        playerstatearr1 = []
        for lines in line:
            playerstatearr1.append((lines.split()[0]))

with open(statefile2) as file1:
        line = file1.readlines()
        playerstatearr2 = []
        for lines in line:
            playerstatearr2.append((lines.split()[0]))

