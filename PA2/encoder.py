import numpy as np
import os
import sys

def policydata(policyfile):
    statearr = []
    probarr = []
    statedict = dict()
    numStates = -1
    with open(policyfile) as file1:
        line = file1.readlines()
        for lines in line: 
            numStates += 1
            if lines.split()[0] == '1':
                player = 1
            elif lines.split()[0] == '2':
                player = 2
            else: 
                statedict[lines.split()[0]] = [float(p) for p in lines.split()[1:]]
    return player, statedict 

def statedata(statefile):
    with open(statefile) as file1:
        line = file1.readlines()
        playerstatearr = []
        for lines in line:
            playerstatearr.append((lines.split()[0])) 
    return playerstatearr



def isWin(state, player):
    if state[0] == str(3-player) and state[1] == str(3-player) and state[2] == str(3-player):
        return True
    elif state[3] == str(3-player) and state[4] == str(3-player) and state[5] == str(3-player):
        return True
    elif state[6] == str(3-player) and state[7] == str(3-player) and state[8] == str(3-player):
        return True
    elif state[0] == str(3-player) and state[3] == str(3-player) and state[6] == str(3-player):
        return True
    elif state[1] == str(3-player) and state[4] == str(3-player) and state[7] == str(3-player):
        return True
    elif state[2] == str(3-player) and state[5] == str(3-player) and state[8] == str(3-player):
        return True
    elif state[0] == str(3-player) and state[4] == str(3-player) and state[8] == str(3-player):
        return True
    elif state[2] == str(3-player) and state[4] == str(3-player) and state[6] == str(3-player):
        return True
    return False
    
def isFull(state):
    for i in range(9):
        if state[i] == '0':
            return False
    return True


if __name__ == "__main__":
    global policyfile
    policyfile = sys.argv[2]
    # player, statearr, probarr = policydata(policyfile)
    oppplayer, oppstatedict = policydata(policyfile)
    # print('120000000' in oppstatedict)
    statefile = sys.argv[4]
    statearr = statedata(statefile)
    initialstates = len(statearr)
    # print(probarr)
    # check = 0
    transitions = []
    termstates = []
    upcounter = 0
    downcounter = []
    for i in range(len(statearr)):
        currentState = (statearr[i])
        # print(currentState)
        # nextstates = giveNextState(statearr,currentState)
        for a in range(9):
            # print(str(currentState))
            if (currentState)[a] == '0':
                oppstate = list(currentState)
                # print(oppstate[a])
                oppstate[a] = str(3-oppplayer)
                # print(oppstate[a])
                oppstate = ''.join(oppstate)
                if isFull(oppstate) or isWin(oppstate, oppplayer):
                    if oppstate not in statearr:
                        statearr.append(oppstate)
                        termstates.append(oppstate)
                        transitions.append([i, a, statearr.index(oppstate), 0.0, 1.0])
                        upcounter += 1
                else:
                # print(oppstate)
                # print(oppstatedict)
                # if (oppstate in oppstatedict):
                    # if oppstate not in termstates: 
                        # termstates.append(oppstate)
                    probs = oppstatedict[oppstate]
                    # print(probs)
                    for i in range(9):
                        if float(probs[i]) > 1e-7:
                            sdash = list(oppstate)
                            sdash[i] = str(oppplayer)
                            sdash = ''.join(sdash)
                            # print(sdash)
                            reward = 0
                            if isFull(sdash):
                                if sdash not in termstates:
                                    statearr.append(sdash)
                                    termstates.append(sdash)
                                    transitions.append([[i, a, statearr.index(sdash), 0.0, float(probs[i])]])
                                    # downcounter.append[sdash]
                            elif isWin(sdash, 3-oppplayer):
                                if sdash not in statearr:
                                    statearr.append(sdash)
                                    termstates.append(sdash)
                                    transitions.append([i, a, statearr.index(sdash), 1.0, float(probs[i])])
                                    # downcounter.append(sdash)
                                # reward = 1
                            else:
                                transitions.append([i, a, statearr.index(sdash), 0.0, float(probs[i])])
                                # downcounter.append(sdash)
                                # statearr.append(sdash)
    
    print("numStates", len(statearr))
    print("numActions 9")
    print("end", end=' ')
    for i in range(initialstates, len(statearr)):
        print(int(statearr[i]),end=' ')
    for tra in transitions:
        print("transition", end=' ')
        print(int(tra[0]), int(tra[1]), int(tra[2]), float(tra[3]), float(tra[4]))        
    print("mdptype episodic")
    print("discount 0.999")
    # print(len(statearr))
    # print(upcounter)
    # print(len(downcounter))
                
                # print(oppstate)
                # check += 1

    # print(check)
        
        
        # for sdash in nextstates:
        #     diff = str(sdash - currentState)
        #     # print(diff)
        #     for i in range(len(diff)):
        #         if diff[i] == str(3 - oppplayer):
        #             a = 9 - len(diff) + i3
        #             # print(a)
        #         if diff[i] == str(oppplayer):
        #             b = 9 - len(diff) + i
        #             # print("b", b)
        #     oppstate = currentState + (3-oppplayer)*(10**a)
        #     print(oppstate)
        #     stateindex = np.where(statearr == str(currentState))
        #     oppstateindex = np.where(oppstatearr == str(oppstate))
        #     # print(stateindex[0], oppstateindex[0])
        #     probability = oppprobarr[oppstateindex[0], b]
        #     # print(probability)

             
            
    # checker(120000012, playerstatearr)