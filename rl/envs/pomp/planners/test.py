from __future__ import print_function,division
from six import iteritems
from builtins import range

from .profiler import Profiler
import time
import numpy as np

# def testPlanner(planner,numTrials,maxTime,filename):    

def testPlanner(problem,numTrials,maxTime,filename, plannerType, **plannerParams):    
    print("Testing planner for %d trials, %f seconds"%(numTrials,maxTime))
    print("Saving to",filename)
    # f = open(filename,'w')
    # f.write("trial,plan iters,plan time,best cost\n")
    successes = 0
    costs = []
    for trial in range(numTrials):
        print()
        print("Trial",trial+1)# 
        planner = problem().planner(plannerType,**plannerParams)
        planner.reset()
        curCost = float('inf')
        t0 = time.time()
        numupdates = 0
        iters = 0
        hadException = False
        # while time.time()-t0 < maxTime:
        # while time.time()-t0 < maxTime and curCost == float('inf'):
    
        planner.planMore(maxTime)
        iters += maxTime
        if planner.bestPathCost != None and planner.bestPathCost != curCost:
            numupdates += 1
            curCost = planner.bestPathCost
            t1 = time.time()
            # f.write(str(trial)+","+str(iters)+","+str(t1-t0)+","+str(curCost)+'\n')

        if hasattr(planner,'stats'):
            print
            temp = Profiler()
            temp.items["Stats:"] = planner.stats
            temp.pretty_print()

        if curCost < float('inf'):
            costs.append(curCost)
            successes += 1

        print()
        print("Final cost:",curCost)
        print()

        # f.write(str(trial)+","+str(iters)+","+str(maxTime)+","+str(curCost)+'\n')

    total_cost = sum(costs)
    if successes > 0:
        ave_cost = total_cost/successes
    else: 
        ave_cost = float('inf')
    print('Success rate: ' + str(successes/numTrials))
    # print('Average cost: ' + str(ave_cost))
    print('Average cost: ' + str(np.mean(costs)))
    print('Cost STD: ' + str(np.std(costs)))
    print('Cost CI: ' + str(2*np.std(costs)/successes**.5))


    # f.write('Num trials: ' + str(numTrials) + '\n')
    # f.write('Success rate: ' + str(successes/numTrials) + '\n')
    # f.write('Average cost: ' + str(np.mean(costs)) + '\n')
    # f.write('Cost STD: ' + str(np.std(costs)) + '\n')
    # f.write('Cost CI: ' + str(2*np.std(costs)/successes**.5) + '\n')


    # f.close()
    return ave_cost, successes/numTrials
