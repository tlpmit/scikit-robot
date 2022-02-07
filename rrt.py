import pdb
import math
import numpy as np
import time
from random import random, randrange

DEBUG_RRT = False
DEBUG_RRT_FAIL = False
DEBUG_RRT_TIME = True
DEBUG_RRT_DISPLAY = False

RRT_STEP = 0.025
RRT_INTERPOLATE_STEP_SIZE = 4*RRT_STEP
RRT_TIMEOUT = 60.
RRT_MAX_STOP_NODE_STEPS = 10
RRT_FAIL_ITER = 10
RRT_MAX_ITER = 100

# Robot interface
# robot.random_conf(conf, chains)
# robot.in_workspace(conf)
# robot.dist_conf_abs(qi, qj, near_d)
# robot.step_along_line(q_f, q_i, self.stepSize, forward = self.init)
# robot.conf_check_collision(q)
# robot.interpolate(qf, qi, stepSize=minStep)

class ChainRRT:
    def __init__(self, robot, initConf, goalConf, chains, stepSize):
        if DEBUG_RRT: print('Setting up RRT')
        self.initConf = initConf
        self.robot = robot
        self.chains = chains
        self.Ta = Tree(robot, initConf, True, stepSize)
        if goalConf:
            self.Tb = Tree(robot, goalConf, False, stepSize)
        self.startTime = time.time()

    def randConf(self, tree):
        return self.robot.random_conf(self.initConf, self.chains)

    def swapTrees(self):
        if self.Ta.size > self.Tb.size:
            self.Ta, self.Tb = self.Tb, self.Ta

    # the qx variables denote confs; the nx variable denote nodes
    def buildBiTree(self, K=1000):
        """Builds the RRT and returns either a pair of nodes (one in each tree)
        with a common configuration or FAILURE."""
        if DEBUG_RRT: print('Building BiRRT')
        if self.Ta.root is None:
            print('Collision at initial conf')
            return 'FAILURE'
        if self.Tb.root is None:
            print('Collision at destination conf')
            return 'FAILURE'        
        n_new = self.Ta.stopNode(self.Tb.root.conf, self.Ta.root)
        if eqChains(n_new.conf, self.Tb.root.conf):
            if DEBUG_RRT: print('Found direct path')
            na_new = self.Ta.addNode(n_new.conf, self.Ta.root)
            nb_new = self.Tb.addNode(n_new.conf, self.Tb.root)
            return (na_new, nb_new)
        for i in range(K):
            if DEBUG_RRT: print(i)
            if (time.time() - self.startTime) > RRT_TIMEOUT:
                print('!!! RRT Timeout !!!')
                return 'FAILURE'
            q_rand = self.randConf(self.Ta)
            if DEBUG_RRT:
                print('q_rand', q_rand['base'])
            na_near = self.Ta.nearest(q_rand)
            # adjust continuous angle values
            na_new = self.Ta.stopNode(q_rand, na_near)

            if not na_new is na_near:
                nb_near = self.Tb.nearest(na_new.conf)
                nb_new = self.Tb.stopNode(na_new.conf, nb_near)
                if eqChains(na_new.conf, nb_new.conf):
                    if DEBUG_RRT:
                        print(f'BiRRT: Goal reached in {i} iterations.  Time={time.time() - self.startTime}')
                    return (na_new, nb_new)
            self.swapTrees()
        if DEBUG_RRT:
            print('\nBiRRT: Goal not reached in' + ' %s iterations\n' %str(K))
        return 'FAILURE'

    # the qx variables denote confs; the nx variable denote nodes
    def buildTree(self, goalTest, K=1000):
        """Builds the RRT and returns either a node or FAILURE."""
        if DEBUG_RRT: print('Building RRT')
        if goalTest(self.Ta.root.conf):
            return self.Ta.root
        for i in range(K):
            if DEBUG_RRT:
                if i % 100 == 0: print(i)
            q_rand = self.randConf()
            na_near = self.Ta.nearest(q_rand)
            # adjust continuous angle values
            na_new = self.Ta.stopNode(q_rand, na_near, maxSteps = RRT_MAX_STOP_NODE_STEPS)
            if goalTest(na_new.conf):
                return na_new
        if DEBUG_RRT:
            print('\nRRT: Goal not reached in' + ' %s iterations\n' %str(K))
        return 'FAILURE'

    def tracePath(self, node):
        path = [node]; cur = node
        while cur.parent != None:
            cur = cur.parent
            path.append(cur)
        return path

    def findGoalPath(self, goalTest, K=None):
        node = self.buildTree(goalTest, K)
        if node == 'FAILURE': return 'FAILURE', None
        path = self.tracePath(node)[::-1]
        goalValues = [goalTest(c.conf) for c in path]
        if True in goalValues:
            goalIndex = goalValues.index(True)
            # include up to first True
            return path[:goalIndex+1]
        else:
            print('Empty solution in findGoalPath')
            input('Continue anyway?')
            return 'FAILURE', None
    
    def findPath(self, K=None):
        sharedNodes = self.buildBiTree(K)
        if sharedNodes == 'FAILURE': return 'FAILURE'
        pathA = self.tracePath(sharedNodes[0])
        pathB = self.tracePath(sharedNodes[1])
        if pathA[0].tree.init:
            return (pathA[::-1] + pathB)
        elif pathB[0].tree.init:
            return (pathB[::-1] + pathA)
        else:
            raise Exception("Neither path is marked init")

idnum = 0
class Node:
    def __init__(self, conf, parent, tree):
        global idnum
        self.id = idnum; idnum += 1
        self.conf = conf
        self.parent = parent
        self.tree = tree
    def __str__(self):
        return 'Node:'+str(self.id)
    def __hash__(self):
        return self.id

class Tree:
    def __init__(self, robot, conf, init, stepSize, ignoreNonPermanent=False):
        self.robot = robot
        self.nodes = []
        self.size = 0
        self.init = init
        self.stepSize = stepSize
        self.root = self.addNode(conf, None)

    # Returns None if conf leads to illegal collision
    def addNode(self, conf, parent):
        coll = self.robot.conf_check_collision(conf)
        if coll: return None
        n_new = Node(conf, parent, self)
        self.nodes.append(n_new)
        self.size += 1
        return n_new

    def nearest(self, q):               # q is conf
        robot = self.robot
        near_d = float('inf')
        near_node = None
        for node in self.nodes:
            d = robot.dist_conf_abs(q, node.conf, near_d)
            if not d is None and d < near_d:
                near_d = d
                near_node = node
        return near_node
    
    def stopNode(self, q_f, n_i,
                 maxSteps = 1000):
        q_i = n_i.conf
        if eqChains(q_f, q_i): return n_i
        step = 0

        while True:
            if maxSteps:
                if step >= maxSteps:
                    return n_i
            step += 1
            q_new = self.robot.step_along_line(q_f, q_i, self.stepSize,
                                               forward = self.init)
            n_new = self.addNode(q_new, n_i)
            if n_new:
                if eqChains(q_f, q_new):
                    return n_new
                n_i = n_new
                q_i = n_i.conf
            else:                       # a collision
                return n_i

    def __str__(self):
        return 'TREE:['+str(len(self.size))+']'

def runRRT(robot, initConf, destConf, chains, maxIter, failIter):
    if not (robot.in_workspace(initConf) and \
            robot.in_workspace(destConf)):
        if DEBUG_RRT or DEBUG_RRTFailed:
            print('    RRT failed because conf is out of workspace')
        return None, None
    nodes = 'FAILURE'
    failIter = (failIter if failIter is not None else RRT_FAIL_ITER)
    failCount = -1
    while nodes == 'FAILURE' and failCount < failIter:
        rrti = ChainRRT(robot, initConf, destConf, chains, RRT_INTERPOLATE_STEP_SIZE)
        nodes = rrti.findPath(K = maxIter or RRT_MAX_ITER)
        failCount += 1

    if (DEBUG_RRT or DEBUG_RRT_FAIL) and failCount > 0:
        print('    RRT has failed', failCount, 'times')
        pdb.set_trace()
    if nodes == 'FAILURE':
        return None
    else:
        path = [c.conf for c in nodes]
        print('    RRT path len = ', len(path))
        return path

## Utilities

def smoothPath(robot, path, verbose=False, nsteps = 100, npasses = 10):
    if len(path) < 3:
        return path
    if verbose: print('Path has %s points'%str(len(path)), '... smoothing')
    inp = removeDuplicateConfs(path)
    if len(inp) < 3:
        return path
    checked = set([])
    outer = 0
    count = 0
    step = 0
    robot = path[0].robot
    if verbose: print('Smoothing...')
    while outer < npasses:
        print('  Start smoothing pass', outer, 'len=', len(inp))
        smoothed = []
        for p in inp:
            if not smoothed or not eqChains(smoothed[-1], p):
                smoothed.append(p)
        n = len(smoothed)
        while count < nsteps and n > 2:
            if verbose: print('step', step, ':', end=' ') 
            if n < 1:
                print('smooth', 'Path is empty!')
                return removeDuplicateConfs(path)
            i = randrange(n)
            j = randrange(n)
            if j < i: i, j = j, i 
            step += 1
            if verbose: print(i, j, len(checked))
            if j-i < 2 or \
                (smoothed[j], smoothed[i]) in checked:
                count += 1
                continue
            else:
                checked.add((smoothed[j], smoothed[i]))
            if verbose:
                print('smooth', 'Testing')
            if safeInterpolation(smoothed[j], smoothed[i], robot, verbose):
                count = 0
                if verbose:
                    print('smooth', 'Safe')
                    print('smooth', 'remaining')
                smoothed[i+1:j] = []
                n = len(smoothed)
                if verbose: print('Smoothed path length is', n)
            else:
                count += 1
        outer += 1
        if outer < npasses:
            count = 0
            if verbose: print('Re-expanding path')
            inp = removeDuplicateConfs(robot.interpolate_path(removeDuplicateConfs(smoothed)))
    if verbose:
        print('Final smooth path len =', len(smoothed), 'dist=')

    ans = removeDuplicateConfs(smoothed)
    return ans

minStep = RRT_INTERPOLATE_STEP_SIZE
def safeInterpolation(qf, qi, robot, verbose=False):
    for conf in qf.robot.interpolate(qf, qi, stepSize=minStep):
        coll = robot.conf_check_collision(conf)    
        if coll: return False
    return True

def removeDuplicateConfs(path):
    inp = []
    for p in path:
        if not inp or not inp[-1].nearEqual(p):
            inp.append(p)
    return inp

def eqChains(conf1, conf2):
    return all(all(conf1[c]==conf2[c]) for c in conf1.chain_vals)

print('Loaded rrt2.py')
