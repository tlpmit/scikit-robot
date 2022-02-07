from __future__ import absolute_import
from __future__ import print_function
import math, pdb
import numpy as np
import pickle, bz2, random, operator
from geometry.transformations import euler_matrix, euler_from_matrix, quaternion_from_euler
import geometry.hu as hu
import graphics.windowManager3D as wm
from autil.globals import glob as aglob

import os
try:
    curDir = os.path.dirname(os.path.abspath(__file__))
except:
    curDir = '.'
ikPath = curDir + '/pr2IKData.pkl'  # pickled datastructure
ikPath_compressed = curDir + '/pr2IKData.pbz2'  # pickled datastructure
ikPath_grp_compressed = curDir + '/pr2IKData_GRP.pbz2'  # pickled datastructure

angleStep = math.pi/10
eulerStep = math.pi/20          # fine
xyzStep = 0.05                  # fine
dangle = math.pi/20             # for GRP
dr = 0.05                       # for GRP

def writeIKData_old(robot):
    with open(ikPath, "wb") as f:
        data = [kinScan(robot, h) for h in ('right', 'left')]
        pickle.dump(data, f)

def writeIKData(robot):
    ikd = [kinScan(robot, h) for h in ('right', 'left')]
    _writeIKData(ikd)

def _writeIKData(data, path=ikPath_compressed):
    with bz2.BZ2File(path, "wb") as f:
        pickle.dump(data, f)

def _readIKData(path=ikPath_compressed):
    with bz2.BZ2File(path, "rb") as f:
        return pickle.load(f)
        
cached_ik_data = None
def readIKData_old():
    global cached_ik_data
    if cached_ik_data is None:
        print('Loading IK data...')
        with open(ikPath, "rb") as f:
            cached_ik_data = pickle.load(f)
        print('...done!')
    return cached_ik_data

def readIKData(path=ikPath_compressed):
    global cached_ik_data
    if cached_ik_data is None:
        print('Loading IK data...')
        cached_ik_data = _readIKData(path)
        print('...done!')
    return cached_ik_data

def readIKData_GRP(path=ikPath_grp_compressed):
    global cached_ik_data
    if cached_ik_data is None:
        print('Loading IK data...')
        cached_ik_data = _readIKData(path)
        print('...done!')
    return cached_ik_data

def jointRanges(robot, hand):
    armChainName = robot.armChainNames[hand]
    limits = list(robot.limits([armChainName])) # an iterable of joint limits
    limEps = math.pi/20
    ranges = []
    for thlim in limits:
        fullRange = (thlim[1] - thlim[0] > 2*math.pi-0.001)
        ranges.append(np.arange(thlim[0] if fullRange else thlim[0]+limEps,
                                thlim[1]-limEps,
                                angleStep))
    return ranges

def kinScan(robot, hand):
    jranges = jointRanges(robot, hand)
    initConf = robot.makeConf(0,0,0)
    kinHash = {}
    def kinScanAux(vals, ranges):
        if ranges:
            for thi in range(len(ranges[0])):
                if not vals: print(thi)
                kinScanAux(vals+[thi], ranges[1:])
        else:
            kinScanEntry(initConf, hand, vals, jranges, kinHash)
    kinScanAux([], jranges)
    compactify(kinHash, 0.05, math.pi/30)
    return (jranges, kinHash)

def kinScanEntry(conf, hand, jindices, jranges, kinHash):
    armChainName = conf.robot.armChainNames[hand]
    assert len(jranges) == len(jindices)
    if not all(jindices[i] < len(jranges[i]) for i in range(len(jindices))):
        print(jranges)
        print(jindices)
    angles = [jranges[joint][ival] for joint, ival in enumerate(jindices)]
    c = conf.set(armChainName, angles)
    cart = c.cartConf()
    tr = cart[armChainName]
    pos, euler = tr.matrix[:,3].tolist(), euler_from_matrix(tr.matrix, axes='sxyz')
    z = pos[2]
    if 2. > z > 0.1:
        assert all(-math.pi <= e <= math.pi for e in euler)
        key = IKkey(pos, euler)
        entry = [pos, euler, jindices, None, None]  # verbose!!
        if key in kinHash:
            kinHash[key].append(entry)
        else:
            kinHash[key] = [entry]

def rehash(ikData):
    newData = [None, None]
    for i in range(2):
        kinHash = {}
        (jranges, kinHash_old) = ikData[i]
        for key_old in kinHash_old:
            for entry in kinHash_old[key_old]:
                pos = entry[ke_pos]; euler = entry[ke_euler]
                key = IKkey(pos, euler)
                if key in kinHash:
                    kinHash[key].append(entry)
                else:
                    kinHash[key] = [entry]
        newData[i] = (jranges, kinHash)
    return newData
            
def IKkey(pos, euler):
    return pos[2]//xyzStep, (euler[0]+math.pi)//eulerStep, (euler[1]+math.pi)//eulerStep

def IKkey_neighbors(pos, euler, delta=math.pi/20):
    neighbors = set()
    for e0 in (euler[0], euler[0]+delta, euler[0]-delta):
        for e1 in (euler[1], euler[1]+delta, euler[1]-delta):
            neighbors.add(IKkey(pos, (e0, e1, euler[2])))
    return neighbors

# Verbose representation
ke_pos, ke_euler, ke_anglei, ke_score, ke_inv_trans = range(5)

def get_inv_trans(entry):
    if entry[ke_inv_trans] is None:
        q = quaternion_from_euler(*entry[ke_euler])
        it = hu.Transform(p=np.array([[a] for a in entry[ke_pos]]), q=np.array(q))
        entry[ke_inv_trans] = it.inverse()
    return entry[ke_inv_trans]

def computeScore(entry, hand, jranges, verbose=False):
    # limit_scores = [k==0 or k==len(jranges[j])-1 for j,k in enumerate(entry[ke_anglei]) if not fullRange[hand][j]]
    limit_scores = [0]
    cross = max(entry[ke_pos][1],0) if hand == 'right' else max(-entry[ke_pos][1],0)
    r = math.sqrt(entry[ke_pos][0]**2 + entry[ke_pos][1]**2)
    dist = max(0.5 - r, 0)
    angle = max(abs(entry[ke_euler][2]) - math.pi/2, 0)
    if verbose:
        print('euler', 10*angle, 'limit', 3*max(limit_scores),
              'dist', (5*dist)**2, 'cross', 10*cross)
    return 10*angle + 3*max(limit_scores) + (5*dist)**2 + 10*cross

def computeAllScores(kinHash, hand, jranges):
    for key, entries in kinHash.items():
        total = 0.
        max_score = -1
        for entry in entries:
            score = computeScore(entry, hand, jranges)
            max_score = max(max_score, score)
            entry[ke_score] = score
        total = sum([(max_score - entry[ke_score]) for entry in entries])
        for entry in entries:
            entry[ke_score] = (max_score - entry[ke_score])/total
        entries.sort(key=operator.itemgetter(ke_score))

def filterOnScores(kinHash, scoreThreshold):
    for key in kinHash:
        entries = kinHash[key]
        total = 0.
        for i, entry in enumerate(entries):
            total += entry[ke_score]
            if total > scoreThreshold:
                thri = i; break
        rem_entries = entries[thri:]
        rem_total = sum([entry[ke_score] for entry in rem_entries])
        for entry in rem_entries:
            entry[ke_score] /= rem_total
        kinHash[key] = rem_entries

# Compactify representation
def compactify(kinHash, dxy = 0.1, dtheta = math.pi/20):
    for key in kinHash:
        bases = {}
        for entry in kinHash[key]:
            # Filter...
            r = math.sqrt(entry[ke_pos][0]**2 + entry[ke_pos][1]**2)
            if r <= 0.5: continue
            w = euler_matrix(*entry[ke_euler])
            if w[0,0] < -0.5: continue
            # end filter
            itrans = get_inv_trans(entry)
            params = list(itrans.pose(zthr = 1.0, fail=False).xyztTuple())
            bkey = (params[0]//dxy, params[1]//dxy, params[3]//dtheta)
            entry[ke_inv_trans] = None
            bases[bkey] = entry
        print(key, len(kinHash[key]), '->', len(bases))
        kinHash[key] = list(bases.values())

def gamma_r_phi(kinHash):
    def update(val, v_r):
        v_r[0] = min(val, v_r[0])
        v_r[1] = max(val, v_r[1])
    nhash = {}
    for key in kinHash:
        gamma_r = [1000, -1000]
        r_r = [1000, -1000]
        phi_r = [1000, -1000]
        for entry in kinHash[key]:
            euler = entry[ke_euler]
            pos = entry[ke_pos]
            r = math.sqrt(pos[0]**2 + pos[1]**2)
            phi = math.atan2(pos[1], pos[0])
            gamma = euler[2]
            update(gamma//dangle, gamma_r)
            update(r//dr, r_r)
            update(phi//dangle, phi_r)
        nhash[key] = (gamma_r, r_r, phi_r)
    return nhash

def grp(ikd):
    return [[ikd[0][0], gamma_r_phi(ikd[0][1])],
            [ikd[1][0], gamma_r_phi(ikd[1][1])]]
        
IKData = None
robot = None
fullRange = None

def setup():
    global robot, fullRange
    aglob.usePR2()
    from robot.pr2.pr2Robot import makeRobot
    robot = makeRobot(np.array(aglob.workspace))
    fullRange = {}
    for hand in ('left', 'right'):
        armChainName = robot.armChainNames[hand]
        limits = list(robot.limits([armChainName])) # an iterable of joint limits
        fullRange[hand] = [(thlim[1] - thlim[0] > 2*math.pi-0.001) for thlim in limits]    

hands = ('right', 'left')
def setup_test(ikd = None):
    global IKData
    setup()
    wm.makeWindow('W', aglob.viewPort, aglob.windowSizes.get('W', 600))
    if ikd is None:
        IKData = readIKData()
            
def test(k0=8.0, k1=8.0, k2=8.0, n=10, hand='right'):
    if IKData is None:
        return
    jranges, kentries = IKData[0 if hand=='right' else 1]
    initConf = robot.makeConf(0,0,0)
    armChainName = robot.armChainNames[hand]
    vals = kentries[(k0, k1, k2)]
    for i in range(n):
        wm.getWindow('W').clear()
        entry = random.choice(vals)
        w = euler_matrix(*entry[ke_euler])
        if w[0,0] >= -0.5: continue
        print('w[0,0]', w[0,0])
        angles = [jranges[j][k] for j,k in enumerate(entry[ke_anglei])]
        c = initConf.set(armChainName, angles)
        c.draw('W')
        print(entry[ke_anglei])
        print('euler', entry[ke_euler])
        print('pos', entry[ke_pos])
        q = quaternion_from_euler(*entry[ke_euler])
        it = hu.Transform(p=np.array([[a] for a in entry[ke_pos]]), q=np.array(q))
        print(it.matrix)
        computeScore(entry, hand, jranges, verbose=True)
        print('score', entry[ke_score])
        input('Ok?')

def review(k0=8.0, k1=8.0, k2=8.0, n=10, hand='right'):
    if IKData is None:
        setup_test()
    jranges, kentries = IKData[0 if hand=='right' else 1]
    initConf = robot.makeConf(0,0,0)
    armChainName = robot.armChainNames[hand]
    vals = kentries[(k0, k1, k2)]
    r_phi_for_gamma = {}
    dangle = math.pi/20
    dr = 0.05
    for entry in vals:
        euler = entry[ke_euler]
        pos = entry[ke_pos]
        r = math.sqrt(pos[0]**2 + pos[1]**2)
        if r <= 0.5: continue
        w = euler_matrix(*euler)
        if w[0,0] < -0.5: continue
        phi = math.atan2(pos[1], pos[0])
        gamma = euler[2]
        gk = gamma//dangle
        if gk not in r_phi_for_gamma:
            r_phi_for_gamma[gk] = set([(r//dr, phi//dangle)])
        else:
            r_phi_for_gamma[gk].add((r//dr, phi//dangle))
    print('gamma', len(r_phi_for_gamma))
    print([len(ge) for ge in r_phi_for_gamma.values()])
    for gamma in sorted(r_phi_for_gamma.keys()):
        print(gamma, sorted(list(r_phi_for_gamma[gamma])))


if __name__ == '__main__':
    setup()
    writeIKData(robot)
