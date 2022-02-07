import pdb
import math, random, string, inspect, time
from itertools import product
from random import sample
import numpy as np

import skrobot
from skrobot.planner.utils import get_robot_config, set_robot_config
from skrobot.coordinates import CascadedCoords, Coordinates
from skrobot.coordinates.math import rpy_angle, rpy_matrix
from skrobot.model import RobotModel, Box
from skrobot.coordinates import Coordinates

from pr2_ik.pr2IkMap import readIKData_GRP, IKkey, IKkey_neighbors, dr, dangle
from pr2_ik.ik import sample_tool_ik, is_ik_compiled
import transformations as transf

import trimesh

from rrt import runRRT, smoothPath, eqChains, RRT_INTERPOLATE_STEP_SIZE

from autil.miscUtil import Hashable
from graphics.colorNames import colors as color_names

# TODO:
# Grasping
# reset_pose leaves base in place
# interpolation for infinite rotation
# interpoaltion within joint limits

DEBUG_RRT = False

approachBackoff = 0.01
approachPerpBackoff = 0.01

# All the state is kept in the Environment
class Environment:
    def __init__(self):
        self._robot = skrobot.models.PR2()
        self._robot.reset_pose()
        self.robot = Robot(self)
        self.chains = ('base', 'larm', 'rarm', 'head')
        self.chain_type = {'base': 'omni_base_chain',}
        for chain in self.chains[1:]:
            self.chain_type[chain] = 'revolute_chain'
        rob = self._robot
        self.robot_base_links = \
            [rob.base_link, rob.base_bellow_link, rob.torso_lift_link]
        self.robot_link_list = \
            {'larm': rob.larm.link_list,
             'rarm': rob.rarm.link_list,
             'head': rob.head.link_list,
             'base': self.robot_base_links}
        self.robot_joint_list = \
            {chain: [link.joint for link in self.robot_link_list[chain]] for chain \
             in self.chains}
        self.robot_end_coords = \
            {'larm': rob.larm_end_coords,
             'rarm': rob.rarm_end_coords,
             'head': rob.head_end_coords}
        self.robot_collision_link_list = \
            {'larm': self.robot_link_list['larm'] + \
             [rob.l_forearm_link,
              rob.l_gripper_palm_link,
              rob.l_gripper_r_finger_link,
              rob.l_gripper_l_finger_link,
              rob.l_gripper_r_finger_tip_link,
              rob.l_gripper_l_finger_tip_link],
             'rarm': self.robot_link_list['rarm'] + \
             [rob.r_forearm_link,
              rob.r_gripper_palm_link,
              rob.r_gripper_r_finger_link,
              rob.r_gripper_l_finger_link,
              rob.r_gripper_r_finger_tip_link,
              rob.r_gripper_l_finger_tip_link],
             'head': self.robot_link_list['head'],
             'base': self.robot_link_list['base']}
        self.workspace = np.array( [[-2.0, -3., 0.0], [3.0, 3., 2.0]] )
        self.bodies = {}
        self.attached = {'larm': None, 'rarm': None}
        self.default_arm = 'rarm'
        self.initialize_robot_collision_manager()
        self.bodies_collision_manager = trimesh.collision.CollisionManager()  # empty
        self._cur_conf = self.conf()
        self.lock = True
        self.viewer = None
        # For inverse kin
        larm = self._robot.larm
        rarm = self._robot.rarm
        self.torso_z = 0.3      # keep constant
        self.tool_to_wrist = {'rarm': rarm.end_coords.transformation(rarm.wrist_coords).T(),
                              'larm': larm.end_coords.transformation(larm.wrist_coords).T(),}
        
        self.GRP = {}
        self.jointValues = {}
        IKData_GRP = readIKData_GRP()
        for i, hand in enumerate(('rarm', 'larm')):
            self.jointValues[hand] = IKData_GRP[i][0] 
            self.GRP[hand] = IKData_GRP[i][1]
        
    def copy(self):
        env = Environment()
        env.set_conf(self._cur_conf, redraw=False)
        for body in self.bodies.values():
            env.add_body(body.copy(env))
        return env
    def add_body(self, body):                # Body instance
        self.bodies[body.name] = body
        body_coords = body.shape.worldcoords()
        print(f'add_body {body.name} at {body_coords}')
        self.bodies_collision_manager.add_object(
            body.name, body.shape.collision_mesh, body_coords.T())
    def get_robot(self):
        return self.robot
    def get_viewer(self):
        self.viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(2*640, 2*480))
        # self.viewer.add(self.robot)
        rlinks = []
        for chain in self.chains:
            rlinks.extend(self.robot_collision_link_list[chain])
        rjoints = []
        for link in rlinks:
            rjoints.append(link.joint)
        self.viewer.add(RobotModel(link_list=rlinks, joint_list=rjoints))
        for body in self.bodies.values():
            self.viewer.add(body.shape)
        self.viewer.show()
        return self.viewer
    def get_body(self, objName):            # objName is string
        return self.bodies[objName]
    def lock(self):
        self.lock = True
        return
    def unlock(self):
        self.lock = False               # allow displays
        return
    def initialize_robot_collision_manager(self):
        cm = trimesh.collision.CollisionManager()
        for chain in self.chains:
            for link in self.robot_collision_link_list[chain]:
                mesh = link.collision_mesh
                if mesh is not None:
                    transform = link.worldcoords().T()
                    cm.add_object(
                        link.name, mesh, transform=transform)
                else:
                    print(f"No collision mesh for {link}")
        self.robot_collision_manager = cm
    def update_robot_collision_manager(self, chain=None):
        cm = self.robot_collision_manager
        assert cm
        for chain in [chain] if chain else self.chains:
            for link in self.robot_collision_link_list[chain]:
                cm.set_transform(link.name, link.worldcoords().T())
            if self.attached.get(chain, None):
                (body, rel) = self.attached[chain]
                arm_end_coords = self.robot_end_coords[chain].worldcoords()
                cm.set_transform(body.name, arm_end_coords.transform(rel).worldcoords().T())
    def update_bodies_collision_manager(self, body):
        cm = self.bodies_collision_manager
        assert cm
        cm.set_transform(body.name, body.shape.worldcoords().T())
    def redraw(self): 
        if not self.lock and self.viewer:
            self.viewer.redraw()       
    def set_default_robot_conf(self, redraw=True):
        self._robot.reset_pose()
        self._cur_conf = self.conf()
        self.update_robot_collision_manager()
        if redraw: self.redraw()
    def set_chain_conf(self, chain, joint_angles, redraw=True, update=True):
        self._cur_conf = self._cur_conf.set(chain, joint_angles)
        joints = self.robot_joint_list[chain]
        set_robot_config(self._robot, joints, joint_angles)
        if update:
            self.update_robot_collision_manager(chain)
        if redraw:
            self.redraw()
    def set_base_values(self, baseConf, redraw=True, update=True):
        assert 'base' in self.chains
        self._cur_conf = self._cur_conf.set('base', baseConf)
        x, y, theta = baseConf
        co = Coordinates(pos=[x, y, 0.0], rot=rpy_matrix(theta, 0.0, 0.0))
        self._robot.newcoords(co)
        if update:
            self.update_attached(redraw=redraw)
            self.update_robot_collision_manager()
        if redraw:
            self.redraw()        
    def baseConf(self):
        x, y, _ = self._robot.translation
        rpy = rpy_angle(self._robot.rotation)[0]
        theta = rpy[0]
        return np.array([x, y, theta])
    def get_chain_values(self, chain):
        return get_robot_config(self._robot, self.robot_joint_list[chain]) 
    def conf(self):
        jv = {chain: get_robot_config(self._robot, self.robot_joint_list[chain]) \
              for chain in self.chains if chain != 'base'}
        for chain in self.chains:
            if chain == 'base': continue
            if 'arm' in chain:
                jv[chain+'_grip'] = np.array([self._robot.gripper_distance(arm=chain)])
        if 'base' in self.chains:
            jv['base'] = self.baseConf()
        return Conf(jv, self.robot)
    def set_conf(self, conf, redraw=True):
        r = self.robot
        if 'base' in self.chains:
            same_base = all(conf['base'] == self._cur_conf['base'])
        else:
            same_base = True
        for chain in self.chains:
            if chain == 'base': continue
            if chain not in conf.chain_vals: continue
            if all(conf[chain] == self._cur_conf[chain]): continue
            self.set_chain_conf(chain, conf[chain], redraw=False, update=False)
            gchain = chain+'_grip' if 'arm' in chain else None
            if gchain in conf.chain_vals:
                if not conf[gchain][0] == self._cur_conf[gchain][0]:
                    self._robot.gripper_distance(arm=chain, dist=conf[gchain][0])
            if same_base:
                self.update_robot_collision_manager(chain)
                self.update_attached(redraw=redraw)
        if not same_base:
            self.set_base_values(conf['base'], redraw=False, update=True)  # will update everything
        if redraw: self.redraw()
    def update_attached(self, redraw=True):
        for chain in self.attached:
            if self.attached.get(chain, None):
                (body, rel) = self.attached[chain]
                arm_end_coords = self.robot_end_coords[chain].worldcoords()
                body.set_pose(arm_end_coords.transform(rel), redraw=False)
    def check_collision(self, body1, body2, return_names=False):
        cm = trimesh.collision.CollisionManager()
        tr2 = body2.shape.worldcoords().T()
        cm.add_object(body2.name, body2.shape.collision_mesh, transform=tr2)
        if isinstance(body1, Robot):
            return cm.in_collision_other(self.robot_collision_manager, return_names=return_names)
        else:
            tr1 = body1.shape.worldcoords().T()
            cm.add_object(body1.name, body1.shape.collision_mesh, transform=tr1)
            return cm.in_collision_internal(return_names=return_names)
    def check_all_collision(self, return_names=True, ignore_bodies=[]):
        cm = self.bodies_collision_manager
        for b in ignore_bodies:
            cm.remove_object(b.name)
        ans = cm.in_collision_other(self.robot_collision_manager, return_names=return_names)
        if return_names:
            if ans[0]:
                print('Collision', ans[1])
            ans = ans[0]
        for b in ignore_bodies:
            cm.add_object(
                b.name, b.shape.collision_mesh, b.worldcoords().T())
        return ans
    def inverse_kinematics(self, target_coords,
                           move_arm='rarm', n=50,
                           base_values=None,
                           check_collision=None):
        assert is_ik_compiled()
        for i, base in enumerate(self.potentialBasePosesGen(target_coords.worldcoords().T(), move_arm, n=n) \
                                 if base_values is None else [base_values]):
            print(f'IK {i}: base={base}')
            x,y,th = base
            base_coords = Coordinates().translate([x,y,0.]).rotate(th, 'z')
            target_coords_rel_base = base_coords.transformation(target_coords)
            sol = sample_tool_ik(self, move_arm, target_coords_rel_base, max_attempts=n)
            if sol:
                print(f'IK {i}: base={base}, sol={sol}')
                conf = self.conf().set('base', np.array([x,y,th]))
                return conf.set(move_arm, np.array(sol))

    def potentialBasePosesGen(self, target_matrix, hand, n=None, complain=True, explain=None):
        assert hand in ('rarm', 'larm')
        wrist_matrix = np.dot(target_matrix, self.tool_to_wrist[hand])
        # print('target\n', target_matrix)
        # print('wrist\n', wrist_matrix)
        pos, euler = wrist_matrix[:,3].tolist(), transf.euler_from_matrix(wrist_matrix, axes='sxyz')
        key = IKkey(pos, euler)
        entries = []
        if key in self.GRP[hand]:
            entries = [self.GRP[hand][key]]
        else:
            entries = []
            for key in IKkey_neighbors(pos, euler):
                if key in self.GRP[hand]:
                    entries.append(self.GRP[hand][key])
                    break       # do only one
        if not entries:
            # print('Failed to find base poses')
            print('-B', end='')
            return
        for i in range(n):
            entry = random.choice(entries)
            grp = [random.randrange(*r) if r[0] < r[1] else r[0] for r in entry]
            grp_vals = [grp[0]*dangle, grp[1]*dr, grp[2]*dangle]
            theta = euler[2] - grp_vals[0]
            x = pos[0] - grp_vals[1]*math.cos(grp_vals[2]+theta)
            y = pos[1] - grp_vals[1]*math.sin(grp_vals[2]+theta)
            yield (x, y, theta)

##################################################
# Types used in StripStream TAMP
##################################################

class Conf(Hashable):
    def __init__(self, conf, robot):
        self.chain_vals = conf
        self.robot = robot
        Hashable.__init__(self)
    def __getitem__(self, name):
        return self.chain_vals[name]
    def nearEqual(self, other, thr = 1.0e-6):
        if not set(self.chain_vals.keys()) == set(other.chain_vals.keys()):
            return False
        return not any([any([abs(x-y) > thr for (x,y) in zip(self.chain_vals[chain],
                                                             other.chain_vals[chain])]) \
                        for chain in self.chain_vals])
    def copy(self):
        return Conf(self.chain_vals.copy(), self.robot)
    def set(self, name, value):
        assert value is None or isinstance(value, np.ndarray), \
               'Illegal value for JointConf set, should be an array'
        c = self.copy()
        c.chain_vals[name] = value
        return c
    def desc(self):
        return self.chain_vals
    def __hash__(self):
        # Hashing ndarrays
        if self.hashValue is None:
            cv = self.chain_vals
            val = tuple(sorted([(chain, cv[chain].tobytes()) for chain in cv]))
            self.hashValue = hash(val)
        return self.hashValue
    def __eq__(self, other):
        return eqChains(self, other)
    def __ne__(self, other):
        return not eqChains(self, other)

class Traj(Hashable):
    def __init__(self, path, grasp=None):
        self.pathConfs = tuple(path)
        self.grasp = grasp
        Hashable.__init__(self)
    def path(self):
        return self.pathConfs
    def traj(self):
        return self.pathConfs
    def end(self):
        return self.pathConfs[-1]
    def reverse(self):
        return Traj(self.pathConfs[::-1], self.grasp)
    def desc(self):
        return (self.pathConfs, self.grasp)

class Body:
    def __init__(self, env, shape, name):
        self.shape = shape              # shape at origin
        self.point = None
        self.env = env
        self.name = name
        self.grasps = []
    def get_pose(self):
        return self.shape.worldcoords()
    def set_pose(self, coords, redraw=True):
        # It could be in the hand
        self.shape.newcoords(coords)
        if self.name in self.env.bodies:
            self.env.update_bodies_collision_manager(self)
        if redraw: self.env.redraw()
    def get_name(self):
        return self.name
    def get_grasps(use_grasp_approach, use_grasp_type):
        return self.grasps       # TODO: implement args
    def copy(self, env=None):
        return Body(env or self.env, self.shape.copy(), self.name)

class Robot:
    def __init__(self, env):
        self.env = env
    def get_configuration(self):
        return self.env.conf()
    def set_configuration(self, conf, redraw=True):
        self.env.set_conf(conf, redraw=redraw)
    def grab(self, arm, body):
        env = self.env
        assert env.bodies[body.name]
        assert env.attached[arm] is None
        del env.bodies[body.name]
        arm_end_coords = self.env.robot_end_coords[arm].worldcoords()
        body_coords = body.shape.worldcoords()
        rel = body_coords.transformation(arm_end_coords)
        env.attached[arm] = (body, rel)
        env.bodies_collision_manager.remove_object(body.name)
        env.robot_collision_manager.add_object(
            body.name, body.shape.collision_mesh, body_coords.T())
    def release(self, arm, body):
        env = self.env
        assert body.name not in env.bodies
        assert env.attached[arm]
        assert body == env.attached[arm][0]
        env.bodies[body.name] = body
        env.attached[arm] = None
        body_coords = body.shape.worldcoords()        
        env.robot_collision_manager.remove_object(body.name)
        env.bodies_collision_manager.add_object(
            body.name, body.shape.collision_mesh, body_coords.T())
    def set_base_values(self, base):
        self.env.set_base_values(base)
    def set_conf(self, conf):
        self.env.set_conf(conf)        
    def set_default_robot_conf(self):
        self.env.set_default_robot_conf()
    # Below is interface to RRT.
    def conf_check_collision(self, conf, protect=False):
        if protect:
            cur_conf = self.get_configuration()
        self.set_configuration(conf, redraw=False)
        ans = self.env.check_all_collision()
        if protect:
            self.set_configuration(cur_conf, redraw=False)
        return ans
    # Note, only moves one chain at a time, in the order indicated by self.env.chains
    def interpolate(self, q_f, q_i, stepSize=None, maxSteps=500):
        return list(self.interpolate_gen(q_f, q_i, stepSize=stepSize, maxSteps=maxSteps))
    def interpolate_gen(self, q_f, q_i, stepSize=None, maxSteps=300):
        if stepSize is None:
            stepSize = RRT_INTERPOLATE_STEP_SIZE
        path = [q_i]
        q = q_i
        step = 0
        yield q_i               # always do initial point
        while q is not q_f:
            if step > maxSteps:
                input('interpolate exceeded maxSteps')
                pdb.set_trace()
            qn = self.step_along_line(q_f, q, stepSize)
            if q is qn: break
            q = qn
            path.append(q)
            yield q
            step += 1
        if path[-1] is q_f:
            path.pop()
        path.append(q_f)
        if len(path) > 1 and not(path[0] is q_i and path[-1] is q_f):
            input('Path inconsistency')
        yield q_f
    def interpolate_path(self, path, stepSize = None):
        interpolated = []
        for i in range(1, len(path)):
            qf = path[i]
            qi = path[i-1]
            confs = self.interpolate(qf, qi, stepSize=stepSize)
            if DEBUG_RRT:
                print(i, 'path segment has', len(confs), 'confs')
            interpolated.extend(confs)
        return interpolated
    def step_along_line(self, q_f, q_i, stepSize, forward = True):
        moveChains = [c for c in self.env.chains if c in q_i.chain_vals]
        q = q_i.copy()
        nv = {}
        # Reverse the order of chains when working on the "from the goal" tree.
        for chain in moveChains if forward else moveChains[::-1]:
            if all(q_f[chain] == q_i[chain]): continue
            jv = chain_step(chain, q_f, q_i, stepSize)
            if DEBUG_RRT:
                print('step chain', chain)
            return q.set(chain, jv) # only move one chain at a time...
        return q_i
    def dist_conf_abs(self, q1, q2, dmax=1.0e6):
        total = 0.
        for chain in self.env.chains:
            if chain in q1.chain_vals and chain in q2.chain_vals:
                td = chain_dist(chain, q1, q2, dmax-total)
                if td is None: return None
                total += td
        return total
    def random_conf(self, conf, chains):
        return Conf({chain : chain_random(chain, conf) if chain in chains else conf[chain] \
                     for chain in conf.chain_vals}, conf.robot)
    def in_workspace(self, conf):
        return True
    def inverse_kinematics(self, target_coords, **kwargs):
        return self.env.inverse_kinematics(target_coords, **kwargs)
    
class Manipulator:
    def __init__(self, robot, arm):
        self.robot = robot
        self.arm = arm
    def get_end_coords(self):
        return self.robot.env.robot_end_coords[self.arm].worldcoords()    
    def set_configuration(self, conf):
        if isinstance(conf, Conf):
            conf = conf[self.arm]  # get the array
        self.robot.env.set_chain_conf(self.arm, conf)
    def get_configuration(self):
        return self.robot.env.get_chain_values(self.arm)
    def open_gripper(self):
        self.robot.env._robot.gripper_distance(arm=self.arm, dist=0.08)
    def set_gripper(grip=0.08):
        self.robot.env._robot.gripper_distance(arm=self.arm, dist=grip)        

class Grasp(Hashable):
    def __init__(self, objName, grasp, trans, approach=None):
        self.g = grasp
        self.objName = objName
        self.grasp_trans = trans
        self.approach_vector = approach
        Hashable.__init__(self)
    def desc(self):
        return (self.objName, self.g)
    
##################################################
# Chain operations
##################################################

def chain_random(chain, conf):
    ctype = conf.robot.env.chain_type.get(chain, None)
    if ctype is None:
        return conf[chain]
    elif ctype == 'omni_base_chain':
        workspace = conf.robot.env.workspace
        return np.array([random.uniform(workspace[0,0], workspace[1,0]),
                         random.uniform(workspace[0,1], workspace[1,1]),
                         random.uniform(-math.pi, math.pi)])
    elif ctype == 'revolute_chain':
        joints = conf.robot.env.robot_joint_list[chain]
        return np.array([random.uniform(max(-math.pi, j.min_angle),
                                        min(math.pi, j.max_angle)) \
                         for j in joints])
    else:
        raise Exception(f'Unknown chain type: {ctype}')

def chain_dist(chain, q1, q2, dmax=1.0e6):
    ctype = q1.robot.env.chain_type[chain]
    jvf = q1[chain]; jvi = q2[chain]
    if ctype == 'omni_base_chain':
        d = 0.0
        for i in range(3):
            if i < 2:
                d += np.pi*(abs(jvf[i]-jvi[i]))
            else:
                d += 10*abs(angle_diff(jvf[-1], jvi[-1]))
            if d > dmax: return None
        return d
    elif ctype == 'revolute_chain':
        assert len(jvf) == len(jvi), 'Inconsistent joints in chain_dist'
        joints = q1.robot.env.robot_joint_list[chain]    
        d = 0.0
        for i in range(len(jvi)):
            diff = joint_diff(joints[i], jvf[i], jvi[i])
            if diff < 0: d += -diff
            else: d += diff
            if d > dmax: return None
        return d
    else:
        raise Exception(f'Unknown chain type: {ctype}')

def chain_step(chain, qf, qi, stepSize):
    ctype = qf.robot.env.chain_type[chain]
    jvf = qf[chain]; jvi = qi[chain]
    assert len(jvf) == len(jvi), 'Inconsistent joints in step_along_line'
    indices = list(range(len(jvi)))
    if ctype == 'omni_base_chain':
        diffs = []
        for i in range(3):
            if i < 2:
                # Cut down step size
                diffs.append(0.25*(jvf[i]-jvi[i]))
            else:
                diffs.append(angle_diff(jvf[-1], jvi[-1]))
    elif ctype == 'revolute_chain':
        joints = qf.robot.env.robot_joint_list[chain]    
        assert all(j.min_angle <= jv <= j.max_angle for (jv, j) \
                   in zip(jvf, joints)), \
            'Invalid joint value in stepAlongLine'
        diffs = [joint_diff(joints[i], jvf[i], jvi[i]) for i in indices]
    diffs = np.array(diffs)
    length = np.linalg.norm(diffs)
    if length == 0. or stepSize/length >= 1.: return jvf
    vals = (stepSize/length)*diffs + jvi
    if DEBUG_RRT:
        print('diffs', chain, diffs)
        print('step', chain, vals)    
    return vals

def joint_diff(joint, x, y):
    if joint.min_angle == -float('inf'):
        return angle_diff(x, y)
    else:
        return x - y

def angle_diff(x, y):
    twoPi = 2*math.pi
    z = (x - y)%twoPi
    if z > math.pi:
        return z - twoPi
    else:
        return z
    
##################################################
# Making box bodies
##################################################

BLUE = 'blue'
RED = 'red'
colors = ['red', 'green', 'blue', 'cyan', 'purple', 'pink', 'orange']

def get_color(color):           # RGBA color
    return [x/256. for x in color_names.get(color, (50,50,50))] + [1.0]

def pickColor(name):
    if name[-1] in string.ascii_uppercase:
        cn = len(colors)
        return colors[string.ascii_uppercase.index(name[-1])%cn]
    else:
        return 'black'

def unit_quat(): return np.array([1,0,0,0])  # Note [w,x,y,z]

##################################################
# Making box bodies
##################################################

USE_HORIZONTAL = True
USE_VERTICAL = True

def box_body(env, length, width, height, name='box', color='black'):
    shape = Box((length, width, height), name=name, face_colors=get_color(color))
    body = Body(env, shape, name)
    if length <= 0.07 or width <= 0.07:  # graspable
        body.grasps = [Grasp(name, 0,
                             transf.rotation_matrix(np.pi/2, (0,1,0)),
                             np.array([0.0, 0.0, 0.01])),
                       Grasp(name, 0, np.eye(4), np.array([0.0, 0.0, 0.01]))]
    else:
        print(f"{name} is not graspable: width={width}, height={height}")
    return body

##################################################
# Problem definitions
##################################################

class ManipulationProblem:
    def __init__(self, name,
                 object_names=[], table_names=[], initial_poses={},
                 goal_object_name=None, goal_poses={}, known_poses=[]):
        self.name = name
        self.object_names = object_names
        self.goal_object_name = goal_object_name
        self.table_names = table_names
        self.initial_poses = initial_poses
        self.goal_poses = goal_poses
        self.known_poses = known_poses

BODY_PLACEMENT_Z_OFFSET = 1e-3

def function_name(stack): # NOTE - stack = inspect.stack()
  return stack[0][3]
def flatten(x):
    xflat = []
    for xi in x: xflat.extend(xi)
    return xflat

# (Incremental Task and Motion Planning: A Constraint-Based Approach)
def dantam2(env, n_obj = 8): 
    m, n = 3, 3
    assert n_obj <= m*n
    side_dim = .05
    height_dim = .1
    box_dims = (side_dim, side_dim, height_dim)
    separation = (side_dim, side_dim)

    coordinates = list(product(list(range(m)), list(range(n))))
    assert n_obj <= len(coordinates)
    obj_coordinates = sample(coordinates, n_obj)  # randomized

    length = m*(box_dims[0] + separation[0])
    width = n*(box_dims[1] + separation[1])
    height = .7

    table = box_body(env, length, width, height, name='table', color='brown')
    env.add_body(table)
    x, y, z = 0, 0, height/2
    table.set_pose(Coordinates(pos=np.array([x, y, z]), rot=unit_quat()))

    robot = env.get_robot()
    robot.set_default_robot_conf()
    robot.set_base_values(np.array([-0.75, 0, 0]))

    poses = []
    z =  height + height_dim/2 + BODY_PLACEMENT_Z_OFFSET
    for r in range(m):
        row = []
        x = -length/2 + (r+.5)*(box_dims[0] + separation[0])
        for c in range(n):
            y = -width/2 + (c+.5)*(box_dims[1] + separation[1])
            row.append(Coordinates(pos=np.array([x, y, z]), rot=unit_quat()))
        poses.append(row)

    initial_poses = {}
    goal_poses = {}
    for i, (r, c) in enumerate(obj_coordinates):
        row_color = np.zeros(4)
        row_color[2-r] = 1.
        if i == 0:
            name = 'goal%d-%d'%(r, c)
            color = BLUE
            # goal_poses[name] = poses[int(m/2)][int(n/2)]
            empty = list(set(coordinates) - set(obj_coordinates))[0]
            goal_poses[name] = poses[empty[0]][empty[1]]
            goal_object_name = name
        else:
            name = 'block%d-%d'%(r, c)
            color = RED
        initial_poses[name] = poses[r][c]
        obj = box_body(env, *box_dims, name=name, color=color)
        env.add_body(obj)                # needs add_body and set_pose...
        obj.set_pose(poses[r][c])
    known_poses = list(flatten(poses))
    return ManipulationProblem(
        function_name(inspect.stack()),
        object_names=list(initial_poses.keys()), table_names=[table.get_name()],
        goal_poses=goal_poses, goal_object_name=goal_object_name,
        initial_poses=initial_poses, known_poses=known_poses)

def manip_from_pose_grasp(pose, grasp):
    # This is the robot gripper
    manip_trans = pose * Coordinates(grasp.grasp_trans)
    # Ad-hoc backoff strategy
    if abs(manip_trans.worldcoords().T()[2,0]) < 0.1:  # horizontal
        offset = Coordinates(pos=np.array((-approachBackoff,0.,approachPerpBackoff)))
    else:                               # vertical
        offset = Coordinates(pos=np.array((approachBackoff,0.,0.)))
    manip_trans_approach = manip_trans * offset
    # The grasp and the approach
    return manip_trans, manip_trans_approach

def object_pose_from_manip_coords(manip_coords, grasp_trans):
    return manip_coords * Coordinates(grasp_trans).inverse_transformation()

def solve_inverse_kinematics(env, manipulator, manip_coords):
    robot = manipulator.robot
    cur_conf = robot.get_configuration()
    new_conf = robot.inverse_kinematics(manip_coords, move_arm=manipulator.arm,
                                        base_values=cur_conf['base']
    )
    assert (new_conf['base'] == cur_conf['base']).all()
    return new_conf

def vector_traj_helper(env, manipulator, approach_trans): # Trajectory from grasp configuration to pregrasp
    approach_conf = solve_inverse_kinematics(env, manipulator, approach_trans)
    # robot is currently at grasp conf, trajectory goes to pre-conf,
    # should be linear intepolation, but just joint interpolation now.
    robot = manipulator.robot
    path = robot.interpolate_path([robot.get_configuration(), approach_conf])
    return Traj([q if isinstance(q, Conf) else Conf(q) for q in path])

def cspace_traj_helper(conf1, conf2, chains=None,
                       max_iterations=100, fail_iterations=10, smooth=True):
    robot = conf1.robot
    initConf = conf1
    destConf = conf2
    if chains is None:
        chains = list(conf2.keys())
    path = runRRT(robot, initConf, destConf, chains,
                  max_iterations, fail_iterations)
    if not path:
        print('Failed to find a path')
        return Traj([initConf, destConf])
    if smooth:
        path = smoothPath(robot, path, npasses=5, verbose=False)
    return Traj([q if isinstance(q, Conf) else Conf(q) for q in path]) if path else None

def sample_manipulator_trajectory(manipulator, traj):
    return traj.path() if isinstance(traj, Traj) else traj



