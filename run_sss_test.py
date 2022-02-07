import pdb
import math
from time import sleep

from sss import *

####################

PROBLEM = dantam2
ARM = 'larm'
MAX_GRASPS = 6
# USE_GRASP_APPROACH = GRASP_APPROACHES.TOP # TOP | SIDE
# USE_GRASP_TYPE = GRASP_TYPES.GRASP # GRASP | TOUCH
USE_GRASP_APPROACH = True
USE_GRASP_TYPE = None
DISABLE_TRAJECTORIES = False
DISABLE_TRAJ_COLLISIONS = False

assert not DISABLE_TRAJECTORIES or DISABLE_TRAJ_COLLISIONS
if DISABLE_TRAJECTORIES:
  print('Warning: trajectories are disabled')
if DISABLE_TRAJ_COLLISIONS:
  print('Warning: trajectory collisions are disabled')

####################

def simple_test(env):
  problem = PROBLEM(env)
  viewer = env.get_viewer()
  robot = env.get_robot()
  print('robot', robot)
  env.set_default_robot_conf()
  env.set_base_values(np.array([-0.75, .2, -math.pi/2]))
  env.unlock()                # make sure we draw
  manipulator = Manipulator(robot, ARM)

  bodies = {obj: env.get_body(obj) for obj in problem.object_names}
  all_bodies = list(bodies.values())

  body1 = bodies[problem.goal_object_name]
  body2 = [bodies[name] for name in bodies if name != problem.goal_object_name][0]
  grasps = body1.grasps
  poses = problem.known_poses if problem.known_poses else []

  manipulator.open_gripper()
  initial_conf = robot.get_configuration()

  ####################

  def _cfree_pose(pose1, pose2): # Collision free test between an object at pose1 and an object at pose2
    body1.set_pose(pose1)
    body2.set_pose(pose2)
    return not env.check_collision(body1, body2)

  def _cfree_traj_pose(traj, pose): # Collision free test between a robot executing traj and an object at pose
    body2.set_pose(pose)
    for conf in traj.path():
      manipulator.set_configuration(conf)
      if env.check_collision(robot, body2):
        return False
    return True

  def _cfree_traj_grasp_pose(traj, grasp, pose): # Collision free test between an object held at grasp while executing traj and an object at pose
    body2.set_pose(pose)
    for conf in traj.path():
      manipulator.set_configuration(conf)
      manip_coords = manipulator.get_end_coords()
      body1.set_pose(object_pose_from_manip_coords(manip_coords, grasp.grasp_trans))
      if env.check_collision(body1, body2):
        input('Collision in _cfree_traj_grasp_pose')
        pdb.set_trace()
        return False
    return True

  def _cfree_traj(traj, pose): # Collision free test between a robot executing traj (which may or may not involve a grasp) and an object at pose
    if DISABLE_TRAJ_COLLISIONS:
      return True
    return _cfree_traj_pose(traj, pose) and (traj.grasp is None or _cfree_traj_grasp_pose(traj, traj.grasp, pose))

  ####################

  def sample_grasp_traj(body, pose, grasp): # Sample pregrasp config and motion plan that performs a grasp
    manip_coords, approach_vector = manip_from_pose_grasp(pose, grasp)
    grasp_conf = solve_inverse_kinematics(env, manipulator, manip_coords) # Grasp configuration
    if grasp_conf is None: return
    if DISABLE_TRAJECTORIES:
      yield [(Conf(grasp_conf), object())]
      return

    manipulator.set_configuration(grasp_conf)
    robot.grab(manipulator.arm, body)
    grasp_traj = vector_traj_helper(env, manipulator, approach_vector) # Trajectory from grasp configuration to pregrasp
    robot.release(manipulator.arm, body)
    if grasp_traj is None: return
    grasp_traj.grasp = grasp
    pregrasp_conf = grasp_traj.end()    # Pregrasp configuration
    yield [(pregrasp_conf, grasp_traj)]

  def sample_free_motion(conf1, conf2): # Sample motion while not holding
    if DISABLE_TRAJECTORIES:
      yield [(object(),)] # [(True,)]
      return
    traj = cspace_traj_helper(conf1, conf2, chains=['larm'])
    if not traj: return
    traj.grasp = None
    yield [(traj,)]

  def sample_holding_motion(conf1, conf2, body, grasp): # Sample motion while holding
    if DISABLE_TRAJECTORIES:
      yield [(object(),)] # [(True,)]
      return
    manipulator.set_configuration(conf1)
    manip_coords = manipulator.get_end_coords()
    body1.set_pose(object_pose_from_manip_coords(manip_coords, grasp.grasp_trans))
    robot.grab(manipulator.arm, body)
    traj = cspace_traj_helper(manipulator, conf2, chains=['larm'], max_iterations=10)
    robot.release(manipulator.arm, body)
    if not traj: return
    traj.grasp = grasp
    yield [(traj,)]

  ####################

  def _execute_traj(traj):
    path = list(sample_manipulator_trajectory(manipulator, traj.traj()))
    for j, conf in enumerate(path):
      manipulator.set_configuration(conf)
      print('%s/%s) Step'%(j, len(path)))
      input('Next?')

  manipulator.set_configuration(initial_conf)

  goal_obj = problem.goal_object_name
  s_pick = next(sample_grasp_traj(bodies[goal_obj],
                                  problem.initial_poses[goal_obj],
                                  grasps[0]))
  (pre_pick, traj_pick) = s_pick[0]
  s_place = next(sample_grasp_traj(bodies[goal_obj],
                                   problem.goal_poses[goal_obj],
                                   grasps[0]))
  (pre_place, traj_place) = s_place[0]
  s_to_pick = next(sample_free_motion(env.conf(), pre_pick))  # approach pick pre_grasp
  (traj_to_pick,) = s_to_pick[0]
  s_to_place = next(sample_holding_motion(pre_pick, pre_place, grasps[0]))  # 
  (traj_to_place,) = s_to_place[0]

  print('==> Press [q] to close window')
  while not viewer.has_exit:
    time.sleep(0.1)
    viewer.redraw()

  print('Executing traj to pick approach conf')
  _execute_traj(traj_to_pick)        # approach
  print('Executing traj to pick conf')
  _execute_traj(traj_pick.reverse())    # pick
  print('Executing traj to place approach conf')  
  _execute_traj(traj_to_place)    # move holding
  print('Executing traj to place conf')  
  _execute_traj(traj_place.reverse())    # place
  input('Continue?')
  # print('_cfree_traj_pose(traj, poses[0])', _cfree_traj_pose(traj2, poses[0]))
  # print('_cfree_traj_grasp_pose(traj, grasps[0], poses[0])', _cfree_traj_grasp_pose(traj2, grasps[0], poses[0]))
  #print('_cfree_traj_grasp_pose(traj, grasps[0], poses[1])', _cfree_traj_grasp_pose(traj2, grasps[0], poses[1]))

  input('Finish?')

##################################################
def test():
  simple_test(Environment())

if __name__ == '__main__':
  # do a test
  test()
