from sss import *

def prof(test, n=100):
    import cProfile
    cProfile.run(test, 'prof')
    prof_print(n)

def prof_print(n=100):
    import pstats
    p = pstats.Stats('prof')
    p.sort_stats('cumulative').print_stats(n)
    p.sort_stats('cumulative').print_callers(n)

################
# Define the tests

test_ik = True
test_path = False
test_grasp = True

################

real_env = Environment()
real_env.unlock()                # make sure we draw
real_env.set_default_robot_conf()
dantam2(real_env)
real_robot = real_env.get_robot()
viewer = real_env.get_viewer()

right = Manipulator(real_robot, 'rarm')

# Conf for a side grasp...
jvals = np.array([-1.162287950515747, 0.8972723484039307, -1.600000023841858, -2.101621389389038, -1.1904525756835938, -0.8628466129302979, 2.0946085453033447])
print('target right arm conf:\n', jvals)
right.set_configuration(jvals)

# Test setting and getting joint vals
assert np.equal(right.get_configuration(), jvals).all()
print('...read it successfully\n', jvals)

# Test inverse kinematics
if test_ik:
    target_coords = right.get_end_coords()  # gripper coords
    print('target transform\n', target_coords.worldcoords().T())
    target_conf = real_robot.env.inverse_kinematics(target_coords)
    print('solved conf\n', target_conf)
    real_robot.set_configuration(target_conf)
    print('solved end point transform\n', right.get_end_coords().T())
    print('solved wrist transform\n', right.robot.env._robot.rarm.wrist_coords.worldcoords().T())

if test_path:
    env = real_env.copy()
    # env = Environment()
    # env.set_default_robot_conf()
    # dantam2(env)
    robot = env.get_robot()

    initConf = robot.get_configuration()
    destConf = initConf.set('base', np.array([1, 0, 0]))

    for conf in (initConf, destConf):
        del conf.chain_vals['larm_grip']
        del conf.chain_vals['rarm_grip']
        del conf.chain_vals['head']

    PATH = None

    def run_path(smooth=True):
        global PATH
        path = runRRT(robot, initConf, destConf, 100, 10)
        if smooth and path:
            path = smoothPath(robot, path, verbose=False)
        PATH = path

    # prof('run_path(False)')
    run_path(smooth=True)

    for conf in PATH:
        real_robot.set_configuration(conf)
        time.sleep(0.1)

print('==> Press [q] to close window')
while not viewer.has_exit:
    time.sleep(0.1)
    viewer.redraw()

