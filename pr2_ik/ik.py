import random, pdb
import numpy as np

IK_FRAME = {
    'left': 'l_gripper_tool_frame',
    'right': 'r_gripper_tool_frame',
}
BASE_FRAME = 'base_link'
TORSO_JOINT = 'torso_lift_joint'
UPPER_JOINT = {
    'left': 'l_upper_arm_roll_joint', # Third arm joint
    'right': 'r_upper_arm_roll_joint',
}

#####################################

def is_ik_compiled():
    try:
        from .ikLeft import leftIK
        from .ikRight import rightIK
        return True
    except ImportError:
        return False

torsoZ = 0.3
torso_limits = [(torsoZ, torsoZ)]

def get_ik_generator(arm, torso_joint, upper_joint, arm_joint_list, ik_coords_rel_base):
    from .ikLeft import leftIK
    from .ikRight import rightIK
    arm_ik = {'larm': leftIK, 'rarm': rightIK}
    sampled_limits = torso_limits + [(upper_joint.min_angle, upper_joint.max_angle)]
    min_limits = [torsoZ] + [j.min_angle for j in arm_joint_list]
    max_limits = [torsoZ] + [j.max_angle for j in arm_joint_list]
    while True:
        sampled_values = [random.uniform(*limits) for limits in sampled_limits]
        confs = compute_inverse_kinematics(arm_ik[arm], ik_coords_rel_base, sampled_values)
        solutions = [q for q in confs if all_between(min_limits, q, max_limits)]
        print(sampled_values, solutions)
        yield solutions
        if all(lower == upper for lower, upper in sampled_limits):
            break

def sample_tool_ik(robot_env, arm, tool_pose_rel_base, nearby_conf_angles=None, max_attempts=100, **kwargs):
    _robot = robot_env._robot
    torso_joint = _robot.torso_lift_joint
    upper_joint = _robot.l_upper_arm_roll_joint if arm == 'larm' else _robot.r_upper_arm_roll_joint
    arm_joint_list = robot_env.robot_joint_list[arm]
    torso_arm_joint_list = [torso_joint] + arm_joint_list
    generator = get_ik_generator(arm, torso_joint, upper_joint, arm_joint_list, tool_pose_rel_base, **kwargs)
    for i in range(int(max_attempts)):
        print(i, end=' ')
        try:
            solutions = next(generator)
            if solutions:
                # Don't return the torso angle.
                return select_solution(torso_arm_joint_list, solutions, nearby_conf=nearby_conf_angles)[1:]
        except StopIteration:
            break
    return None

def compute_inverse_kinematics(ik_fn, coords, sampled=[]):
    pos = coords.translation
    rot = coords.rotation
    if len(sampled) == 0:
        solutions = ik_fn(rot.tolist(), pos.tolist())
    else:
        solutions = ik_fn(rot.tolist(), pos.tolist(), list(sampled))
    if solutions is None:
        return []
    return solutions

def select_solution(joints, solutions, nearby_conf_angles=None, **kwargs):
    if not solutions:
        return None
    if nearby_conf_angles is None:
        return random.choice(solutions)
    return min(solutions, key=lambda conf: get_distance(nearby_conf_angles, conf, **kwargs))

def get_length(vec, norm=2):
    return np.linalg.norm(vec, ord=norm)

def get_difference(p1, p2):
    assert len(p1) == len(p2)
    return np.array(p2) - np.array(p1)

def get_distance(p1, p2, **kwargs):
    return get_length(get_difference(p1, p2), **kwargs)

def all_between(lower_limits, values, upper_limits):
    assert len(lower_limits) == len(values)
    assert len(values) == len(upper_limits)
    return np.less_equal(lower_limits, values).all() and \
           np.less_equal(values, upper_limits).all()
