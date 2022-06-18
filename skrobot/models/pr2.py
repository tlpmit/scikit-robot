from cached_property import cached_property
import numpy as np
import trimesh

from skrobot.coordinates import CascadedCoords
from skrobot.data import pr2_urdfpath
from skrobot.model import RobotModel

from .urdf import RobotModelFromURDF


class PR2(RobotModelFromURDF):

    """PR2 Robot Model.

    """

    def __init__(self, *args, **kwargs):
        super(PR2, self).__init__(*args, **kwargs)

        self.rarm_end_coords = CascadedCoords(
            parent=self.r_gripper_tool_frame,
            name='rarm_end_coords')
        self.larm_end_coords = CascadedCoords(
            parent=self.l_gripper_tool_frame,
            name='larm_end_coords')
        self.head_end_coords = CascadedCoords(
            pos=[0.08, 0.0, 0.13],
            parent=self.head_tilt_link,
            name='head_end_coords').rotate(np.pi / 2.0, 'y')
        self.torso_end_coords = CascadedCoords(
            parent=self.torso_lift_link,
            name='head_end_coords')

        # Wrist
        self.rarm_wrist_coords = CascadedCoords(
            parent=self.r_wrist_roll_link,
            name='rarm_wrist_coords')
        self.larm_wrist_coords = CascadedCoords(
            parent=self.l_wrist_roll_link,
            name='larm_wrist_coords')

        # limbs
        self.torso = [self.torso_lift_link]
        self.torso_root_link = self.torso_lift_link
        self.larm_root_link = self.l_shoulder_pan_link
        self.rarm_root_link = self.r_shoulder_pan_link
        self.head_root_link = self.head_pan_link

        self.chains = set(['base', 'right', 'left', 'head'])
        self.chain_type = {'base': 'omni_base_chain',
                           'right': 'revolute_chain',
                           'left': 'revolute_chain',
                           'head': 'revolute_chain'}
        self.base_links = \
            [self.base_link, self.base_bellow_link, self.torso_lift_link]
        self.link_lists = \
            {'left': self.larm.link_list,
             'right': self.rarm.link_list,
             'head': self.head.link_list,
             'base': self.base_links}
        self.joint_lists = \
            {chain: [link.joint for link in self.link_lists[chain]] for chain \
             in self.chains}
        self.end_coords = \
            {'left': self.larm_end_coords,
             'right': self.rarm_end_coords,
             'head': self.head_end_coords}
        self.hand_links = \
            {'left' : [self.l_gripper_palm_link,
                       self.l_gripper_r_finger_link,
                       self.l_gripper_l_finger_link,
                       self.l_gripper_r_finger_tip_link,
                       self.l_gripper_l_finger_tip_link],
             'right': [self.r_gripper_palm_link,
                       self.r_gripper_r_finger_link,
                       self.r_gripper_l_finger_link,
                       self.r_gripper_r_finger_tip_link,
                       self.r_gripper_l_finger_tip_link]}
        self.collision_link_lists = \
            {'left': self.link_lists['left'] + \
             [self.l_forearm_link, self.l_upper_arm_link] + self.hand_links['left'],
             'right': self.link_lists['right'] + \
             [self.r_forearm_link, self.r_upper_arm_link] + self.hand_links['right'],
             'head': self.link_lists['head'],
             'base': self.link_lists['base']}
        self.hand_body_names = \
            {h : set([l.name for l in self.hand_links[h]]) \
             for h in ('left', 'right')}
        self.hand_body_names['left'] |= set([self.l_wrist_flex_link, self.l_wrist_roll_link])
        self.hand_body_names['right'] |= set([self.r_wrist_flex_link, self.r_wrist_roll_link])
        self.base_body_names = set([l.name for l in self.base_links])
        self.arm_body_names = \
            {h : set([l.name for l in set(self.collision_link_lists[h]) - set(self.hand_links[h])]) \
             for h in ('left', 'right')}
        # To enable GJK collision checking
        for chain in self.collision_link_lists:
            for link in self.collision_link_lists[chain]:
                link.collision_mesh.convex_mesh = trimesh.convex.convex_hull(link.collision_mesh)
                link.collision_mesh.convex_mesh_vertices = \
                    np.ascontiguousarray(link.collision_mesh.convex_mesh.vertices, dtype=np.double)
                # print('Computing convex_mesh_vertices for', link)
        self.gripper_distance_forward_cache = {}
        self.gripper_distance_inverse_cache = {}
        # Self collision links
        self.self_collision_link_name_pairs = set()
        for link1 in self.collision_link_lists['left']:
            for link2 in self.collision_link_lists['base']:
                if link1 in (self.l_shoulder_pan_link, self.l_shoulder_lift_link,
                             self.l_upper_arm_link, self.l_upper_arm_roll_link) \
                   and link2 in self.base_links:
                    continue
                self.self_collision_link_name_pairs.add(tuple(sorted((link1.name, link2.name))))
            for link2 in self.collision_link_lists['right'] + self.collision_link_lists['head']:
                self.self_collision_link_name_pairs.add(tuple(sorted((link1.name, link2.name))))
        for link1 in self.collision_link_lists['right']:
            for link2 in self.collision_link_lists['base']:
                if link1 in (self.r_shoulder_pan_link, self.r_shoulder_lift_link,
                             self.r_upper_arm_link, self.r_upper_arm_roll_link) \
                   and link2 in self.base_links:
                    continue
                self.self_collision_link_name_pairs.add(tuple(sorted((link1.name, link2.name))))
            for link2 in self.collision_link_lists['head']:
                self.self_collision_link_name_pairs.add(tuple(sorted((link1.name, link2.name))))
                
        # custom min_angle and max_angle for joints
        '''
        joint_list = [
            self.torso_lift_joint, self.l_shoulder_pan_joint,
            self.l_shoulder_lift_joint, self.l_upper_arm_roll_joint,
            self.l_elbow_flex_joint, self.l_forearm_roll_joint,
            self.l_wrist_flex_joint, self.l_wrist_roll_joint,
            self.r_shoulder_pan_joint, self.r_shoulder_lift_joint,
            self.r_upper_arm_roll_joint, self.r_elbow_flex_joint,
            self.r_forearm_roll_joint, self.r_wrist_flex_joint,
            self.r_wrist_roll_joint, self.head_pan_joint, self.head_tilt_joint
        ]
        for j, min_angle, max_angle in zip(
                joint_list,
                (0.0115, np.deg2rad(-32.3493),
                 np.deg2rad(-20.2598), np.deg2rad(-37.2423),
                 np.deg2rad(-121.542), -float('inf'),
                 np.deg2rad(-114.592), -float('inf'), np.deg2rad(-122.349),
                 np.deg2rad(-20.2598), np.deg2rad(-214.859),
                 np.deg2rad(-121.542), -float('inf'),
                 np.deg2rad(-114.592), -float('inf'),
                 np.deg2rad(-163.694), np.deg2rad(-21.2682)),
                (0.325, np.deg2rad(122.349), np.deg2rad(74.2725),
                 np.deg2rad(214.859), np.deg2rad(-8.59437),
                 float('inf'), np.deg2rad(-5.72958),
                 float('inf'), np.deg2rad(32.3493), np.deg2rad(74.2725),
                 np.deg2rad(37.2423), np.deg2rad(-8.59437), float('inf'),
                 np.deg2rad(-5.72958), float('inf'), np.deg2rad(163.694),
                 np.deg2rad(74.2702))):
            j.min_angle = min_angle
            j.max_angle = max_angle
        '''
    @cached_property
    def default_urdf_path(self):
        return pr2_urdfpath()

    @cached_property
    def rarm(self):
        rarm_links = [
            self.r_shoulder_pan_link,
            self.r_shoulder_lift_link,
            self.r_upper_arm_roll_link,
            self.r_elbow_flex_link,
            self.r_forearm_roll_link,
            self.r_wrist_flex_link,
            self.r_wrist_roll_link
        ]

        rarm_joints = []
        for link in rarm_links:
            rarm_joints.append(link.joint)
        r = RobotModel(link_list=rarm_links, joint_list=rarm_joints)
        r.end_coords = self.rarm_end_coords
        r.wrist_coords = self.rarm_wrist_coords
        return r

    @cached_property
    def larm(self):
        larm_links = [
            self.l_shoulder_pan_link,
            self.l_shoulder_lift_link,
            self.l_upper_arm_roll_link,
            self.l_elbow_flex_link,
            self.l_forearm_roll_link,
            self.l_wrist_flex_link,
            self.l_wrist_roll_link,
        ]
        larm_joints = []
        for link in larm_links:
            larm_joints.append(link.joint)
        r = RobotModel(link_list=larm_links, joint_list=larm_joints)
        r.end_coords = self.larm_end_coords
        r.wrist_coords = self.larm_wrist_coords
        return r

    @cached_property
    def head(self):
        links = [
            self.head_pan_link,
            self.head_tilt_link]
        joints = []
        for link in links:
            joints.append(link.joint)
        r = RobotModel(link_list=links, joint_list=joints)
        r.end_coords = self.head_end_coords
        return r

    def reset_manip_pose(self):
        self.torso_lift_joint.joint_angle(0.3)
        self.l_shoulder_pan_joint.joint_angle(np.deg2rad(75))
        self.l_shoulder_lift_joint.joint_angle(np.deg2rad(50))
        self.l_upper_arm_roll_joint.joint_angle(np.deg2rad(110))
        self.l_elbow_flex_joint.joint_angle(np.deg2rad(-110))
        self.l_forearm_roll_joint.joint_angle(np.deg2rad(-20))
        self.l_wrist_flex_joint.joint_angle(np.deg2rad(-10))
        self.l_wrist_roll_joint.joint_angle(np.deg2rad(-10))
        self.r_shoulder_pan_joint.joint_angle(np.deg2rad(-75))
        self.r_shoulder_lift_joint.joint_angle(np.deg2rad(50))
        self.r_upper_arm_roll_joint.joint_angle(np.deg2rad(-110))
        self.r_elbow_flex_joint.joint_angle(np.deg2rad(-110))
        self.r_forearm_roll_joint.joint_angle(np.deg2rad(20))
        self.r_wrist_flex_joint.joint_angle(np.deg2rad(-10))
        self.r_wrist_roll_joint.joint_angle(np.deg2rad(-10))
        self.head_pan_joint.joint_angle(0)
        self.head_tilt_joint.joint_angle(np.deg2rad(50))
        return self.angle_vector()

    def reset_pose(self):
        self.torso_lift_joint.joint_angle(0.3)
        self.l_shoulder_pan_joint.joint_angle(np.deg2rad(60))
        self.l_shoulder_lift_joint.joint_angle(np.deg2rad(74))
        self.l_upper_arm_roll_joint.joint_angle(np.deg2rad(70))
        self.l_elbow_flex_joint.joint_angle(np.deg2rad(-120))
        self.l_forearm_roll_joint.joint_angle(np.deg2rad(20))
        self.l_wrist_flex_joint.joint_angle(np.deg2rad(-30))
        self.l_wrist_roll_joint.joint_angle(np.deg2rad(180))
        self.r_shoulder_pan_joint.joint_angle(np.deg2rad(-60))
        self.r_shoulder_lift_joint.joint_angle(np.deg2rad(74))
        self.r_upper_arm_roll_joint.joint_angle(np.deg2rad(-70))
        self.r_elbow_flex_joint.joint_angle(np.deg2rad(-120))
        self.r_forearm_roll_joint.joint_angle(np.deg2rad(-20))
        self.r_wrist_flex_joint.joint_angle(np.deg2rad(-30))
        self.r_wrist_roll_joint.joint_angle(np.deg2rad(180))
        self.head_pan_joint.joint_angle(0)
        self.head_tilt_joint.joint_angle(0)
        return self.angle_vector()

    def gripper_distance(self, dist=None, arm='arms'):
        """Change gripper angle function

        Parameters
        ----------
        dist : None or float
            gripper distance.
            If dist is None, return gripper distance.
            If flaot value is given, change joint angle.
        arm : str
            Specify target arm.  You can specify 'left', 'right', 'arms'.

        Returns
        -------
        dist : float
            Result of gripper distance in meter.
        """
        if arm == 'left':
            joints = [self.l_gripper_l_finger_joint]
        elif arm == 'right':
            joints = [self.r_gripper_l_finger_joint]
        elif arm == 'arms':
            joints = [self.r_gripper_l_finger_joint,
                      self.l_gripper_l_finger_joint]
        else:
            raise ValueError('Invalid arm arm argument. You can specify '
                             "'left', 'right' or 'arms'.")

        def _dist(angle):
            if angle not in self.gripper_distance_forward_cache:
                self.gripper_distance_forward_cache[angle] = \
                    0.0099 * (18.4586 * np.sin(angle) + np.cos(angle) - 1.0101)
            return self.gripper_distance_forward_cache[angle]

        if dist is not None:
            if dist not in self.gripper_distance_inverse_cache:
                # calculate joint_angle from approximated equation
                max_dist = _dist(joints[0].max_angle)
                dist = max(min(dist, max_dist), 0)
                d = dist / 2.0
                angle = 2 * np.arctan(
                    (9137 - np.sqrt(2)
                     * np.sqrt(-5e9 * (d**2) - 5e7 * d + 41739897))
                    / (5 * (20000 * d + 199)))
                self.gripper_distance_inverse_cache[dist] = angle
            angle = self.gripper_distance_inverse_cache[dist]
            for joint in joints:
                joint.joint_angle(angle)
        angle = joints[0].joint_angle()
        return _dist(angle)
