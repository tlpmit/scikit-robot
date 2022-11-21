from cached_property import cached_property
import numpy as np
import trimesh
import pdb

from skrobot.coordinates import CascadedCoords
from skrobot.data import fetch_urdfpath
from skrobot.model import RobotModel

from .urdf import RobotModelFromURDF

class Fetch(RobotModelFromURDF):
    """Fetch Robot Model.

    http://docs.fetchrobotics.com/robot_hardware.html
    """
    def __init__(self, *args, **kwargs):
        super(Fetch, self).__init__(*args, **kwargs)

        self.torsoZ = 0.3

        self.rarm_end_coords = CascadedCoords(
            parent=self.gripper_link,
            name='rarm_end_coords')
        self.rarm_end_coords.translate([0.03, 0.0, 0.0], 'world')
        self.rarm_end_coords.rotate(-np.pi/2 , axis='y')        
        
        self.head_end_coords = CascadedCoords(
            pos=[0.08, 0.0, 0.13],        # ?? Is this right - this is for PR2
            parent=self.head_tilt_link,
            name='head_end_coords').rotate(np.pi / 2.0, 'y')
        self.torso_end_coords = CascadedCoords(
            parent=self.torso_lift_link,
            name='torso_end_coords')

        # Wrist
        self.rarm_wrist_coords = CascadedCoords(
            parent=self.wrist_roll_link,
            name='rarm_wrist_coords')
        # limbs
        self.torso = [self.torso_lift_link]
        self.torso_root_link = self.torso_lift_link
        self.rarm_root_link = self.shoulder_pan_link
        self.head_root_link = self.head_pan_link

        self.chains = set(['base', 'right', 'head'])
        self.chain_type = {'base': 'omni_base_chain',
                           'right': 'revolute_chain',
                           'head': 'revolute_chain'}
        self.base_links = \
            [self.base_link, self.torso_lift_link, self.bellows_link2]
        self.link_lists = \
            {'right': self.rarm.link_list,
             'head': self.head.link_list,
             'base': self.base_links}
        self.joint_lists = \
            {chain: [link.joint for link in self.link_lists[chain]] for chain \
             in self.chains}
        self.end_coords = \
            {'right': self.rarm_end_coords,
             'head': self.head_end_coords}
        self.hand_links = \
            {'right': [self.gripper_link,
                       self.r_gripper_finger_link,
                       self.l_gripper_finger_link]}
        self.finger_links = \
            set([self.r_gripper_finger_link,
                 self.l_gripper_finger_link                 
                ])
        self.collision_link_lists = \
            {'right': self.link_lists['right'] + \
             [self.forearm_roll_link, self.upperarm_roll_link] + self.hand_links['right'],
             'head': self.link_lists['head'],
             'base': self.link_lists['base']}
        self.hand_body_names = \
            {h : set([l.name for l in self.hand_links[h]]) \
             for h in ('right',)}
        self.hand_body_names['right'] |= set([self.wrist_flex_link, self.wrist_roll_link])
        self.finger_body_names = set([l.name for l in self.finger_links])
        self.base_body_names = set([l.name for l in self.base_links])
        self.arm_body_names = \
            {h : set([l.name for l in set(self.collision_link_lists[h]) - set(self.hand_links[h])]) \
             for h in ('right',)}
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
        for link1 in self.collision_link_lists['right']:
            for link2 in self.collision_link_lists['base']:
                if link1 in (self.shoulder_pan_link, self.shoulder_lift_link,
                             self.upperarm_roll_link) \
                   and link2 in self.base_links:
                    continue
                self.self_collision_link_name_pairs.add(tuple(sorted((link1.name, link2.name))))
            for link2 in self.collision_link_lists['head']:
                self.self_collision_link_name_pairs.add(tuple(sorted((link1.name, link2.name))))
        

    @cached_property
    def default_urdf_path(self):
        return fetch_urdfpath()

    def reset_pose(self):
        self.torso_lift_joint.joint_angle(self.torsoZ)
        self.shoulder_pan_joint.joint_angle(np.deg2rad(75.6304))
        self.shoulder_lift_joint.joint_angle(np.deg2rad(80.2141))
        self.upperarm_roll_joint.joint_angle(np.deg2rad(-11.4592))
        self.elbow_flex_joint.joint_angle(np.deg2rad(98.5487))
        self.forearm_roll_joint.joint_angle(0.0)
        self.wrist_flex_joint.joint_angle(np.deg2rad(95.111))
        self.wrist_roll_joint.joint_angle(0.0)
        self.head_pan_joint.joint_angle(0.0)
        self.head_tilt_joint.joint_angle(0.0)
        return self.angle_vector()

    def reset_manip_pose(self):
        self.torso_lift_joint.joint_angle(self.torsoZ)
        self.shoulder_pan_joint.joint_angle(0)
        self.shoulder_lift_joint.joint_angle(0)
        self.upperarm_roll_joint.joint_angle(0)
        self.elbow_flex_joint.joint_angle(np.pi / 2.0)
        self.forearm_roll_joint.joint_angle(0)
        self.wrist_flex_joint.joint_angle(- np.pi / 2.0)
        self.wrist_roll_joint.joint_angle(0)
        self.head_pan_joint.joint_angle(0)
        self.head_tilt_joint.joint_angle(0)
        return self.angle_vector()

    @cached_property
    def rarm(self):
        rarm_links = [self.shoulder_pan_link,
                      self.shoulder_lift_link,
                      self.upperarm_roll_link,
                      self.elbow_flex_link,
                      self.forearm_roll_link,
                      self.wrist_flex_link,
                      self.wrist_roll_link]
        rarm_joints = []
        for link in rarm_links:
            rarm_joints.append(link.joint)
        r = RobotModel(link_list=rarm_links,
                       joint_list=rarm_joints)
        r.end_coords = self.rarm_end_coords
        r.wrist_coords = self.rarm_wrist_coords        
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


    def gripper_distance(self, dist=None, arm='right'):
        """Change gripper angle function

        Parameters
        ----------
        dist : None or float
            gripper distance.
            If dist is None, return gripper distance.
            If flaot value is given, change joint angle.
        arm : str
            Specify target arm.  You can only specify 'right'

        Returns
        -------
        dist : float
            Result of gripper distance in meter.
        """
        if dist is not None:
            self.l_gripper_finger_joint.joint_angle(dist/2)
            self.r_gripper_finger_joint.joint_angle(dist/2)
        return self.l_gripper_finger_joint.joint_angle() + self.r_gripper_finger_joint.joint_angle()
    
