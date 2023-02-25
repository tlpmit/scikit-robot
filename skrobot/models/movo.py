from cached_property import cached_property
import numpy as np
import trimesh

from skrobot.coordinates import CascadedCoords
from skrobot.data import movo_urdfpath
from skrobot.model import RobotModel

from .urdf import RobotModelFromURDF

class Movo(RobotModelFromURDF):

    """Movo Robot Model.

    """

    def __init__(self, *args, **kwargs):
        super(Movo, self).__init__(*args, **kwargs)

        self.torsoZ = 0.3

        self.rarm_end_coords = CascadedCoords(
            parent=self.right_ee_link,
            name='rarm_end_coord')
        self.larm_end_coords = CascadedCoords(
            parent=self.left_ee_link,
            name='larm_end_coords')
        self.head_end_coords = CascadedCoords(
            pos=[0.08, 0.0, 0.13],        # TODO: Check
            parent=self.tilt_link,
            name='head_end_coords').rotate(np.pi / 2.0, 'y')
        self.torso_end_coords = CascadedCoords(
            parent=self.linear_actuator_link,
            name='torso_end_coords')

        # Wrist
        self.rarm_wrist_coords = CascadedCoords(
            parent=self.right_wrist_3_link,
            name='rarm_wrist_coords')
        self.larm_wrist_coords = CascadedCoords(
            parent=self.left_wrist_3_link,
            name='larm_wrist_coords')

        # limbs
        self.torso = [self.linear_actuator_link]
        self.torso_root_link = self.linear_actuator_link
        self.larm_root_link = self.left_shoulder_link
        self.rarm_root_link = self.right_shoulder_link
        self.head_root_link = self.pan_link

        self.chains = set(['base', 'right', 'left', 'head'])
        self.chain_type = {'base': 'omni_base_chain',
                           'right': 'revolute_chain',
                           'left': 'revolute_chain',
                           'head': 'revolute_chain'}
        self.base_links = \
            [self.base_chassis_link, self.linear_actuator_link]
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
            {'left' : [self.left_gripper_base_link,
                       self.left_gripper_finger1_knuckle_link,                       
                       self.left_gripper_finger1_finger_tip_link,
                       self.left_gripper_finger2_knuckle_link,
                       self.left_gripper_finger2_finger_tip_link,
                       self.left_gripper_finger3_knuckle_link,
                       self.left_gripper_finger3_finger_tip_link,                       
             ],
             'right': [self.right_gripper_base_link,
                       self.right_gripper_finger1_finger_link,                                              
                       self.right_gripper_finger2_finger_link,                                              
                       self.right_gripper_finger1_knuckle_link,                                              
                       self.right_gripper_finger2_knuckle_link,                       
                       self.right_gripper_finger1_inner_knuckle_link,
                       self.right_gripper_finger1_finger_tip_link,
                       self.right_gripper_finger2_inner_knuckle_link,
                       self.right_gripper_finger2_finger_tip_link,                       
             ]}
        self.finger_links = \
            set([
                       self.left_gripper_finger1_knuckle_link,                       
                       self.left_gripper_finger1_finger_tip_link,
                       self.left_gripper_finger2_knuckle_link,
                       self.left_gripper_finger2_finger_tip_link,
                       self.left_gripper_finger3_knuckle_link,
                       self.left_gripper_finger3_finger_tip_link,
                       self.right_gripper_finger1_finger_link,                                              
                       self.right_gripper_finger2_finger_link,                                              
                       self.right_gripper_finger1_knuckle_link,                                              
                       self.right_gripper_finger2_knuckle_link,                       
                       self.right_gripper_finger1_inner_knuckle_link,
                       self.right_gripper_finger1_finger_tip_link,
                       self.right_gripper_finger2_inner_knuckle_link,
                       self.right_gripper_finger2_finger_tip_link,                       

             ])
        self.collision_link_lists = \
            {'left': self.link_lists['left'] + self.hand_links['left'],
             'right': self.link_lists['right'] + self.hand_links['right'],
             'head': self.link_lists['head'],
             'base': self.link_lists['base']}
        self.hand_body_names = \
            {h : set([l.name for l in self.hand_links[h]]) \
             for h in ('left', 'right')}
        self.hand_body_names['left'] |= set([self.left_wrist_spherical_1_link, self.left_wrist_spherical_2_link])
        self.hand_body_names['right'] |= set([self.right_wrist_spherical_1_link, self.right_wrist_spherical_2_link])
        self.finger_body_names = set([l.name for l in self.finger_links])
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
        '''
        for link1 in self.collision_link_lists['left']:
            for link2 in self.collision_link_lists['base']:
                if link1 in (self.left_shoulder_link, self.left_arm_half_1_link) \
                   and link2 in self.base_links:
                    continue
                self.self_collision_link_name_pairs.add(tuple(sorted((link1.name, link2.name))))
            for link2 in self.collision_link_lists['right'] + self.collision_link_lists['head']:
                self.self_collision_link_name_pairs.add(tuple(sorted((link1.name, link2.name))))
        '''
        for link1 in self.collision_link_lists['right']:
            for link2 in self.collision_link_lists['base']:
                if link1 in (self.right_shoulder_link, self.right_arm_half_1_link) \
                   and link2 in self.base_links:
                    continue
                self.self_collision_link_name_pairs.add(tuple(sorted((link1.name, link2.name))))
            for link2 in self.collision_link_lists['head']:
                self.self_collision_link_name_pairs.add(tuple(sorted((link1.name, link2.name))))
                
    @cached_property
    def default_urdf_path(self):
        return movo_urdfpath()

    @cached_property
    def rarm(self):
        rarm_links = [
            self.right_shoulder_link,
            self.right_arm_half_1_link,
            self.right_arm_half_2_link,            
            self.right_forearm_link,
            self.right_wrist_spherical_1_link,
            self.right_wrist_spherical_2_link,            
            self.right_wrist_3_link
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
            self.left_shoulder_link,
            self.left_arm_half_1_link,
            self.left_arm_half_2_link,            
            self.left_forearm_link,
            self.left_wrist_spherical_1_link,
            self.left_wrist_spherical_2_link,            
            self.left_wrist_3_link
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
            self.pan_link,
            self.tilt_link]
        joints = []
        for link in links:
            joints.append(link.joint)
        r = RobotModel(link_list=links, joint_list=joints)
        r.end_coords = self.head_end_coords
        return r

    def reset_manip_pose(self):

        self.linear_joint.joint_angle(self.torsoZ)
        
        self.right_shoulder_pan_joint.joint_angle(np.deg2rad(0))
        self.right_shoulder_lift_joint.joint_angle(np.deg2rad(-90))
        self.right_arm_half_joint.joint_angle(np.deg2rad(0))            
        self.right_elbow_joint.joint_angle(np.deg2rad(-90))
        self.right_wrist_spherical_1_joint.joint_angle(np.deg2rad(0))
        self.right_wrist_spherical_2_joint.joint_angle(np.deg2rad(0))            
        self.right_wrist_3_joint.joint_angle(np.deg2rad(0))

        self.left_shoulder_pan_joint.joint_angle(np.deg2rad(90))
        self.left_shoulder_lift_joint.joint_angle(np.deg2rad(90))
        self.left_arm_half_joint.joint_angle(np.deg2rad(0))            
        self.left_elbow_joint.joint_angle(np.deg2rad(0))
        self.left_wrist_spherical_1_joint.joint_angle(np.deg2rad(0))
        self.left_wrist_spherical_2_joint.joint_angle(np.deg2rad(0))            
        self.left_wrist_3_joint.joint_angle(np.deg2rad(0))

        self.left_gripper_finger1_joint.joint_angle(np.deg2rad(45))
        self.right_gripper_finger1_joint.joint_angle(np.deg2rad(0))        

        self.pan_joint.joint_angle(np.deg2rad(0))
        self.tilt_joint.joint_angle(np.deg2rad(-50))
                    
        return self.angle_vector()

    def reset_pose(self):
        return self.reset_manip_pose()

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
            joints = [self.left_gripper_finger1_joint]
        elif arm == 'right':
            joints = [self.right_gripper_finger1_joint]
        elif arm == 'arms':
            joints = [self.left_gripper_finger1_joint,
                     self.right_gripper_finger1_joint
                ]
        else:
            raise ValueError('Invalid arm arm argument. You can specify '
                             "'left', 'right' or 'arms'.")

        # TODO: This is based on a few readings of angle vs distance.

        def _dist(angle):
            ang = np.rad2deg(angle)
            return 0.085 - 2*0.001*ang

        if dist is not None:
            max_dist = _dist(0.)
            dist = max(min(dist, max_dist), 0)
            angle = max(0, np.deg2rad(1000*(0.085 - dist)/2))
            for joint in joints:
                joint.joint_angle(angle)
        angle = joints[0].joint_angle()
        return _dist(angle)

