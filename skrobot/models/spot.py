from cached_property import cached_property
import numpy as np
import trimesh
import pdb

from skrobot.coordinates import CascadedCoords
from skrobot.data import spot_urdfpath
from skrobot.model import RobotModel

from .urdf import RobotModelFromURDF

class Spot(RobotModelFromURDF):
    """Spot Robot Model.
    """
    def __init__(self, *args, **kwargs):
        super(Spot, self).__init__(*args, **kwargs)

        self.rarm_end_coords = CascadedCoords(
            parent=self.arm0_end_effector_link,
            name='rarm_end_coords')
        # self.rarm_end_coords.translate([0.03, 0.0, 0.0], 'world')
        # self.rarm_end_coords.rotate(-np.pi/2 , axis='y')        
        
        # Wrist
        self.rarm_wrist_coords = CascadedCoords(
            parent=self.arm0_link_wr1,
            name='rarm_wrist_coords')
        # limbs
        self.rarm_root_link = self.arm0_link_sh0

        self.chains = set(['base', 'right'])
        self.chain_type = {'base': 'omni_base_chain',
                           'right': 'revolute_chain'}
        self.base_links = \
            [self.world_link, self.base,
             self.fl_hip, self.fl_uleg, self.fl_lleg, self.fr_hip, self.fr_uleg, self.fr_lleg,
             self.hl_hip, self.hl_uleg, self.hl_lleg, self.hr_hip, self.hr_uleg, self.hr_lleg,
             ]
        self.link_lists = \
            {'right': self.rarm.link_list,
             'base': self.base_links}
        self.joint_lists = \
            {chain: [link.joint for link in self.link_lists[chain]] for chain \
             in self.chains}
        self.end_coords = \
            {'right': self.rarm_end_coords}
        self.hand_links = \
            {'right': [self.arm0_link_wr1, self.arm0_link_fngr]}
        self.finger_links = \
            set([self.arm0_link_fngr,                 
                ])
        self.collision_link_lists = \
            {'right': list(set(self.link_lists['right'] + self.hand_links['right'])),
             'base': self.link_lists['base']}
        self.hand_body_names = \
            {h : set([l.name for l in self.hand_links[h]]) \
             for h in ('right',)}
        self.finger_body_names = set([l.name for l in self.finger_links])
        self.base_body_names = set([l.name for l in self.base_links])
        self.arm_body_names = \
            {h : set([l.name for l in set(self.collision_link_lists[h]) - set(self.hand_links[h])]) \
             for h in ('right',)}
        # To enable GJK collision checking
        for chain in self.collision_link_lists:
            for link in self.collision_link_lists[chain]:
                if link.collision_mesh is None:
                    link.collision_mesh = link.visual_mesh[0]
                    link.collision_meshes = [link.collision_mesh]
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
                if link1 in (self.arm0_link_sh0, self.arm0_link_sh1,) \
                   and link2 in self.base_links:
                    continue
                self.self_collision_link_name_pairs.add(tuple(sorted((link1.name, link2.name))))

    @cached_property
    def default_urdf_path(self):
        return spot_urdfpath()

    def reset_manip_pose(self):
        for j in [self.fl_hx, self.fl_hy,
                  self.fr_hx, self.fr_hy,
                  self.hl_hx, self.hl_hy,
                  self.hr_hx, self.hr_hy,
                  ]:
            j.joint_angle(0)
        for j in [self.fl_kn, self.fr_kn, self.hl_kn, self.hr_kn,
                  ]:
            j.joint_angle(-0.3)            
        self.arm0_sh0.joint_angle(0)
        self.arm0_sh1.joint_angle(0)
        self.arm0_hr0.joint_angle(0)
        self.arm0_el0.joint_angle(0)
        self.arm0_el1.joint_angle(0)
        self.arm0_wr0.joint_angle(0)
        self.arm0_wr1.joint_angle(0)
        return self.angle_vector()

    @cached_property
    def rarm(self):
        rarm_links = [self.arm0_link_sh0,
                      self.arm0_link_sh1,
                      self.arm0_link_hr0,
                      self.arm0_link_el0,
                      self.arm0_link_el1,
                      self.arm0_link_wr0,
                      self.arm0_link_wr1]
        rarm_joints = []
        for link in rarm_links:
            rarm_joints.append(link.joint)
        r = RobotModel(link_list=rarm_links,
                       joint_list=rarm_joints)
        r.end_coords = self.rarm_end_coords
        r.wrist_coords = self.rarm_wrist_coords        
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
        finger_length = 0.1
        if dist is not None:
            angle = -np.arctan2(dist, finger_length)
            self.arm0_f1x.joint_angle(angle)
        return np.cos(-self.arm0_f1x.joint_angle())*finger_length
    
