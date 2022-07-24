from cached_property import cached_property
import numpy as np
import trimesh

from skrobot.coordinates import CascadedCoords
from skrobot.data import panda_urdfpath
from skrobot.model import RobotModel

from .urdf import RobotModelFromURDF

class Panda(RobotModelFromURDF):

    """Panda Robot Model.

    https://frankaemika.github.io/docs/control_parameters.html
    """

    def __init__(self, *args, **kwargs):
        super(Panda, self).__init__(*args, **kwargs)

        # Tool
        self.rarm_end_coords = CascadedCoords(
            parent=self.panda_hand,
            name='rarm_end_coords')
        self.rarm_end_coords.translate(
            np.array([0.0, 0.0, 0.1], dtype=np.float32))
        self.rarm_end_coords.rotate(- np.pi , axis='x')
        # Wrist
        self.rarm_wrist_coords = CascadedCoords(
            parent=self.panda_link8,
            name='rarm_wrist_coords')
        self.chains = set(['base', 'right'])
        self.chain_type = {'base': 'fixed',
                           'right': 'revolute_chain'}
        self.base_link = self.panda_link0
        self.base_links = \
            [self.panda_link0]
        self.link_lists = \
            {'right': self.rarm.link_list,
             'base': self.base_links}
        self.joint_lists = \
            {chain: [link.joint for link in self.link_lists[chain]] for chain \
             in self.chains}
        self.end_coords = \
            {'right': self.rarm_end_coords}
        self.hand_links = \
            {'right': [self.panda_hand,
                       self.panda_rightfinger,
                       self.panda_leftfinger]}
        self.collision_link_lists = \
            {'right': self.link_lists['right'] + self.hand_links['right'],
             'base': self.link_lists['base']}
        self.hand_body_names = \
            {h : set([l.name for l in self.hand_links[h]]) \
             for h in ('right',)}
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
        # Self collision links
        self.self_collision_link_name_pairs = set()
        for link1 in self.collision_link_lists['right']:
            for link2 in self.collision_link_lists['base']:
                if link1 in (self.panda_link1,) \
                   and link2 in self.base_links:
                    continue
                self.self_collision_link_name_pairs.add(tuple(sorted((link1.name, link2.name))))
        
        self._reset_manip_pose()

    @cached_property
    def default_urdf_path(self):
        return panda_urdfpath()

    def _reset_manip_pose(self):
        angle_vector = [
            0.03942226991057396,
            -0.9558116793632507,
            -0.014800949953496456,
            -2.130282163619995,
            -0.013104429468512535,
            1.1745525598526,
            0.8112226724624634,
        ]
        for link, angle in zip(self.rarm.link_list, angle_vector):
            link.joint.joint_angle(angle)
        return self.angle_vector()

    reset_manip_pose = _reset_manip_pose

    @cached_property
    def rarm(self):
        link_names = ['panda_link{}'.format(i) for i in range(1, 8)]
        links = [getattr(self, n) for n in link_names]
        joints = [l.joint for l in links]
        model = RobotModel(link_list=links, joint_list=joints)
        model.end_coords = self.panda_hand
        return model

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
            self.panda_finger_joint1.joint_angle(dist/2)
            self.panda_finger_joint2.joint_angle(dist/2)
        return self.panda_finger_joint1.joint_angle() + self.panda_finger_joint2.joint_angle()
            
