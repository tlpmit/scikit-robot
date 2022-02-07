import pyrender
import trimesh
import numpy as np

import collections
import logging

from .. import model as model_module

logger = logging.getLogger('trimesh')
logger.setLevel(logging.ERROR)


class TrimeshSceneViewer(object):

    def __init__(self, resolution=None):
        print('Created pyrender viewer')
        self._links = collections.OrderedDict()
        self.scene = trimesh.Scene()
        self.pr_scene = pyrender.Scene()
        self.v = pyrender.Viewer(self.pr_scene, use_raymond_lighting=True, run_in_thread=True)

    def show(self):
        pass

    def redraw(self):
        pass

    def _reset_scene(self):
        self.v.render_lock.acquire()
        # self.pr_scene.clear()
        # convert trimesh geometries to pyrender geometries
        geometries = {name: pyrender.Mesh.from_trimesh(geom, smooth=False)
                      for name, geom in self.scene.geometry.items()}
        # add every node with geometry to the pyrender scene
        for node in self.scene.graph.nodes_geometry:
            pose, geom_name = self.scene.graph[node]
            self.pr_scene.add(geometries[geom_name], pose=pose)
        self.v.render_lock.release()

    def _add_link(self, link):
        assert isinstance(link, model_module.Link)

        link_id = str(id(link))
        if link_id in self._links:
            return
        transform = link.worldcoords().T()
        mesh = link.visual_mesh
        # TODO(someone) fix this at trimesh's scene.
        if (isinstance(mesh, list) or isinstance(mesh, tuple)) \
           and len(mesh) > 0:
            mesh = trimesh.util.concatenate(mesh)
        self.scene.add_geometry(
            geometry=mesh,
            node_name=link_id,
            geom_name=link_id,
            transform=transform,
        )
        self._links[link_id] = link

        for child_link in link._child_links:
            self._add_link(child_link)

    def add(self, geometry):
        if isinstance(geometry, model_module.Link):
            links = [geometry]
        elif isinstance(geometry, model_module.CascadedLink):
            links = geometry.link_list
        else:
            raise TypeError('geometry must be Link or CascadedLink')

        for link in links:
            self._add_link(link)
        self._reset_scene()

    def delete(self, geometry):
        if isinstance(geometry, model_module.Link):
            links = [geometry]
        elif isinstance(geometry, model_module.CascadedLink):
            links = geometry.link_list
        else:
            raise TypeError('geometry must be Link or CascadedLink')

        for link in links:
            link_id = str(id(link))
            if link_id not in self._links:
                continue
            self.scene.delete_geometry(link_id)
            self._links.pop(link_id)
        self._reset_scene()

    def set_camera(self, *args, **kwargs):
        pass
