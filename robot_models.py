#!/usr/bin/env python

import time

import numpy as np

import skrobot

import pdb


def _get_tile_shape(num, hw_ratio=1):
    r_num = int(round(np.sqrt(num / hw_ratio)))  # weighted by wh_ratio
    c_num = 0
    while r_num * c_num < num:
        c_num += 1
    while (r_num - 1) * c_num >= num:
        r_num -= 1
    return r_num, c_num


def main():
    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(2*640, 2*480))

    robots = [
        skrobot.models.Spot()
    ]
    nrow, ncol = _get_tile_shape(len(robots))
    row, col = 2, 2

    for i in range(nrow):
        for j in range(ncol):
            try:
                robot = robots[i * nrow + j]
            except IndexError:
                break
            plane = skrobot.model.Box(extents=(row - 0.01, col - 0.01, 0.01))
            plane.translate((row * i, col * j, -0.01))
            viewer.add(plane)
            robot.translate((row * i, col * j, 0))
            viewer.add(robot)

    viewer.set_camera(angles=[np.deg2rad(30), 0, 0])
    dist = 0.08
    print('Setting gripper to', dist)
    robot.gripper_distance(dist)
    print(robot.arm_link_sh0.worldcoords().T())    
    print(robot.rarm_end_coords.worldcoords().T())

    viewer.show()
    viewer.redraw()

    print('==> Press [q] to close window')
    while not viewer.has_exit:
        time.sleep(0.1)
        viewer.redraw()


if __name__ == '__main__':
    main()
