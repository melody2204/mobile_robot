#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import bresenham as bh
import os


def plot_gridmap(gridmap, robot_cell=None, robot_theta=None, trajectory=None,
                 endpoints=None, ax=None, title=None):
    """
    draw the current map, the robot pose, its path so far and the current scan
    :param gridmap: occupancy probability map
    :param robot_cell: robot position in cells (x, y)
    :param robot_theta: robot heading in radians
    :param trajectory: list of past robot cells [(x, y), ...]
    :param endpoints: current scan endpoints in cells, shape (2, N)
    """
    if ax is None:
        ax = plt.gca()
    ax.clear()
    ax.imshow(gridmap, cmap='Greys', vmin=0, vmax=1, origin='lower')

    if endpoints is not None and endpoints.size:
        ax.plot(endpoints[0], endpoints[1], '.', color='tab:red',
                markersize=2, label='current scan')
    if trajectory is not None and len(trajectory) > 1:
        traj = np.array(trajectory)
        ax.plot(traj[:, 0], traj[:, 1], '-', color='tab:blue',
                linewidth=1.5, label='path')
    if robot_cell is not None:
        ax.plot(robot_cell[0], robot_cell[1], 'o', color='tab:blue',
                markersize=7, label='robot')
        if robot_theta is not None:
            arrow_len = 0.05 * max(gridmap.shape)  # arrow length in cells
            ax.arrow(robot_cell[0], robot_cell[1],
                     arrow_len * np.cos(robot_theta), arrow_len * np.sin(robot_theta),
                     head_width=arrow_len * 0.4, color='tab:orange')

    ax.set_xlim(0, gridmap.shape[1])
    ax.set_ylim(0, gridmap.shape[0])
    ax.set_xlabel('x [cells]')
    ax.set_ylabel('y [cells]')
    if title:
        ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)


def init_gridmap(size, res):
    gridmap = np.zeros([int(np.ceil(size / res)), int(np.ceil(size / res))])
    return gridmap


def world2map(pose, gridmap, map_res):
    """
    the original point 0, 0 is in the middle of the gridmap
    convert the absolute pose into the grid map cell position
    :param pose:
    :param gridmap:
    :param map_res:
    :return:
    """
    origin = np.array(gridmap.shape) / 2
    new_pose = np.zeros_like(pose)
    new_pose[0] = np.round(pose[0] / map_res) + origin[0];
    new_pose[1] = np.round(pose[1] / map_res) + origin[1];
    return new_pose.astype(int)


def v2t(pose):
    """
    convert the absolute pose into homogeneous transformation matrix
    :param pose: x, y, theta
    :return:
    """
    p_cos = np.cos(pose[2])
    p_sin = np.sin(pose[2])
    tr = np.array([[p_cos, -p_sin, pose[0]], [p_sin, p_cos, pose[1]], [0, 0, 1]])
    return tr


def ranges2points(ranges):
    # laser properties
    start_angle = -1.5708
    angular_res = 0.0087270
    max_range = 30
    # rays within range
    num_beams = ranges.shape[0]
    idx = (ranges < max_range) & (ranges > 0)
    # 2D points
    angles = (start_angle + np.arange(num_beams) * angular_res)[idx]
    points = np.array([np.multiply(ranges[idx], np.cos(angles)), np.multiply(ranges[idx], np.sin(angles))])
    # homogeneous points
    points_hom = np.append(points, np.ones((1, points.shape[1])), axis=0)
    return points_hom


def ranges2cells(r_ranges, w_pose, gridmap, map_res):
    """
    :param r_ranges: beam ranges
    :param w_pose: robot world pose
    :param gridmap:
    :param map_res:
    :return: endpoints of those ranges in gridmap
    """
    # ranges to points
    r_points = ranges2points(r_ranges)
    w_P = v2t(w_pose)
    w_points = np.matmul(w_P, r_points)
    # covert to map frame
    m_points = world2map(w_points, gridmap, map_res)
    m_points = m_points[0:2, :]
    return m_points


def poses2cells(w_pose, gridmap, map_res):
    # covert world pose to map frame
    m_pose = world2map(w_pose, gridmap, map_res)
    return m_pose[0:2]


def bresenham(x0, y0, x1, y1):
    l = np.array(list(bh.bresenham(x0, y0, x1, y1)))
    return l


def prob2logodds(p):
    return np.log(p / (1.0 - p))


def logodds2prob(l):
    return 1.0 / (1 + np.exp(-l))


def inv_sensor_model(cell, endpoint, prob_occ, prob_free):
    if np.array_equal(cell, endpoint):
        return prob_occ
    else:
        return prob_free


def grid_mapping_with_known_poses(ranges_raw, poses_raw, occ_gridmap,
                                  map_res, prob_occ, prob_free, prior,
                                  frames_dir=None):
    assert 0 < prob_free < prior < prob_occ < 1
    l0 = prob2logodds(prior)
    occ_gridmap = prob2logodds(occ_gridmap)
    h, w = occ_gridmap.shape
    trajectory = []

    plt.ion()  # interactive: window updates live
    fig, ax = plt.subplots(figsize=(7, 7))
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)

    for t, (pose, ranges) in enumerate(zip(poses_raw, ranges_raw)):
        robot_cell = poses2cells(pose, occ_gridmap, map_res)
        endpoints = ranges2cells(ranges, pose, occ_gridmap, map_res)
        trajectory.append(robot_cell)

        for endpoint in endpoints.T:  # iterate over points
            for x, y in bresenham(*robot_cell, *endpoint):  # only cells on the beam
                if not (0 <= x < w and 0 <= y < h):
                    continue
                p = inv_sensor_model((x, y), endpoint, prob_occ, prob_free)  # p(m | z_t)
                occ_gridmap[y, x] += prob2logodds(p) - l0  # Bayes update

        # plot after every scan
        title = (f'scan {t + 1}/{len(poses_raw)}   '
                 f'pose x={pose[0]:.2f} m, y={pose[1]:.2f} m, '
                 f'theta={np.degrees(pose[2]):.0f} deg')
        plot_gridmap(logodds2prob(occ_gridmap), robot_cell, pose[2],
                     trajectory, endpoints, ax, title)
        plt.pause(0.01)  # draw and show this frame
        if frames_dir:
            fig.savefig(os.path.join(frames_dir, f'frame_{t:04d}.png'))

    plt.ioff()
    plt.show()  # keep the final map open

    return logodds2prob(occ_gridmap)


if __name__ == '__main__':
    map_size = 100
    map_res = 0.25

    prior = 0.50
    prob_occ = 0.90
    prob_free = 0.35

    # load data
    ranges_raw = np.loadtxt("./data/ranges.data", delimiter=',', dtype='float')
    poses_raw = np.loadtxt("./data/poses.data", delimiter=',', dtype='float')

    # initialize gridmap
    occ_gridmap = init_gridmap(map_size, map_res) + prior
    grid_mapping_with_known_poses(ranges_raw, poses_raw, occ_gridmap,
                                  map_res, prob_occ, prob_free, prior)

