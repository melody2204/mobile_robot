#!/usr/bin/env python3

import pickle

import matplotlib.pyplot as plt
import numpy as np


def world2map(pose, gridmap, map_res):
    max_y = np.size(gridmap, 0) - 1
    new_pose = np.zeros_like(pose)
    new_pose[0] = np.round(pose[0] / map_res)
    new_pose[1] = max_y - np.round(pose[1] / map_res)
    return new_pose.astype(int)


def v2t(pose):
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
    return tr


def t2v(tr):
    x = tr[0, 2]
    y = tr[1, 2]
    th = np.arctan2(tr[1, 0], tr[0, 0])
    v = np.array([x, y, th])
    return v


def ranges2points(ranges, angles):
    # rays within range
    max_range = 80
    idx = (ranges < max_range) & (ranges > 0)
    # 2D points
    points = np.array([
        np.multiply(ranges[idx], np.cos(angles[idx])),
        np.multiply(ranges[idx], np.sin(angles[idx]))
    ])
    # homogeneous points
    points_hom = np.append(points, np.ones((1, np.size(points, 1))), axis=0)
    return points_hom


def ranges2cells(r_ranges, r_angles, w_pose, gridmap, map_res):
    # ranges to points
    r_points = ranges2points(r_ranges, r_angles)
    w_P = v2t(w_pose)
    w_points = np.matmul(w_P, r_points)
    # world to map
    m_points = world2map(w_points, gridmap, map_res)
    m_points = m_points[0:2, :]
    return m_points


def poses2cells(w_pose, gridmap, map_res):
    # covert to map frame
    m_pose = world2map(w_pose, gridmap, map_res)
    return m_pose


def init_uniform(num_particles, img_map, map_res):
    particles = np.zeros((num_particles, 4))
    particles[:, 0] = np.random.rand(num_particles) * np.size(img_map,
                                                              1) * map_res
    particles[:, 1] = np.random.rand(num_particles) * np.size(img_map,
                                                              0) * map_res
    particles[:, 2] = np.random.rand(num_particles) * 2 * np.pi
    particles[:, 3] = 1.0
    return particles


def free_space(img_map):
    """White pixels are free space (works for 0-1 or 0-255, gray or RGB images)."""
    img = np.asarray(img_map, dtype=float)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    return img > 0.5 * img.max()


def init_free_space(num_particles, free, map_res):
    """like init_uniform, but only in free cells"""
    cells = np.argwhere(free)                                    # (row, col)
    pick = cells[np.random.randint(len(cells), size=num_particles)]
    max_y = free.shape[0] - 1
    particles = np.zeros((num_particles, 4))
    particles[:, 0] = pick[:, 1] * map_res                       # col -> x
    particles[:, 1] = (max_y - pick[:, 0]) * map_res             # row -> y
    particles[:, 2] = normalize_angle(np.random.rand(num_particles) * 2 * np.pi)
    particles[:, 3] = 1.0 / num_particles
    return particles


def compute_likelihood_map(free, map_res, sigma=0.5):
    """likelihood field: Gaussian of the distance to the nearest occupied cell"""
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(free) * map_res                # distance to nearest non-free cell [m]
    return np.exp(-0.5 * (dist / sigma) ** 2)


def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def sample_motion_model(particles, odo_pair, noise):
    """odometry motion model, sampled: odo_pair = (odom[t-1], odom[t])"""
    (x0, y0, th0), (x1, y1, th1) = odo_pair
    dx, dy = x1 - x0, y1 - y0
    trans = np.sqrt(dx ** 2 + dy ** 2)
    rot1 = normalize_angle(np.arctan2(dy, dx) - th0) if trans > 1e-6 else 0.0
    rot2 = normalize_angle(th1 - th0 - rot1)

    a1, a2, a3, a4 = noise
    n = particles.shape[0]
    r1 = rot1 + np.random.normal(0, np.sqrt(a1 * rot1 ** 2 + a2 * trans ** 2), n)
    t = trans + np.random.normal(0, np.sqrt(a3 * trans ** 2 + a4 * (rot1 ** 2 + rot2 ** 2)), n)
    r2 = rot2 + np.random.normal(0, np.sqrt(a1 * rot2 ** 2 + a2 * trans ** 2), n)

    new = particles.copy()
    new[:, 0] += t * np.cos(particles[:, 2] + r1)
    new[:, 1] += t * np.sin(particles[:, 2] + r1)
    new[:, 2] = normalize_angle(particles[:, 2] + r1 + r2)
    return new


def compute_weight(particle, observe_z, gridmap, map_res, likelihood_map, p_outside=1e-3):
    """log-weight of ONE particle: sum of log-likelihoods at its beam endpoints"""
    endpoints = ranges2cells(r_ranges = observe_z[0, :], r_angles = observe_z[1, :], w_pose=particle,
                             gridmap=gridmap, map_res=map_res).T
    log_weight = 0.0
    height, width = gridmap.shape[:2]

    for x, y in endpoints:                       # x = column, y = row
        x, y = int(x), int(y)
        if 0 <= x < width and 0 <= y < height:
            log_weight += np.log(max(float(likelihood_map[y, x]), 1e-12))
        else:
            log_weight += np.log(p_outside)      # endpoint outside the map: unlikely

    return float(log_weight)

def resample(weights, particles):
    """
    low variance resampling
    pick one random offset r between 0 and 1/n
    place N pointers exactly 1/n apart starting at r
    Parameters
    ----------
    weights
    particles

    Returns
    -------

    """
    n = particles.shape[0]
    cum_weights = np.cumsum(weights) / np.sum(weights)
    cum_weights[-1] = 1.0

    r = np.random.uniform(0, 1.0/n)
    pointers = r + np.arange(n) / n
    idx = np.searchsorted(cum_weights, pointers, side="right")
    new_particles = particles[np.minimum(idx, n - 1)].copy()
    new_particles[:, 3] = 1.0 / n
    return new_particles

def estimate_pose(particles):
    """weighted mean, circular mean for the heading"""
    w = particles[:, 3] / np.sum(particles[:, 3])
    th = np.arctan2(np.sum(w * np.sin(particles[:, 2])), np.sum(w * np.cos(particles[:, 2])))
    return np.array([np.sum(w * particles[:, 0]), np.sum(w * particles[:, 1]), th])

def mc_localization(odometries, observations_z, image_map, likelihood_map, map_res, num_particle, noise,
                    gt=None, beam_step=4, plot_every=5):
    free = free_space(image_map)
    particles = init_free_space(num_particle, free, map_res)
    estimates = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    max_y = np.size(image_map, 0) - 1

    for t in range(1, len(odometries)):
        # prediction with u_t = (odom[t-1], odom[t])
        particles = sample_motion_model(particles, [odometries[t - 1], odometries[t]], noise)

        # correction: one weight per particle
        ob_z = np.asarray(observations_z[t])[:, ::beam_step]
        log_w = np.array([compute_weight(p, ob_z, image_map, map_res, likelihood_map) for p in particles])
        particle_weights = np.exp(log_w - np.max(log_w))
        particle_weights /= np.sum(particle_weights)
        particles[:, 3] = particle_weights

        estimates.append(estimate_pose(particles))

        particles = resample(particle_weights, particles)

        if t % plot_every == 0 or t == len(odometries) - 1:
            ax.clear()
            ax.imshow(image_map, cmap="gray")
            ax.plot(particles[:, 0] / map_res, max_y - particles[:, 1] / map_res, '.b', markersize=1)
            e = np.array(estimates)
            ax.plot(e[:, 0] / map_res, max_y - e[:, 1] / map_res, '-r', label='estimate')
            if gt is not None:
                g = np.array(gt)[1:t + 1]
                ax.plot(g[:, 0] / map_res, max_y - g[:, 1] / map_res, '-g', label='ground truth')
            ax.set_xlim(0, np.size(image_map, 1))
            ax.set_ylim(np.size(image_map, 0), 0)
            ax.set_title(f'MCL step {t}')
            ax.legend(loc='upper right')
            plt.pause(0.01)

    plt.ioff()
    plt.show()
    return particles, np.array(estimates)

def plot_particles(particles, img_map, map_res):
    plt.matshow(img_map, cmap="gray")
    max_y = np.size(img_map, 0) - 1
    xs = np.copy(particles[:, 0]) / map_res
    ys = max_y - np.copy(particles[:, 1]) / map_res
    plt.plot(xs, ys, '.b')
    plt.xlim(0, np.size(img_map, 1))
    plt.ylim(np.size(img_map, 0), 0)
    plt.show()


if __name__ == '__main__':
    #load data and show it
    data = pickle.load(open("./data/dataset_mit_csail.p", "rb"))
    print("dataset keys:", list(data.keys()))
    plt.matshow(data['img_map'], cmap="gray")
    plt.show()

    #set up
    num_particles = 5000
    map_res = 0.1
    noise = [0.1, 0.1, 0.05, 0.05]

    particles = init_uniform(num_particles, data['img_map'], map_res)
    plot_particles(particles, data['img_map'], map_res)

    # likelihood field: from the dataset if available, otherwise computed from the map
    if 'likelihood_map' in data:
        likelihood_map = data['likelihood_map']
    else:
        likelihood_map = compute_likelihood_map(free_space(data['img_map']), map_res)

    particles, estimates = mc_localization(data['odom'], data['z'], data['img_map'], likelihood_map,
                                           map_res, num_particles, noise, gt=data.get('gt'))

    if 'gt' in data:
        gt = np.array(data['gt'])[1:len(estimates) + 1]
        err = np.linalg.norm(estimates[:, :2] - gt[:, :2], axis=1)
        print(f"final position error: {err[-1]:.3f} m")