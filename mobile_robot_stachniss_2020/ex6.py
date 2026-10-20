#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import pickle

from ex4 import inverse_motion_model, normalize_angle


def plot_state(mu, S, M):

    # initialize figure
    ax = plt.gca()
    ax.set_xlim([np.min(M[:, 0]) - 2, np.max(M[:, 0]) + 2])
    ax.set_ylim([np.min(M[:, 1]) - 2, np.max(M[:, 1]) + 2])
    plt.plot(M[:, 0], M[:, 1], '^r')
    plt.title('EKF Localization')

    # visualize result
    plt.plot(mu[0], mu[1], '.b')
    plot_2dcov(mu, S)
    plt.draw()
    plt.pause(0.01)


def plot_2dcov(mu, cov):

    # covariance only in x,y
    d, v = np.linalg.eigh(cov[:-1, :-1])

    # ellipse orientation
    a = np.sqrt(d[0])
    b = np.sqrt(d[1])

    # compute ellipse orientation
    if (v[0, 0] == 0):
        theta = np.pi / 2
    else:
        theta = np.arctan2(v[1, 0], v[0, 0])

    # create an ellipse
    ellipse = Ellipse((mu[0], mu[1]),
                      width=a * 2,
                      height=b * 2,
                      angle=np.rad2deg(theta),
                      edgecolor='blue',
                      alpha=0.3)

    ax = plt.gca()

    return ax.add_patch(ellipse)


def wrapToPi(theta):
    while theta < -np.pi:
        theta = theta + 2 * np.pi
    while theta > np.pi:
        theta = theta - 2 * np.pi
    return theta

def ekf_predict(current_pose_mean, current_pose_sigma,  odometry, R):
    rot1, trans, rot2 = inverse_motion_model(odometry)
    x,  y, theta = current_pose_mean
    predict_pose_mean = np.array([x + trans*np.cos(theta + rot1),
                                  y + trans*np.sin(theta + rot1),
                                  wrapToPi(theta + rot1 + rot2)])

    #Jacobian matrix
    G = np.array([[1.0, 0.0, -trans*np.sin(theta + rot1)],
                  [0.0, 1.0, trans*np.cos(theta + rot1)],
                  [0.0, 0.0, 1.0]])

    predict_pose_sigma = G @ current_pose_sigma @ G.T + R

    return predict_pose_mean, predict_pose_sigma

def ekf_correct(predict_pose_mean, predict_pose_sigma, measurement_z, measurement_noise_q, landmark_m):
    mu = np.array(predict_pose_mean, dtype=float)
    sigma = np.array(predict_pose_sigma, dtype=float)

    for i in range(measurement_z.shape[1]):
        # get measurement and landmark
        distance, angle, lm_id = measurement_z[:, i]
        landmark_x, landmark_y = landmark_m[int(lm_id)]

        # expected measurement from the predicted pose
        dx, dy = landmark_x - mu[0], landmark_y - mu[1]
        square_expected_dist = dx**2 +dy**2
        measurement_z_ = np.array([np.sqrt(square_expected_dist),
                                   normalize_angle(np.arctan2(dy, dx) - mu[2])])

        # Jacobian of measurement model w.r.t. the pose
        H = np.array([[-dx/np.sqrt(square_expected_dist), -dy/np.sqrt(square_expected_dist), 0.0],
                      [dy / square_expected_dist, -dx / square_expected_dist, -1.0]
                      ])

        # Kalman gain
        S = H @ sigma @ H.T + measurement_noise_q
        K = sigma @ H.T @ np.linalg.inv(S)

        innovation = np.array([distance, angle]) - measurement_z_
        innovation[1] = normalize_angle(innovation[1])

        mu = mu + K @ innovation
        mu[2] = normalize_angle(mu[2])
        sigma = (np.eye(3) - K @ H) @ sigma

        return mu, sigma


if __name__ == "__main__":
    # read dataset
    data = pickle.load(open("data/dataset_2d_landmarks.p", "rb"))
    print(data.keys())

    # get landmark coordinates
    M = data['M']

    # get ground truth trajectory
    gt_traj = data['gt']

    # show map
    plt.figure(1)
    plt.plot(M[:, 0], M[:, 1], '^r')

    # show ground truth trajectory
    for i in range(0, len(gt_traj), 10):
        plt.plot(gt_traj[i][0], gt_traj[i][1], '.b')
    # initial set up
    # 3x3 process noise
    sigma_x = 0.25  # [m]
    sigma_y = 0.25  # [m]
    sigma_theta = np.deg2rad(10)  # [rad]
    R = np.diag(np.array([sigma_x, sigma_y, sigma_theta]) ** 2)

    # 2x2 observation noise
    sigma_r = 0.1  # [m]
    sigma_phi = np.deg2rad(5)  # [rad]
    Q = np.diag(np.array([sigma_r, sigma_phi]) ** 2)

    # initial state
    mu = np.array([2, 2, np.pi / 2])
    S = np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.pi / 3]])

    # visualize
    plt.figure()
    plot_state(mu, S, M)

    # show ground truth trajectory
    for t in range(len(data['odom'])):
        mu, S = ekf_predict(mu, S, [data['odom'][t-1], data['odom'][t]], R)
        mu, S = ekf_correct(mu, S, data['z'][t], Q, M)
        plot_state(mu, S, M)
