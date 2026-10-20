#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def landmark_observation_model(z_observed, z_variance, robot_p, beacon_m):
    #expected range from robot to beacon
    z_predicted = np.linalg.norm(np.asarray(beacon_m) - np.asarray(robot_p))
    prob = np.exp(-0.5 * ((z_predicted - z_observed) / z_variance) ** 2) / (np.sqrt(2 * np.pi) * z_variance)
    return prob

if __name__ == '__main__':
    # visualize observation model test
    grid_size = 20
    beacon_m = [[0, 0], [0, 10], [10, 0]]
    z_measured = [10, 5, 9]
    z_sigma = [1, 3, 2]

    likelihood = np.ones([grid_size, grid_size])
    for i in range(grid_size):
        for j in range(grid_size):
            for n in range(len(z_measured)):
                prob = landmark_observation_model(z_measured[n], z_sigma[n], [i, j], beacon_m[n])
                likelihood[i,j] = prob*likelihood[i,j]

    # plot the grid map with likelihood
    # Plot
    plt.figure(figsize=(6, 5))
    plt.imshow(likelihood, cmap="viridis", origin="upper")  # (0,0) at the top-left
    plt.colorbar(label="p(z | x, m)")
    for i, beacon in enumerate(beacon_m):
        plt.scatter(beacon[1], beacon[0], c="red", marker="*", s=200, label="beacon $i$")
    plt.title(f"Observation likelihood")
    plt.xlabel("column")
    plt.ylabel("row")
    plt.legend(loc="lower right")
    plt.show()

    print("max likelihood:", likelihood.max())