#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# icp_known_corresp: performs icp given that the input datasets
# are aligned so that Line1(:, QInd(k)) corresponds to Line2(:, PInd(k))
def icp_known_corresp(Line1, Line2, QInd, PInd):
    Q = Line1[:, QInd]
    P = Line2[:, PInd]

    MuQ = compute_mean(Q)
    MuP = compute_mean(P)

    W = compute_W(Q, P, MuQ, MuP)

    [R, t] = compute_R_t(W, MuQ, MuP)

    # Compute the new positions of the points after
    # applying found rotation and translation to them
    NewLine = R @ Line2 + t[:, np.newaxis]

    E = compute_error(Q, R @ P + t[:, np.newaxis])

    return NewLine, E


# compute_W: compute matrix W to use in SVD
def compute_W(Q, P, MuQ, MuP):
    Q_centered = Q - MuQ[:, np.newaxis]
    P_centered = P - MuP[:, np.newaxis]

    W = Q_centered @ P_centered.T

    return W


# compute_R_t: compute rotation matrix and translation vector
# based on the SVD as presented in the lecture
def compute_R_t(W, MuQ, MuP):
    # Step 1: Singular Value Decomposition
    U, S, Vt = np.linalg.svd(W)

    # Step 2: Compute rotation matrix
    R = U @ Vt

    # Step 3: Correct reflection if necessary
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # Step 4: Compute translation vector
    t = MuQ - R @ MuP

    return R, t

# compute_mean: compute mean value for a [M x N] matrix
def compute_mean(M):
    return np.mean(M, axis=1)


# compute_error: compute the icp error
def compute_error(Q, OptimizedPoints):
    # sum of squared distances between corresponding points
    return np.sum(np.linalg.norm(Q - OptimizedPoints, axis=0) ** 2)


# simply show the two lines
def show_figure(Line1, Line2):
    plt.figure()
    plt.scatter(Line1[0], Line1[1], marker='o', s=2, label='Line 1')
    plt.scatter(Line2[0], Line2[1], s=1, label='Line 2')

    plt.xlim([-8, 8])
    plt.ylim([-8, 8])
    plt.legend()

    plt.show()


# initialize figure
def init_figure():
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()

    line1_fig = plt.scatter([], [], marker='o', s=2, label='Line 1')
    line2_fig = plt.scatter([], [], marker='o', s=1, label='Line 2')
    # plt.title(title)
    plt.xlim([-8, 8])
    plt.ylim([-8, 8])
    plt.legend()

    return fig, line1_fig, line2_fig


# update_figure: show the current state of the lines
def update_figure(fig, line1_fig, line2_fig, Line1, Line2, hold=False):
    line1_fig.set_offsets(Line1.T)
    line2_fig.set_offsets(Line2.T)
    if hold:
        plt.show()
    else:
        fig.canvas.flush_events()
        fig.canvas.draw()
        plt.pause(0.5)

if __name__ == "__main__":
    #------------------ICP with given correspondences----------------#
    Data = np.load('./data/icp_data.npz')
    Line1 = Data['LineGroundTruth']
    Line2 = Data['LineMovedCorresp']

    # Show the initial positions of the lines
    show_figure(Line1, Line2)

    # We assume that the there are 1 to 1 correspondences for this data
    QInd = np.arange(len(Line1[0]))
    PInd = np.arange(len(Line2[0]))

    # Perform icp given the correspondences
    Line2, E = icp_known_corresp(Line1, Line2, QInd, PInd)

    # Show the adjusted positions of the lines
    show_figure(Line1, Line2)

    # print the error
    print('Error value is: ', E)

    #--------------------ICP without given correspondences------------------#
    Data = np.load('./data/icp_data.npz')
    Line1 = Data['LineGroundTruth']
    Line2 = Data['LineMovedNoCorresp']

    MaxIter = 100
    Epsilon = 0.001
    E = np.inf

    # show figure
    show_figure(Line1, Line2)

    for i in range(MaxIter):

        # Step 1: Find correspondences using nearest neighbors
        distances = np.linalg.norm(
            Line1[:, :, np.newaxis] - Line2[:, np.newaxis, :],
            axis=0
        )

        QInd = np.argmin(distances, axis=0)
        PInd = np.arange(Line2.shape[1])

        # Step 2: Apply ICP with known correspondences
        EOld = E

        [Line2, E] = icp_known_corresp(Line1, Line2, QInd, PInd)

        print('Error value on ' + str(i) + ' iteration is: ', E)

        # Step 3: Check convergence
        if abs(EOld - E) < Epsilon:
            print("ICP converged at iteration", i)
            break

    # Show final alignment
    show_figure(Line1, Line2)