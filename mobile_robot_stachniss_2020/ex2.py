#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def plot_belief(belief):
    plt.figure()

    ax = plt.subplot(2, 1, 1)
    ax.matshow(belief.reshape(1, belief.shape[0]))
    ax.set_xticks(np.arange(0, belief.shape[0], 1))
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks([])
    ax.title.set_text("Grid")

    ax = plt.subplot(2, 1, 2)
    ax.bar(np.arange(0, belief.shape[0]), belief)
    ax.set_xticks(np.arange(0, belief.shape[0], 1))
    ax.set_ylim([0, 1.05])
    ax.title.set_text("Histogram")


def plot_world_timeline(history, actions, observations, world):
    """Draw the world map once per time step, with the belief on top of it,
    the action on the left and the observation on the right.

    history      : list of beliefs, history[0] = initial belief
    actions      : list of actions ('F' / 'B')
    observations : observations, observations[t] is read at time step t
    world        : world map, 1 = white cell, 0 = black cell
    """
    history = np.array(history)
    n_steps, n_pos = history.shape
    row_h = 1.5                                   # vertical distance between rows

    fig, ax = plt.subplots(figsize=(0.55 * n_pos + 5, 0.8 * n_steps + 1.5))

    # column headers
    top = 1.4
    ax.text(-2.2, top, 'action', ha='center', fontsize=10, weight='bold')
    ax.text(n_pos / 2, top, 'world map + belief', ha='center', fontsize=10, weight='bold')
    ax.text(n_pos + 1.6, top, 'observation', ha='center', fontsize=10, weight='bold')

    best = history.argmax(axis=1)

    for t in range(n_steps):
        y = -t * row_h

        # ----- action column -----
        if t == 0:
            label = 't=0  start'
        else:
            arrow = '\u2192' if actions[t - 1] == 'F' else '\u2190'
            label = 't={}  {} {}'.format(t, actions[t - 1], arrow)
        ax.text(-2.2, y + 0.5, label, ha='center', va='center',
                fontsize=10, family='monospace')

        # ----- world cells with belief -----
        for x in range(n_pos):
            white = world[x] > 0.0
            ax.add_patch(plt.Rectangle((x, y), 1, 1,
                                       facecolor='white' if white else 'black',
                                       edgecolor='0.6', linewidth=0.8))
            p = history[t, x]
            if p > 0.0:
                # belief as an orange bar rising from the bottom of the cell
                ax.add_patch(plt.Rectangle((x + 0.2, y), 0.6, 0.7 * p,
                                           facecolor='tab:orange', edgecolor='none'))
            if p >= 0.05:
                ax.text(x + 0.5, y + 0.85, '{:.2f}'.format(p), ha='center', va='center',
                        fontsize=6, color='black' if white else 'white')

        # most likely cell
        ax.add_patch(plt.Rectangle((best[t], y), 1, 1, fill=False,
                                   edgecolor='tab:red', linewidth=2))

        # ----- observation column -----
        seen_white = observations[t] > 0.0
        ax.add_patch(plt.Rectangle((n_pos + 0.6, y + 0.15), 0.7, 0.7,
                                   facecolor='white' if seen_white else 'black',
                                   edgecolor='black'))
        ax.text(n_pos + 1.5, y + 0.5, 'saw {}'.format('w' if seen_white else 'b'),
                va='center', fontsize=10)

    # cell indices under the last row
    y_last = -(n_steps - 1) * row_h
    for x in range(n_pos):
        ax.text(x + 0.5, y_last - 0.35, str(x), ha='center', va='center', fontsize=8)
    ax.text(n_pos / 2, y_last - 0.85, 'cell', ha='center', fontsize=9)

    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor='tab:orange'),
               plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='tab:red', linewidth=2)]
    ax.legend(handles, ['belief (bar height = probability)', 'most likely cell'],
              loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)

    ax.set_xlim(-4.2, n_pos + 3.6)
    ax.set_ylim(y_last - 1.1, top + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Bayes filter over time: action, world map + belief, observation')
    fig.tight_layout()


def motion_model(action, belief):
    if action not in ['F', 'B']:
        raise ValueError("Invalid action, only support [F,B]")
    n_pos = len(belief)
    prob_fb = {'F': [0.7, 0.2, 0.1], 'B': [0.1, 0.2, 0.7]}

    new_belief = np.zeros_like(belief)

    for i in range(n_pos):
        prob_prev_pos = belief[i - 1] if i > 0 else 0
        prob_next_pos = belief[i + 1] if i < n_pos - 1 else 0

        probs = np.array([prob_prev_pos, belief[i], prob_next_pos])
        new_belief[i] = np.dot(probs, prob_fb[action])

    return new_belief


def sensor_model(observation, belief, world):
    observation_over_world_prob = {'w_w': 0.7, 'b_b': 0.9, 'b_w': 0.3, 'w_b': 0.1}

    new_belief = np.zeros_like(belief)
    for i, world_data in enumerate(world):
        ob_key = 'w' if observation > 0.0 else 'b'
        world_key = 'w' if observation > 0.0 else 'b'
        ob_world_key = f"{ob_key}_{world_key}"
        #print(f"observation world key: {ob_world_key}")
        new_belief[i] = belief[i] * observation_over_world_prob[ob_world_key]

    # normalize and then return
    return new_belief / np.sum(new_belief)


def recursive_bayes_filter(actions, observations, belief, world):
    if len(belief) != len(world):
        raise ValueError(f"belief and world must have same length, but they are in turn {len(belief)}, {len(world)}")

    for i, action in enumerate(actions):
        # motion model
        belief = motion_model(action, belief)

        # observation model
        sensor_model(observations[i + 1], belief, world)

    return belief

if __name__ == "__main__":
    belief = np.zeros(15)
    x_start = 7
    belief[x_start] = 1.0

    actions = ['F', 'F', 'F', 'F', 'B', 'F', 'F', 'B']
    observations = np.loadtxt("./data/observations.data", delimiter=',', dtype='float')
    world = np.loadtxt("./data/world.data", delimiter=',', dtype='float')

    # belief after 0, 1, ..., len(actions) steps (for the timeline plot)
    history = [recursive_bayes_filter(actions[:k], observations, belief, world)
               for k in range(len(actions) + 1)]
    belief = history[-1]

    plot_world_timeline(history, actions, observations, world)
    plot_belief(belief)
    plt.show()