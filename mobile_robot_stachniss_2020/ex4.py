#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def normalize_angle(angle):
    """
    make sure angle is between [-pi, pi]
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def normalize_prob(a, variance):
    """
    zero-mean Gaussian density N(a, 0, variance)
    """
    variance = np.maximum(variance, 1e-12)
    return np.exp(-0.5 * a**2 / variance) / np.sqrt(2 * np.pi * variance)

def inverse_motion_model(u_motion):
    """
    rot1: the rot bot first need to rotate from p_prev.theta to move to p_curr
    trans: move from p_rev to p_curr
    rot2: then rotate to p_curr.theta heading
    """
    (p_prev_x, p_prev_y, p_prev_theta), (p_curr_x, p_curr_y, p_curr_theta) = u_motion
    dx = p_curr_x - p_prev_x
    dy = p_curr_y - p_prev_y

    rot1 = normalize_angle(np.arctan2(dy, dx) - p_prev_theta)
    rot2 = normalize_angle(p_curr_theta - p_prev_theta - rot1)
    return rot1, np.sqrt(dx**2 + dy**2), rot2

NOISE_PARAMS = [0.1, 0.1, 0.01, 0.01]
# rotation noise caused by rotation
# rotation noise caused by translation
# translation noise caused by translation
# translation noise caused by rotation
def motion_model_odometry(p_prev, u_odometry, p_query, ignore_rot2 = False):
    rot1, trans, rot2 = inverse_motion_model(u_odometry)
    rot1_q, trans_q, rot2_q = inverse_motion_model([p_prev, p_query])

    p1 = normalize_prob(normalize_angle(rot1 - rot1_q), NOISE_PARAMS[0]*rot1_q**2 + NOISE_PARAMS[1]*trans**2)
    p2 = normalize_prob(trans - trans_q, NOISE_PARAMS[2]*trans_q**2 + NOISE_PARAMS[3]*(rot1_q**2 + rot2_q**2 ))
    p3 = normalize_prob(normalize_angle(rot2 - rot2_q), NOISE_PARAMS[0]*rot2_q**2 + NOISE_PARAMS[1]*trans_q**2)

    return p1*p2 if ignore_rot2 else p1*p2*p3

def sample_motion_model_odometry(p_prev, p_odo, rng):
    rot1, trans, rot2 = inverse_motion_model(p_odo)

    n = p_prev.shape[0]

    r1 = rot1 - rng.normal(0, np.sqrt(NOISE_PARAMS[0] * rot1**2 + NOISE_PARAMS[1]*trans**2), n)
    t = trans - rng.normal(0, np.sqrt(NOISE_PARAMS[2] * trans ** 2 + NOISE_PARAMS[3] * (rot1 ** 2 + rot2 ** 2)))
    r2 = rot2 - rng.normal(0, NOISE_PARAMS[0] * rot2 ** 2 + NOISE_PARAMS[1] * trans ** 2, n)

    x, y, theta = p_prev.T

    return np.stack([x + t*np.cos(theta + r1),
                     y + t*np.sin(theta + r1),
                     normalize_angle(theta + r1 + r2)], axis = 1)

def run(x_start, odometry, n, rng):
    """Propagate n samples through every odometry step.
    Returns a list of (n, 3) arrays, one per time step (including t = 0)."""
    particles = np.tile(x_start, (n, 1))
    history = [particles]
    for t in range(1, len(odometry)):
        u_odometry = [odometry[t - 1], odometry[t]]
        particles = sample_motion_model_odometry(particles, u_odometry, rng)
        history.append(particles)
    return history

def plot_samples(history, ideal, x_start, filename="sample_odometry_evaluation.png"):
    """Plot the sample positions of every step in one figure.

    history: list of (n, 3) sample arrays, one per time step (index 0 = start)
    ideal:   (T, 3) noise-free path
    """
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    fig, ax = plt.subplots(figsize=(7, 7.5), constrained_layout=True)

    for t in range(1, len(history)):
        pts = history[t]
        ax.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.45, color=colors[t - 1],
                   linewidths=0, label=f"$x_{t}$ samples")

    ax.plot(ideal[:, 0], ideal[:, 1], "-o", color="#333333", lw=1.5, ms=5,
            label="noise-free odometry path", zorder=3)
    ax.plot(*x_start[:2], "s", color="black", ms=8, label="start $x_0$", zorder=4)
    for t, p in enumerate(ideal):
        ax.annotate(f"$x_{t}$", p[:2], xytext=(-14, 6),
                    textcoords="offset points", fontsize=11, color="#222222")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Sample odometry motion model: {len(history[1])} samples per step\n"
                 r"$\alpha = [0.1, 0.1, 0.01, 0.01]$")
    ax.set_aspect("equal")
    ax.grid(True, color="#e5e5e5", lw=0.6)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4,
                    fontsize=9, frameon=False)
    for handle in leg.legend_handles[:len(history) - 1]:   # sample dots only
        handle.set_sizes([30])
        handle.set_alpha(1.0)

    fig.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    return fig

def plot_samples_step_by_step(history, ideal, x_start, odometry=None,
                              frames_dir=None, pause=1.0):
    """Show the samples step by step in a live window (like the grid mapping exercise).

    history:    list of (n, 3) sample arrays, one per time step (index 0 = start)
    ideal:      (T, 3) noise-free path
    odometry:   (T, 3) odometry readings, used to show rot1, trans, rot2 per step
    frames_dir: if given, every step is also saved as an image
    pause:      seconds to show each step
    """
    import os
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    n_steps = len(history) - 1

    # fixed axis limits for all steps, so the view does not jump
    all_pts = np.vstack([h[:, :2] for h in history[1:]] + [ideal[:, :2]])
    lo, hi = all_pts.min(0), all_pts.max(0)
    margin = 0.1 * max(hi - lo)
    arrow_len = 0.08 * max(hi - lo)

    plt.ion()  # interactive: window updates live
    fig, ax = plt.subplots(figsize=(7, 7.5), constrained_layout=True)
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)

    for t in range(1, n_steps + 1):
        ax.clear()

        # samples of earlier steps, faded
        for k in range(1, t):
            pts = history[k]
            ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.12, color=colors[(k - 1) % len(colors)],
                       linewidths=0, label=f"$x_{k}$ samples (earlier)")

        # samples of the current step, highlighted
        pts = history[t]
        ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.6, color=colors[(t - 1) % len(colors)],
                   linewidths=0, label=f"$x_{t}$ samples (current)")
        mean = pts[:, :2].mean(0)
        ax.plot(*mean, "x", color="black", ms=10, mew=2, label="sample mean", zorder=5)

        # noise-free path up to this step
        ax.plot(ideal[:t + 1, 0], ideal[:t + 1, 1], "-o", color="#333333", lw=1.5, ms=5,
                label="noise-free odometry path", zorder=3)
        ax.plot(*x_start[:2], "s", color="black", ms=8, label="start $x_0$", zorder=4)
        for k, p in enumerate(ideal[:t + 1]):
            ax.annotate(f"$x_{k}$", p[:2], xytext=(-14, 6),
                        textcoords="offset points", fontsize=11, color="#222222")

        # heading of the noise-free pose at this step
        px, py, pth = ideal[t]
        ax.arrow(px, py, arrow_len * np.cos(pth), arrow_len * np.sin(pth),
                 head_width=arrow_len * 0.35, color="tab:orange", zorder=6)

        # title: step, odometry motion and sample spread
        title = f"step {t}/{n_steps}   ({len(pts)} samples)"
        if odometry is not None:
            rot1, trans, rot2 = inverse_motion_model(odometry[t - 1:t + 1])
            title += (f"\nrot1={np.degrees(rot1):.0f} deg, trans={trans:.2f} m, "
                      f"rot2={np.degrees(rot2):.0f} deg")
        std = pts[:, :2].std(0)
        title += f"\nsample std: x={std[0]:.3f} m, y={std[1]:.3f} m"
        ax.set_title(title)

        ax.set_xlim(lo[0] - margin, hi[0] + margin)
        ax.set_ylim(lo[1] - margin, hi[1] + margin)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        ax.grid(True, color="#e5e5e5", lw=0.6)
        leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3,
                        fontsize=9, frameon=False)
        for handle in leg.legend_handles[:t]:   # sample dots only
            handle.set_sizes([30])
            handle.set_alpha(1.0)

        plt.pause(pause)  # draw and show this step
        if frames_dir:
            fig.savefig(os.path.join(frames_dir, f"step_{t:02d}.png"), dpi=150)

    plt.ioff()
    plt.show()  # keep the windows open
    return fig

if __name__ == '__main__':
    rng = np.random.RandomState(42)
    x0 = np.array([2.0, 3.0, 0.0])
    u_odometry = np.array([[0.0, 0.0, 0.0],
                           [0.5, 0.0, np.pi / 2],
                           [0.5, 0.5, 0.0],
                           [1.0, 0.5, 0.0],
                           [1.0, 1.5, np.pi / 2],
                           [1.0, 2.5, np.pi / 2]])
    n_samples = 1000

    # noise-free path: apply each odometry motion (rot1, trans, rot2) to x0
    ideal = [x0]
    for t in range(1, len(u_odometry)):
        rot1, trans, rot2 = inverse_motion_model(u_odometry[t - 1:t + 1])
        print(f"Odometry motion {t}: rot1={rot1:.4f}, trans={trans:.4f}, rot2={rot2:.4f}")
        x, y, theta = ideal[-1]
        ideal.append(np.array([x + trans * np.cos(theta + rot1),
                               y + trans * np.sin(theta + rot1),
                               normalize_angle(theta + rot1 + rot2)]))
    ideal = np.array(ideal)

    history = run(x0, u_odometry, n_samples, rng)
    print("\nSample spread per step:")
    for t in range(1, len(history)):
        m, s = history[t][:, :2].mean(0), history[t][:, :2].std(0)
        print(f"  t={t}: mean=({m[0]:.2f}, {m[1]:.2f}), std=({s[0]:.3f}, {s[1]:.3f})")

    # ---------------- Plotting ----------------
    plot_samples(history, ideal, x0)
    plot_samples_step_by_step(history, ideal, x0, u_odometry)