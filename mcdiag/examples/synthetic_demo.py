"""Runnable synthetic demo: see the full mcdiag report in under a minute, no real data needed.

We build one synthetic recording where the neural activity encodes movement plus a small genuine
choice signal, the movement is multi-dimensional with both high- and low-variance directions, and
choice is partly correlated with movement (the realistic confound). Then we run diagnose.

    python examples/synthetic_demo.py
"""
import numpy as np

import mcdiag


def make_block(n_cells=20, n_trials=70, n_move_features=4, seed=0):
    rng = np.random.default_rng(seed)
    # movement with a clear high-variance direction and weaker low-variance directions
    scales = np.array([3.0, 1.0, 0.5, 0.3])[:n_move_features]
    movement = rng.standard_normal((n_trials, n_move_features)) * scales
    # choice partly driven by movement (the realistic confound), partly independent
    w = rng.standard_normal(n_move_features)
    latent = movement @ w
    latent = latent / latent.std()
    choice_drive = 0.6 * latent + 0.8 * rng.standard_normal(n_trials)
    choice = (choice_drive > np.median(choice_drive)).astype(int)
    # cells encode movement strongly and choice weakly (a real but small decision signal)
    cell_move = rng.standard_normal((n_cells, n_move_features))
    cell_choice = 0.15 * rng.standard_normal(n_cells)
    base = rng.standard_normal((n_cells, n_trials))
    activity = base + cell_move @ movement.T + np.outer(cell_choice, (choice * 2 - 1.0))
    return activity, choice, movement


if __name__ == "__main__":
    # several co-recorded blocks, as with multi-session pseudo-population data
    blocks = [make_block(seed=i) for i in range(3)]
    activity = [b[0] for b in blocks]
    choice = [b[1] for b in blocks]
    movement = [b[2] for b in blocks]
    print(f"synthetic recording: {len(blocks)} blocks, "
          f"{activity[0].shape[0]} cells x {activity[0].shape[1]} trials each\n")

    dx = mcdiag.diagnose(
        activity=activity, choice=choice, movement=movement,
        firing_rate_hz=20.0, n_simultaneous_cells=activity[0].shape[0],
        calibrate_kwargs={"n_repeats": 12},
    )
    print(dx)
