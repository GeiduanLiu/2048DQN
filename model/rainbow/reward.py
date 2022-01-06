import numpy as np
import torch

winding_idx = [
    # (3, 0),
    (3, 1), (3, 2), (3, 3),
    (2, 3), (2, 2), (2, 1), (2, 0),
    (1, 0), (1, 1), (1, 2), (1, 3),
    (0, 3), (0, 2), (0, 1), (0, 0),
]


def get_winding(state):
    t = state[(3, 0)]
    if t == 0:
        return []
    winding = [t]
    for idx in winding_idx:
        t = state[idx]
        if t <= winding[-1] and t != 0:
            winding.append(t)
        else:
            break
    return winding


def custom_reward(reward, old_state, new_state, done):
    if (old_state == new_state).all():
        return -5

    # original reward: np.log2(clear_score + 1) / 10.0
    # the reward of combining two-level-lower blocks twice is lower than that of current level
    # the reward should not be too large (less than V_max)
    # 16: 0.22, 128: 1.93, 1024: 8.00, 4096: 16.59
    reward = float(reward)
    reward = reward ** 4 * 8

    # check winding
    old_w = get_winding(old_state)
    new_w = get_winding(new_state)
    if old_w != new_w:
        reward += len(new_w)

    if done:
        reward -= 10

    return reward
