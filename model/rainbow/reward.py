import numpy as np
import torch

winding_idx = [
    (0, 0),
    (0, 1), (0, 2), (0, 3),
    (1, 3), (1, 2), (1, 1), (1, 0),
    (2, 0), (2, 1), (2, 2), (2, 3),
    (3, 3), (3, 2), (3, 1), (3, 0),
]


def get_winding(state: np.ndarray):
    t = state[winding_idx[0]]
    if t == 0:
        return []
    winding = [t]
    for idx in winding_idx[1:]:
        t = state[idx]
        if t <= winding[-1] and t != 0:
            winding.append(t)
        else:
            break
    return winding


def get_winding_len(state: np.ndarray):
    t = state[winding_idx[0]]
    if t == 0:
        return 0
    cnt = 1
    for idx in winding_idx[1:]:
        cur = state[idx]
        if cur <= t and cur != 0:
            cnt += 1
            t = cur
        else:
            break
    return cnt


def rotate_state(state: np.ndarray):
    # let (0, 0) be the largest corner state
    smax = np.max(state[(0, 0, 3, 3), (0, 3, 0, 3)])
    if state[(0, 0)] == smax:
        pass
    elif state[(0, 3)] == smax:
        state = state[:, ::-1]
    elif state[(3, 0)] == smax:
        state = state[::-1, :]
    elif state[(3, 3)] == smax:
        state = state[::-1, ::-1]

    assert state[(0, 0)] == smax

    # let (0, 1) >= (1, 0)
    if state[(0, 1)] < state[(1, 0)]:
        state = state.transpose((1, 0))

    return state


def custom_reward(reward, old_state: np.ndarray, new_state: np.ndarray, done):
    if (old_state == new_state).all():
        return -5

    # original reward: np.log2(clear_score + 1) / 10.0
    # the reward of combining two-level-lower blocks twice is lower than that of current level
    # the reward should not be too large (V_max should be large enough)
    # 16: 0.22, 128: 1.93, 1024: 8.00, 4096: 16.59
    reward = float(reward)
    reward = reward ** 4 * 8

    # check winding
    old_state = rotate_state(old_state)
    new_state = rotate_state(new_state)
    old_w = get_winding(old_state)
    new_w = get_winding(new_state)
    if new_w > old_w:
        reward += 2 * len(new_w)
    elif new_state[(0, 0)] < old_state[(0, 0)]:
        reward -= 2 * len(old_w)
    else:
        reward += len(new_w) - 0.5 * len(old_w)

    if done:
        reward -= 10

    return reward
