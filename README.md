# 2048DQN

This is a course project of the 2048 Game for PKU's graduate students' lecture *Reinforcement Learning* in Autumn 2021.

#### Environment

Python >= 3.7 
PyTorch >= 1.6

Please pay attention that Python versions earlier than 3.7 may not support this project. F-string is widely used in this project for performance. However, earlier versions of Python do not support f-string.

#### Model

We implement three network architectures and two encoding methods. For details, please read `model/Layers.py`

#### Hyperparameters

This project involves many carefully-designed hyperparameters defined in `train_AI.py`.

#### Results
##### DQN Baseline
We use the default hyperparameters in `train_AI.py` for training (20w episodes)

|               | Avg_score  | 4096 rate | 2048 rate | 1024 rate | 512 rate | 256 rate |
|  ----         | :----:     | :----:    | :----:    | :----:    | :----:   | :----:   |
| emd + Conv2   |  14497.81  |   0.00%   |  11.85%   | 71.35%    |  96.40%  |   99.40% |
| emd + CNN     |  13734.06  |   0.00%   |  11.35%   | 71.20%    |  96.80%  |   99.80% |
| emd + CNNpool |  11126.82  |   0.00%   |  3.85%    | 50.00%    |  91.70%  |   99.70% |   
