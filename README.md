# 2048DQN

This is a course project of the 2048 Game for PKU's graduate students' lecture *Reinforcement Learning* in Autumn 2021.

#### Environment

Python >= 3.7 
PyTorch >= 1.6

Please pay attention that Python versions earlier than 3.7 may not support this project. F-string is widely used in this project for performance. However, earlier versions of Python do not support f-string.

#### Model

We implement three network architectures and two encoding methods. For details, please read `model/Layers.py`

#### Hyperparameters

This project involves many carefully-designed hyperparameters defined in `play_human.py`.
