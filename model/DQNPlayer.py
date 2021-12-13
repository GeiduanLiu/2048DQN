import abc


class DQNPlayer:
    def __init__(self):
        pass

    @abc.abstractmethod
    def choose_action(self, states, *args, **kwargs):
        pass

    @abc.abstractmethod
    def store_memory(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def update_episode(self, episode, *args, **kwargs):
        pass

    @abc.abstractmethod
    def learn(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def  target_replace_op(self, *args, **kwargs):
        """
        By default: DDQN

        :param args: args
        :param kwargs: kwargs
        :return:
        """
        pass

    @abc.abstractmethod
    def save_model(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def load_model(self, *args, **kwargs):
        pass
