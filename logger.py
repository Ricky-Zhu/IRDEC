import numpy as np

class PathLogger:
    def __init__(self,episode_length):
        self.logger = []
        self.episode_length = episode_length
        self.reset()

    def reset(self):
        self.current_episode = []

    def log(self,state):
        self.current_episode.append(state)
        if len(self.current_episode)==self.episode_length:
            self.logger.append(self.current_episode)
            self.reset()

    def clear(self):
        self.logger = []

    def dump(self):
        return np.asarray(self.logger)


