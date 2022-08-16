import abc
import torch

class Policy(torch.nn.Module, abc.ABC):

    def __init__(self, env_spec, name):
        super().__init__()
        self._env_spec = env_spec
        self._name = name

    # @abc.abstractmethod
    # def get_action(self, observation):
    #     pass

    # @abc.abstractmethod
    # def get_actions(self, observations):
    #     pass


    def reset(self):
        pass

    @property
    def observation_space(self):
        return self._env_spec.observation_space

    @property
    def action_space(self):
        return self._env_spec.action_space


    @property
    def name(self):
        """Name of policy.

        Returns:
            str: Name of policy

        """
        return self._name

    
