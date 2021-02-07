import gym
import numpy as np
import itertools

from utils import UnityGymException
from mlagents_envs import logging_util
from mlagents_envs.base_env import ActionTuple, BaseEnv
from mlagents_envs.base_env import DecisionSteps, TerminalSteps
from typing import Any, Dict, List, Tuple, Union

logger = logging_util.get_logger(__name__)
logging_util.set_log_level(logging_util.INFO)


class UnityToGymWrapperMaSS(gym.Env):
    def __init__(
            self,
            unity_env: BaseEnv,
            agent_setting: dict = {"agent_0": 1, "agent_1": 1},
            uint8_visual: bool = False,
            flatten_branched: bool = False,
            allow_multiple_obs: bool = False,
    ):
        self._env = unity_env

        # Take a single step so that the brain information will be sent over
        if not self._env.behavior_specs:
            self._env.step()

        self.visual_obs = None
        # Save the step result from the last time all Agents requested decisions.
        self._previous_decision_step: DecisionSteps = None
        self._flattener = None
        # Hidden flag used by Atari environments to determine if the game is over
        self.game_over = False
        self._allow_multiple_obs = allow_multiple_obs

        # Check brain configuration
        if len(self._env.behavior_specs) > sum(agent_setting.values()):
            raise UnityGymException(
                "Behavior needs to be less than the amount of agents"
            )

        self.agent_names = list(self._env.behavior_specs.keys())
        self.group_specs = self._env.behavior_specs  # behavior object of agents

        print(self.group_specs[self.agent_names[1]].observation_shapes)

        self._env.reset()

        decision_steps, terminal_steps = zip(*[self._env.get_steps(name) for name in self.agent_names])
        print(decision_steps, terminal_steps)
        self._previous_decision_step = decision_steps

        # Set action spaces
        ss_action_spec = self.group_specs[self.agent_names[0]].action_spec
        if ss_action_spec.is_discrete():
            self.action_size = self.group_specs[self.agent_names[0]].action_spec.discrete_size
            branches = self.group_specs[self.agent_names[0]].action_spec.discrete_branches
            if self.action_size == 1:
                self._action_space = gym.spaces.Discrete(branches[0])
            else:
                if flatten_branched:
                    self._flattener = ActionFlattener(branches)
                    self._action_space = self._flattener.action_space
                else:
                    self._action_space = gym.spaces.MultiDiscrete(branches)
        elif ss_action_spec.is_continuous():
            if flatten_branched:
                logger.warning(
                    "The environment has a non-discrete action space. It will "
                    "not be flattened."
                )

            self.action_size = ss_action_spec.continuous_size
            high = np.array([1] * ss_action_spec.continuous_size)
            self._action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        else:
            raise UnityGymException(
                "The gym wrapper does not provide explicit support for both discrete "
                "and continuous actions."
            )

        # Set observations space
        list_spaces: List[gym.Space] = []
        shapes = self._get_vis_obs_shape()
        for shape in shapes:
            if uint8_visual:
                list_spaces.append(gym.spaces.Box(0, 255, dtype=np.uint8, shape=shape))
            else:
                list_spaces.append(gym.spaces.Box(0, 1, dtype=np.float32, shape=shape))
        if self._get_vec_obs_size() > 0:
            # vector observation is last
            high = np.array([np.inf] * self._get_vec_obs_size())
            list_spaces.append(gym.spaces.Box(-high, high, dtype=np.float32))
        if self._allow_multiple_obs:
            self._observation_space = gym.spaces.Tuple(list_spaces)
        else:
            self._observation_space = list_spaces[0]  # only return the first one

        print(self._observation_space)

    def _get_vis_obs_shape(self) -> List[Tuple]:
        result: List[Tuple] = []
        for name in self.agent_names:
            for shape in self.group_specs[name].observation_shapes:
                if len(shape) == 3:
                    result.append(shape)
        return result

    def _get_vis_obs_list(self, step_result: Union[DecisionSteps, TerminalSteps]) -> List[np.ndarray]:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 4:
                result.append(obs)
        return result

    def _get_vector_obs(self, step_result: Union[DecisionSteps, TerminalSteps]) -> np.ndarray:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 2:
                result.append(obs)
        return np.concatenate(result, axis=1)

    def _get_vec_obs_size(self) -> int:
        result = 0
        for name in self.agent_names:
            for shape in self.group_specs[name].observation_shapes:
                if len(shape) == 1:
                    result += shape[0]
        return result


class ActionFlattener:
    """
    Flattens branched discrete action spaces into single-branch discrete action spaces.
    """

    def __init__(self, branched_action_space):
        """
        Initialize the flattener.
        :param branched_action_space: A List containing the sizes of each branch of the action
        space, e.g. [2,3,3] for three branches with size 2, 3, and 3 respectively.
        """
        self._action_shape = branched_action_space
        self.action_lookup = self._create_lookup(self._action_shape)
        self.action_space = gym.spaces.Discrete(len(self.action_lookup))

    @classmethod
    def _create_lookup(self, branched_action_space):
        """
        Creates a Dict that maps discrete actions (scalars) to branched actions (lists).
        Each key in the Dict maps to one unique set of branched actions, and each value
        contains the List of branched actions.
        """
        possible_vals = [range(_num) for _num in branched_action_space]
        all_actions = [list(_action) for _action in itertools.product(*possible_vals)]
        # Dict should be faster than List for large action spaces
        action_lookup = {
            _scalar: _action for (_scalar, _action) in enumerate(all_actions)
        }
        return action_lookup

    def lookup_action(self, action):
        """
        Convert a scalar discrete action into a unique set of branched actions.
        :param: action: A scalar value representing one of the discrete actions.
        :return: The List containing the branched actions.
        """
        return self.action_lookup[action]


if __name__ == '__main__':
    from mlagents_envs.environment import UnityEnvironment
    from gym_unity.envs import UnityToGymWrapper

    env_id = "/home/miku/PythonObjects/unity-exercise/envs/Socker4/Socker.x86_64"
    # env_id = "/home/miku/PythonObjects/unity-exercise/envs/Basic/Basic"
    env = UnityEnvironment(env_id)

    env = UnityToGymWrapperMaSS(env)
