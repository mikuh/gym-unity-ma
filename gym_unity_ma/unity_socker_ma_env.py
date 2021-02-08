from ray.rllib.env.multi_agent_env import MultiAgentEnv
from mlagents_envs.environment import UnityEnvironment
from typing import Any, Dict, List, Tuple, Union
import gym
import itertools
import numpy as np
from mlagents_envs.base_env import ActionTuple, BaseEnv
from mlagents_envs.base_env import DecisionSteps, TerminalSteps

GymStepResult = Tuple[np.ndarray, float, bool, Dict]

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


class MultiAgentSocker(MultiAgentEnv):

    def __init__(self, env_config, flatten_branched: bool = True, uint8_visual: bool = False,
                 allow_multiple_obs: bool = False, ):
        num = env_config.pop("num_agents", 1)

        # self.vector_index = env_config.vector_index
        # self.worker_index = env_config.worker_index
        # self.worker_id = env_config["unity_worker_id"] + env_config.worker_index
        self.worker_id = 10
        env_name = '/home/miku/PythonObjects/unity-exercise/envs/Socker4/Socker.x86_64'
        self._env = UnityEnvironment(env_name, worker_id=self.worker_id, no_graphics=False)

        if not self._env.behavior_specs:
            self._env.step()

        self.agents_name = list(self._env.behavior_specs.keys())

        self.agents = [self._env.behavior_specs[name] for name in self.agents_name]

        self.dones = set()
        self.game_over = False
        self._allow_multiple_obs = allow_multiple_obs
        # self.observation_space = self.agents[0].observation_shapes
        # self.action_space = self.agents[0].action_space

        # print(self.agents[0].observation_shapes)
        # print(self.action_space)

        # set observation space
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

        if allow_multiple_obs:
            self._observation_space = gym.spaces.Tuple(list_spaces)
        else:
            self._observation_space = list_spaces[0]  # Box(-inf, inf, (336,), float32)  264 + 72

        # set action space
        if self.agents[0].action_spec.is_discrete():
            self.action_size = self.agents[0].action_spec.discrete_size  # 3
            print(self.action_size)
            branches = self.agents[0].action_spec.discrete_branches  # (3, 3, 3)
            print(branches)
            if self.agents[0].action_spec.discrete_size == 1:
                self._action_space = gym.spaces.Discrete(branches[0])
            else:
                if flatten_branched:
                    self._flattener = ActionFlattener(branches)
                    self._action_space = self._flattener.action_space  # [3 3 3]
                else:
                    self._action_space = gym.spaces.MultiDiscrete(branches)  # 3* 3* 3

    # def reset(self):
    #     self.dones = set()
    #     return {i: a.reset() for i, a in enumerate(self.agents)}

    def reset(self) -> Union[List[np.ndarray], np.ndarray]:
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object/list): the initial observation of the
        space.
        """
        self._env.reset()
        decision_steps, _ = zip(*[self._env.get_steps(name) for name in self.agents_name])
        self.game_over = False
        # res = self._single_step
        for obs in decision_steps[0].obs:
            print(obs.shape)

        print(self._get_vector_obs(decision_steps[0]).shape)

    def _single_step(self, info: Union[DecisionSteps, TerminalSteps]) -> GymStepResult:
        if self._allow_multiple_obs:
            visual_obs = self._get_vis_obs_list(info)
            visual_obs_list = []
            for obs in visual_obs:
                visual_obs_list.append(self._preprocess_single(obs[0]))
            default_observation = visual_obs_list
            if self._get_vec_obs_size() >= 1:
                default_observation.append(self._get_vector_obs(info)[0, :])
        else:
            if self._get_n_vis_obs() >= 1:
                visual_obs = self._get_vis_obs_list(info)
                default_observation = self._preprocess_single(visual_obs[0][0])
            else:
                default_observation = self._get_vector_obs(info)[0, :]

        if self._get_n_vis_obs() >= 1:
            visual_obs = self._get_vis_obs_list(info)
            self.visual_obs = self._preprocess_single(visual_obs[0][0])

        done = isinstance(info, TerminalSteps)

        return (default_observation, info.reward[0], done, {"step": info})

    def _get_vis_obs_shape(self) -> List[Tuple]:
        """ get the visual observations shape
        Return a list of shape
        """
        result: List[Tuple] = []
        for shape in self.agents[0].observation_shapes:
            if len(shape) == 3:
                result.append(shape)
        return result

    def _get_vis_obs_list(self, step_result: Union[DecisionSteps, TerminalSteps]) -> List[np.ndarray]:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 4:
                result.append(obs)
        return result

    def _get_vec_obs_size(self) -> int:
        """get the vector observations size
        Return the len of vector
        """
        result = 0
        for shape in self.agents[0].observation_shapes:
            if len(shape) == 1:
                result += shape[0]
        return result

    def _get_vector_obs(self, step_result: Union[DecisionSteps, TerminalSteps]) -> np.ndarray:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 2:
                result.append(obs)
        return np.concatenate(result, axis=1)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


if __name__ == '__main__':
    env_config = {
        "num_agents": 2
    }

    socker = MultiAgentSocker(env_config, flatten_branched=True)

    socker.reset()