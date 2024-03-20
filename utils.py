import numpy as np
import torch
from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F
import gym
import os
from collections import deque
import random
import math
import pandas as pd

import flow.config as config
import sys
from gym.spaces import Box
from copy import deepcopy
from flow.utils.registry import make_create_env


#########################################################################
# The following code snippet is adopted from the repository 
# https://github.com/AboudyKreidieh/h-baselines.git and slightly modified.
# The code snippet comprises the 'import_flow_env' function
# and the 'FlowEnv'

def import_flow_env(env_name, render, evaluate):
    
     # Add flow/examples to your path to located the below modules.
    sys.path.append(os.path.join(config.PROJECT_PATH, "examples"))

    # Import relevant information from the exp_config script.
    module_ma = __import__("exp_configs.rl.multiagent", fromlist=[env_name])

    # Import the sub-module containing the specified exp_config
    if hasattr(module_ma, env_name):
        submodule = getattr(module_ma, env_name)
        multiagent = True
    else:
        raise ValueError("Unable to find experiment config.")

    # Collect the flow_params object.
    flow_params = deepcopy(submodule.flow_params)

    # Update the evaluation flag to match what is requested.
    flow_params['env'].evaluate = evaluate

    # Return the environment.
    return FlowEnv(
        flow_params,
        multiagent=multiagent,
        render=render,
    )


class FlowEnv(gym.Env):

    def __init__(self,
                 flow_params,
                 multiagent=True,
                 render=False,
                 version=0):
        
        # Initialize some variables.
        self.multiagent = multiagent


        # Create the wrapped environment.
        create_env, _ = make_create_env(flow_params, version, render)
        self.wrapped_env = create_env()
        self.horizon = self.wrapped_env.env_params.horizon

        # Collect the IDs of individual vehicles
        self.agents = list(self.wrapped_env.reset().keys())

    @property
    def action_space(self):
        """See wrapped environment."""
        return {key: self.wrapped_env.action_space for key in self.agents}

    @property
    def observation_space(self):
        """See wrapped environment."""
        return self.wrapped_env.observation_space

    def step(self, action):
        """See wrapped environment.

        The done term is also modified in case the time horizon is met.
        """
        obs, reward, done, infos = self.wrapped_env.step(action)

        return obs, reward, done, infos

    def reset(self):
        """Reset the environment."""

        obs = self.wrapped_env.reset()

        return obs


    def render(self, mode='human'):
        """Do nothing."""
        pass

    def query_expert(self, obs):
        if hasattr(self.wrapped_env, "query_expert"):
            return self.wrapped_env.query_expert(obs)
        else:
            raise ValueError("Environment does not have a query_expert method")


#########################################################################################

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def set_seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def load_randomizer_states(torch_randomizer_state, cuda_randomizer_state):
    torch.set_rng_state(torch_randomizer_state)
    torch.cuda.set_rng_state(cuda_randomizer_state)
    
def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()
    
def scale_action(orig_min, orig_max, des_min, des_max, action):
    orig_middle = (orig_max + orig_min) / 2
    des_middle = (des_max + des_min) / 2
    orig_range = orig_max - orig_min
    des_range = des_max - des_min
    range_factor = des_range / orig_range
    scaled_orig_middle = range_factor * orig_middle
    shift = des_middle - scaled_orig_middle

    return action * range_factor + shift

def print_accumulated_rewards(rewards):
    reward_sum = 0
    for agent in rewards.keys():
        reward_sum += rewards[agent]
    print("--------------------------------------")
    print("Sum of rewards: " + str(reward_sum))
    print("______________________________________")


def log_eval_data(work_dir, eval_state_data, eval_reward_data, eval_leader_data, agent_ids):
    
    for agent_id in agent_ids:
        
        for scenario in eval_state_data[agent_id].keys():
            file_path = work_dir  + "/" + agent_id + "-" + str(scenario) + "_eval_state_data.csv"
            eval_data_frame = pd.DataFrame.from_dict(eval_state_data[agent_id][scenario])
            eval_data_frame.to_csv(file_path, index=False)

        for scenario in eval_reward_data[agent_id].keys():
            file_path = work_dir  + "/" + agent_id + "-" + str(scenario) + "_eval_reward_data.csv"
            eval_data_frame = pd.DataFrame.from_dict(eval_reward_data[agent_id][scenario])
            eval_data_frame.to_csv(file_path, index=False)

    for scenario in eval_leader_data.keys():
        file_path = work_dir  + "/" + "leader" + "-" + str(scenario) + "_eval_leader_data.csv"
        eval_data_frame = pd.DataFrame.from_dict(eval_leader_data[scenario])
        eval_data_frame.to_csv(file_path, index=False)

