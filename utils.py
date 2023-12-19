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

import dmc2gym

import flow.config as config
import sys
from gym.spaces import Box
from copy import deepcopy
from flow.utils.registry import make_create_env

def import_flow_env(env_name, render, evaluate):
    
     # Add flow/examples to your path to located the below modules.
    sys.path.append(os.path.join(config.PROJECT_PATH, "examples"))

    # Import relevant information from the exp_config script.
    module = __import__("exp_configs.rl.singleagent", fromlist=[env_name])
    module_ma = __import__("exp_configs.rl.multiagent", fromlist=[env_name])

    # Import the sub-module containing the specified exp_config and determine
    # whether the environment is single agent or multi-agent.
    if hasattr(module, env_name):
        submodule = getattr(module, env_name)
        multiagent = False
    elif hasattr(module_ma, env_name):
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
                 multiagent=False,
                 render=False,
                 version=0):
        
        # Initialize some variables.
        self.multiagent = multiagent


        # Create the wrapped environment.
        create_env, _ = make_create_env(flow_params, version, render)
        self.wrapped_env = create_env()

        # Collect the IDs of individual vehicles if using a multi-agent env.
        if self.multiagent:
            self.agents = list(self.wrapped_env.reset().keys())

        # for tracking the time horizon
        self.step_number = 0
        self.horizon = self.wrapped_env.env_params.horizon

    @property
    def action_space(self):
        """See wrapped environment."""
        if self.multiagent:
            return {key: self.wrapped_env.action_space for key in self.agents}
        else:
            return self.wrapped_env.action_space

    @property
    def observation_space(self):
        """See wrapped environment."""
        if self.multiagent:
            return {
                key: self.wrapped_env.observation_space for key in self.agents}
        else:
            return self.wrapped_env.observation_space

    def step(self, action):
        """See wrapped environment.

        The done term is also modified in case the time horizon is met.
        """
        obs, reward, done, info_dict = self.wrapped_env.step(action)

        # Check if the time horizon has been met.
        self.step_number += 1
        if isinstance(done, dict):
            done = {key: done[key] or self.step_number == self.horizon
                    for key in obs.keys()}
            done["__all__"] = all(done.values())
        else:
            done = done or self.step_number == self.horizon

        return obs, reward, done["__all__"], info_dict  ########################quick fix

    def reset(self):
        """Reset the environment."""
        self.step_number = 0

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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


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
