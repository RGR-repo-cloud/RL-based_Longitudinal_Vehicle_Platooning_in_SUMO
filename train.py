#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils
import hydra
from agent_system import MultiAgent


def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=True)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        
        #register environment
        self.env = utils.import_flow_env("multi_lane_highway", False, False) #quick fix
        self.env.seed(cfg.seed)
        self.agent_ids = self.env.agents

        #initialize loggers
        self.loggers = {}
        for agent in self.agent_ids:
            self.loggers[agent] = Logger(   self.work_dir,
                                            agent_id=agent,
                                            save_tb=cfg.log_save_tb,
                                            log_frequency=cfg.log_frequency,
                                            agent=cfg.agent.name)
        
        #initialize input and output parameters
        obs_spaces, act_spaces, act_ranges = {}, {}, {}
        for agent in self.agent_ids:
            obs_spaces[agent] = self.env.observation_space[agent].shape
            act_spaces[agent] = self.env.action_space[agent].shape
            act_ranges[agent] = [  float(self.env.action_space[agent].low.min()),
                                    float(self.env.action_space[agent].high.max())]

        #initialize agents
        self.agents = MultiAgent(cfg, self.agent_ids, obs_spaces, act_spaces, act_ranges, int(cfg.replay_buffer_capacity), self.device)
            
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0


    def evaluate(self):
        average_episode_rewards = {}
        for agent in self.agent_ids:
            average_episode_rewards[agent] = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agents.reset_all()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_rewards = {}
            for agent in self.agent_ids:
                    episode_rewards[agent] = 0
            while not done:
                actions = self.agents.act_all(obs, sample=False, mode="eval")
                obs, rewards, done, _ = self.env.step(actions)
                self.video_recorder.record(self.env)
                for agent in self.agent_ids:
                    episode_rewards[agent] += rewards[agent]

            for agent in self.agent_ids:
                average_episode_rewards[agent] += episode_rewards[agent]
            self.video_recorder.save(f'{self.step}.mp4')
        for agent in self.agent_ids:
            average_episode_rewards[agent] /= self.cfg.num_eval_episodes
            self.loggers[agent].log('eval/episode_reward', average_episode_rewards[agent],
                        self.step)
            self.loggers[agent].dump(self.step)

    def run(self):
        episode, episode_rewards, done = 0, {}, True
        for agent in self.agent_ids:
            episode_rewards[agent] = 0
        start_time = time.time()
        
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    for agent in self.agent_ids:
                        self.loggers[agent].log('train/duration',
                                            time.time() - start_time, self.step)
                        start_time = time.time()
                        self.loggers[agent].dump(
                                            self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    for agent in self.agent_ids:
                        self.loggers[agent].log('eval/episode', episode, self.step)
                    self.evaluate()

                for agent in self.agent_ids:
                    self.loggers[agent].log('train/episode_reward', episode_rewards[agent],
                                           self.step) 

                obs = self.env.reset()
                self.agents.reset_all()
                done = False
                for agent in self.agent_ids:
                    episode_rewards[agent] = 0
                episode_step = 0
                episode += 1

                for agent in self.agent_ids:
                    self.loggers[agent].log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                actions = {}
                for agent in self.agent_ids:
                    actions[agent] = self.env.action_space[agent].sample()
            else:
                actions = self.agents.act_all(obs, sample=False, mode="eval")

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agents.update_all(self.loggers, self.step)

            next_obs, rewards, done, _ = self.env.step(actions)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env.horizon else done
            for agent in self.agent_ids:
                episode_rewards[agent] += rewards[agent]
            self.agents.add_to_buffers(obs, actions, rewards, next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
