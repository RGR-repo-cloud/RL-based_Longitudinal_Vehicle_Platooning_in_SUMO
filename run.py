#!/usr/bin/env python3
import datetime
from pathlib import Path
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

from logger import Logger
from replay_buffer import ReplayBuffer
import utils
import hydra
from agent_system import IndividualMultiAgent, SharedMultiAgent


class Workspace(object):
    def __init__(self, cfg):

        self.cfg = cfg

        #change working directory to location of checkpoint or create new one
        if self.cfg.load_checkpoint:
            os.chdir(os.path.join(os.getcwd(), self.cfg.run_directory))
        else:
            dir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d'), datetime.datetime.now().strftime('%H-%M'))
            Path(dir).mkdir(parents=True, exist_ok=True)
            os.chdir(dir)
            
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        utils.set_seed_everywhere(self.cfg.seed)
        self.device = torch.device(self.cfg.device)
        
        #register environment
        self.env = utils.import_flow_env(env_name=self.cfg.env, render=self.cfg.render, evaluate=False)
        self.agent_ids = self.env.agents

        #initialize loggers
        self.loggers = {}
        for agent in self.agent_ids:
            self.loggers[agent] = Logger(   self.work_dir,
                                            agent_id=agent,
                                            save_tb=self.cfg.log_save_tb,
                                            log_frequency=self.cfg.log_frequency,
                                            agent=self.cfg.agent.name,
                                            file_exists=self.cfg.load_checkpoint)
        
        #initialize agents
        if self.cfg.multi_agent_mode == 'individual':
            
            #initialize input and output parameters
            obs_spaces, act_spaces, act_ranges = {}, {}, {}
            for agent in self.agent_ids:
                obs_spaces[agent] = self.env.observation_space[agent].shape
                act_spaces[agent] = self.env.action_space[agent].shape
                act_ranges[agent] = [   float(self.env.action_space[agent].low.min()),
                                        float(self.env.action_space[agent].high.max())]
                
            self.multi_agent = IndividualMultiAgent(self.cfg, self.agent_ids, obs_spaces, act_spaces, act_ranges, int(self.cfg.replay_buffer_capacity), self.device, self.cfg.mode, self.cfg.agent)

            if self.cfg.equalize_agents and not self.cfg.load_checkpoint:
                self.multi_agent.equalize_agents()
        
        elif self.cfg.multi_agent_mode == 'shared':
            
            #initialize input and output parameters of the first agent_id (all agents should be the same)
            obs_space = self.env.observation_space[self.agent_ids[0]].shape
            act_space = self.env.action_space[self.agent_ids[0]].shape
            act_range = [   float(self.env.action_space[self.agent_ids[0]].low.min()),
                            float(self.env.action_space[self.agent_ids[0]].high.max())]
                
            self.multi_agent = SharedMultiAgent(self.cfg, self.agent_ids, obs_space, act_space, act_range, int(self.cfg.replay_buffer_capacity)*len(self.agent_ids), self.device, self.cfg.mode, self.cfg.agent)
  
        else:
            raise Exception('no valid multiagent_mode')

        self.step = 0
        self.episode = 0
        self.min_step_num = 0
        
        #load checkpoint
        if self.cfg.load_checkpoint:
            self.step, self.episode, self.min_step_num = self.multi_agent.load_checkpoint(os.path.join(os.getcwd(), 'checkpoints'), self.cfg.checkpoint, self.cfg.device, int(self.cfg.replay_buffer_capacity))
        


    def evaluate(self):
        average_episode_rewards = {}
        for agent in self.agent_ids:
            average_episode_rewards[agent] = 0
        for episode in range(self.cfg.num_eval_episodes):
            episode_step = 0

            self.env.wrapped_env.set_mode('eval')
            self.env.wrapped_env.set_scenario(episode % self.env.wrapped_env.env_params.additional_params['num_scenarios'])
            obs = self.env.reset()
            self.multi_agent.reset()

            done = False
            episode_rewards = {}
            for agent in self.agent_ids:
                    episode_rewards[agent] = 0

            while not done:
                actions = self.multi_agent.act(obs, sample=False, mode="eval")
                for agent in self.agent_ids:
                    actions[agent] = utils.scale_action(-1, 1,
                                                        float(self.env.action_space[self.agent_ids[0]].low.min()),
                                                        float(self.env.action_space[self.agent_ids[0]].high.max()), 
                                                        actions[agent])
                obs, rewards, dones, _ = self.env.step(actions)
                done = dones['__all__']
                for agent in self.agent_ids:
                    episode_rewards[agent] += rewards[agent]
                episode_step += 1

            for agent in self.agent_ids:
                average_episode_rewards[agent] += episode_rewards[agent]
        
        for agent in self.agent_ids:
            average_episode_rewards[agent] /= self.cfg.num_eval_episodes
            self.loggers[agent].log('eval/episode_reward', average_episode_rewards[agent],
                        self.step)
            self.loggers[agent].dump(self.step)

        utils.print_accumulated_rewards(average_episode_rewards)


    def train(self):
        episode_rewards, done,  = {}, False
        episode_step, eval_count, checkpoint_count = 0, 0, 0
        virtual_session_step = self.step - self.min_step_num # for consistency reasons
        for agent in self.agent_ids:
            episode_rewards[agent] = 0

        self.env.wrapped_env.set_mode('train')
        obs = self.env.reset()
        self.multi_agent.reset()

        training_done = False # ensure that the last episode does not get interrupted

        while not training_done:
            start_time = time.time()

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                actions = {}
                for agent in self.agent_ids:
                    actions[agent] = np.random.uniform(low=float(self.env.action_space[self.agent_ids[0]].low.min()),
                                                       high=float(self.env.action_space[self.agent_ids[0]].high.max()))
            else:
                actions = self.multi_agent.act(obs, sample=True, mode="eval")
                #scale actions to the action ranges of the environmnent
                for agent in self.agent_ids:
                    actions[agent] = utils.scale_action(-1, 1,
                                                        float(self.env.action_space[self.agent_ids[0]].low.min()),
                                                        float(self.env.action_space[self.agent_ids[0]].high.max()), 
                                                        actions[agent])

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                if not (self.cfg.fed_enabled and self.step % self.cfg.fed_frequency == 0) or self.cfg.multi_agent_mode == 'shared':
                    self.multi_agent.update(self.loggers, self.step)
                else:
                    if self.cfg.fed_and_update:
                        self.multi_agent.update(self.loggers, self.step)
                    self.multi_agent.federate(self.cfg.fed_actor, self.cfg.fed_critic, self.cfg.fed_target, self.cfg.fed_alpha, self.cfg.fed_pre_weight, self.cfg.fed_post_weight, self.cfg.fed_first_post_weight, self.cfg.fed_last_pre_weight)

            # advance one step in the environment
            next_obs, rewards, dones, _ = self.env.step(actions)
            
            # allow infinite bootstrap
            done = float(dones['__all__'])
            dones_no_max = {}
            for agent in self.agent_ids:
                dones_no_max[agent] = 0 if episode_step + 1 == self.env.horizon else dones[agent]
                episode_rewards[agent] += rewards[agent]

            # scale actions back to the range of the actor
            for agent in self.agent_ids:
                actions[agent] = utils.scale_action(float(self.env.action_space[self.agent_ids[0]].low.min()),
                                                    float(self.env.action_space[self.agent_ids[0]].high.max()),
                                                    -1, 1, actions[agent])
            self.multi_agent.add_to_buffer(obs, actions, rewards, next_obs, done, dones_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            virtual_session_step += 1


            if done:
                
                for agent in self.agent_ids:
                    self.loggers[agent].log('train/duration',
                                        time.time() - start_time, self.step)
                    self.loggers[agent].log('train/episode', self.episode, self.step)
                    self.loggers[agent].log('train/episode_reward', episode_rewards[agent],
                                           self.step)
                    self.loggers[agent].dump(
                                        self.step, save=(self.step > self.cfg.num_seed_steps))
                utils.print_accumulated_rewards(episode_rewards)
                    
                # evaluate agent periodically
                if int(virtual_session_step / self.cfg.eval_frequency) > eval_count:
                    for agent in self.agent_ids:
                        self.loggers[agent].log('eval/episode', self.episode, self.step)
                    self.evaluate()
                    eval_count += 1
                
                self.env.wrapped_env.set_mode('train')
                obs = self.env.reset()
                self.multi_agent.reset()

                done = False
                for agent in self.agent_ids:
                    episode_rewards[agent] = 0
                episode_step = 0
                self.episode += 1
                training_done = self.step >= self.cfg.num_train_steps
                
                #save models and optimizers
                if self.cfg.save_checkpoint and int(virtual_session_step / self.cfg.checkpoint_frequency) > checkpoint_count:
                    self.min_step_num += self.cfg.checkpoint_frequency
                    self.multi_agent.save_checkpoint(os.path.join(os.getcwd(), 'checkpoints'), self.step, self.episode, self.min_step_num)
                    checkpoint_count += 1
            

            
@hydra.main(config_path='config/run.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    
    if cfg.mode == 'train':
        start = time.time()
        workspace.train()
        end = time.time()
        print('TOTAL_TIME:')
        print(end-start)
    elif cfg.mode == 'eval':
        workspace.evaluate()
    else:
        raise Exception('no valid running mode')


if __name__ == '__main__':
    main()
