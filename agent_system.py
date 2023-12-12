import os
from pathlib import Path
import torch
from replay_buffer import ReplayBuffer
import hydra
from torch.nn import ModuleDict
import utils
import numpy as np


class MultiAgent:

    def __init__(self, cfg, agent_ids, obs_spaces, act_spaces, act_ranges, replay_buffer_cap, device):

        self.agent_ids = agent_ids
        self.device = device

        #instantiate agents and buffers
        self.replay_buffers = {}
        agents = {}
        for agent in self.agent_ids:

            #quick fix
            cfg.agent.params.obs_dim = obs_spaces[agent][0]
            cfg.agent.params.action_dim = act_spaces[agent][0]
            cfg.agent.params.action_range = act_ranges[agent]
        
            agents[agent] = hydra.utils.instantiate(cfg.agent)
            self.replay_buffers[agent] = ReplayBuffer(  obs_spaces[agent],
                                                        act_spaces[agent],
                                                        replay_buffer_cap,
                                                        self.device)
        
        self.agents = ModuleDict(agents)


    def reset_all(self):

        for agent in self.agent_ids:
            self.agents[agent].reset()


    def act_all(self, obs, sample=False, mode=None):
        
        actions = {}

        if mode == "eval":
            for agent in self.agent_ids:
                with utils.eval_mode(self.agents[agent]):
                    actions[agent] = self.agents[agent].act(obs[agent], sample)
            return actions
        
        if mode == "train":
            for agent in self.agent_ids:
                with utils.train_mode(self.agents[agent]):
                    actions[agent] = self.agents[agent].act(obs[agent], sample)
            return actions

        if mode == None:
            for agent in self.agent_ids:
                actions[agent] = self.agents[agent].act(obs[agent], sample)
            return actions

        raise Exception("ERROR: NO VALID ACTING MODE!")


    
    def update_all(self, loggers, step):

        for agent in self.agent_ids:
            self.agents[agent].update(self.replay_buffers[agent], loggers[agent], step)


    def add_to_buffers(self, obs, actions, rewards, next_obs, done, done_no_max):

        for agent in self.agent_ids:
            self.replay_buffers[agent].add(obs[agent], actions[agent], rewards[agent], next_obs[agent], done, done_no_max)


    def load_checkpoint(self, dir):

        #load model parameters
        model_checkpoint = torch.load(os.path.join(dir, 'checkpoint.pt'))
        self.agents.load_state_dict(model_checkpoint['models'])
        step = model_checkpoint['step']
      
        for agent in self.agent_ids:
            self.agents[agent].critic_optimizer.load_state_dict(model_checkpoint['optims'][agent]['critic'])
            self.agents[agent].actor_optimizer.load_state_dict(model_checkpoint['optims'][agent]['actor'])
            self.agents[agent].log_alpha_optimizer.load_state_dict(model_checkpoint['optims'][agent]['alpha'])
            self.agents[agent].log_alpha = model_checkpoint['log_alpha'][agent]

        #load replay buffer entries
        rep_dir = os.path.join(dir, 'replay_buffers')

        for agent in self.agent_ids:
            data = np.load(os.path.join(rep_dir, agent + '.npz'))
            
            for i in range(step):

                obs = data['obses'][i]
                next_obs = data['next_obses'][i]
                action = data['actions'][i]
                reward = data['rewards'][i]
                not_done = data['not_dones'][i]
                not_done_no_max = data['not_dones_no_max'][i]

                self.replay_buffers[agent].add(obs=obs,
                                               action=action,
                                               reward=reward,
                                               next_obs=next_obs,
                                               done=not not_done,
                                               done_no_max=not not_done_no_max)
            
            data.close()
        

        return step
    

    def save_checkpoint(self, dir, step):
        
        checkpoint_dir = os.path.join(dir, 'cp_{:d}'.format(step))
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            
        #save model parameters
        state = {}
        state['step'] = step
        state['models'] = self.agents.state_dict()
        state['log_alpha'] = {}
        state['optims'] = {}
        
        for agent in self.agent_ids:
            state['optims'][agent] = {}
            state['optims'][agent]['critic'] = self.agents[agent].critic_optimizer.state_dict()
            state['optims'][agent]['actor'] = self.agents[agent].actor_optimizer.state_dict()
            state['optims'][agent]['alpha'] = self.agents[agent].log_alpha_optimizer.state_dict()
            state['log_alpha'][agent] = self.agents[agent].log_alpha

        torch.save(state, os.path.join(checkpoint_dir, 'checkpoint.pt'))

        #save replay_buffer
        rep_dir = os.path.join(checkpoint_dir, 'replay_buffers')
        Path(rep_dir).mkdir(parents=True, exist_ok=True)
        
        for agent in self.agent_ids:

            #save numpy arrays
            np_file = os.path.join(rep_dir, agent + '.npz')
            
            np.savez(np_file,
                     obses = self.replay_buffers[agent].obses,
                     next_obses = self.replay_buffers[agent].next_obses,
                     actions = self.replay_buffers[agent].actions,
                     rewards = self.replay_buffers[agent].rewards,
                     not_dones = self.replay_buffers[agent].not_dones,
                     not_dones_no_max = self.replay_buffers[agent].not_dones_no_max
                     )
            
            """
            #save additional attributes
            attr_file_path = os.path.join(rep_dir, agent + '.txt')
            
            attr_file = open(attr_file_path, 'w')
            print(self.replay_buffers[agent].idx, file=attr_file)
            print(self.replay_buffers[agent].last_save, file=attr_file)
            print(self.replay_buffers[agent].full, file=attr_file)
            attr_file.close()
            """
            

            
            

