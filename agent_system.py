import os
from pathlib import Path
import torch
from replay_buffer import ReplayBuffer
import hydra
from torch.nn import ModuleDict
import utils


class MultiAgent:

    def __init__(self, cfg, agent_ids, obs_spaces, act_spaces, act_ranges, replay_buffer_caps, device):

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
                                                        replay_buffer_caps,
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

        checkpoint = torch.load(dir)
        self.agents.load_state_dict(checkpoint['models'])
        
        #for param_tensor in self.agents.state_dict():
         #   print(param_tensor, "\t", self.agents.state_dict()[param_tensor].size())

        for agent in self.agent_ids:
            self.agents[agent].critic_optimizer.load_state_dict(checkpoint['optims'][agent]['critic'])
            self.agents[agent].actor_optimizer.load_state_dict(checkpoint['optims'][agent]['actor'])
            self.agents[agent].log_alpha_optimizer.load_state_dict(checkpoint['optims'][agent]['alpha'])
            self.agents[agent].log_alpha = checkpoint['log_alpha'][agent]

        return checkpoint['step']
    

    def save_checkpoint(self, dir, step):
        
        Path(dir).mkdir(parents=True, exist_ok=True)
            
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

        torch.save(state, os.path.join(dir, 'checkpoint-{:d}.pt'.format(step)))