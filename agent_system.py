import os
from pathlib import Path
import torch
from replay_buffer import ReplayBuffer
import hydra
from torch.nn import ModuleDict
import utils
import numpy as np
from abc import ABC, abstractmethod 


class MultiAgent(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def act(self, obs, sample, mode):
        pass

    @abstractmethod
    def update(self, loggers, step):
        pass

    @abstractmethod
    def add_to_buffer(self, obs, actions, rewards, next_obs, done, done_no_max):
        pass

    @abstractmethod
    def save_checkpoint(self, dir, step):
        pass

    @abstractmethod
    def load_checkpoint(self, dir, checkpoint):
        pass


class IndividualMultiAgent(MultiAgent):

    def __init__(self, cfg, agent_ids, obs_spaces, act_spaces, act_ranges, replay_buffer_cap, device, mode, learning_model, randomizer, control_mode):

        self.agent_ids = agent_ids
        self.device = device
        self.mode = mode
        self.randomizer = randomizer
        self.control_mode = control_mode

        #instantiate agents and buffers
        self.replay_buffers = {}
        agents = {}
        for agent in self.agent_ids:

            #quick fix
            cfg.agent.params.obs_dim = obs_spaces[agent][0]
            cfg.agent.params.action_dim = act_spaces[agent][0]
            cfg.agent.params.action_range = act_ranges[agent]
        
            agents[agent] = hydra.utils.instantiate(learning_model)
            self.replay_buffers[agent] = ReplayBuffer(  obs_spaces[agent],
                                                        act_spaces[agent],
                                                        replay_buffer_cap,
                                                        self.device,
                                                        self.randomizer)
        
        self.agents = ModuleDict(agents)

        # agents with the name network structure
        self.aggregatable_agents = []
        if self.control_mode == 'unilateral':
            self.aggregatable_agents = self.agent_ids
        elif self.control_mode == 'bilateral':
            self.aggregatable_agents = self.agent_ids[:-1]
        else:
            raise Exception('no valid control_mode')


    def equalize_agents(self):

        for agent in self.aggregatable_agents:
            for param, target_param in zip(self.agents[self.agent_ids[0]].parameters(), self.agents[agent].parameters()):
                target_param.data.copy_(param.data)


    def reset(self):

        for agent in self.agent_ids:
            self.agents[agent].reset()


    def act(self, obs, sample=False, mode=None):
        
        actions = {}

        if mode == "eval":
            for agent in self.agent_ids:
                with utils.eval_mode(self.agents[agent]):
                    actions[agent] = self.agents[agent].act(obs[agent], sample)
        
        elif mode == "train":
            for agent in self.agent_ids:
                with utils.train_mode(self.agents[agent]):
                    actions[agent] = self.agents[agent].act(obs[agent], sample)

        elif mode == None:
            for agent in self.agent_ids:
                actions[agent] = self.agents[agent].act(obs[agent], sample)

        else:
            raise Exception("ERROR: NO VALID ACTING MODE!")
        
        return actions

    
    def update(self, loggers, step):

        for agent in self.agent_ids:
            self.agents[agent].update(self.replay_buffers[agent], loggers[agent], step)


    def federate(self, aggregate_actor, aggregate_critic, aggregate_target, aggregate_alpha, pre_weight, post_weight, first_post_weight, last_pre_weight):

            with torch.no_grad():
                
                orig_weights = {}
                
                # detach weights
                for agent in self.aggregatable_agents:

                    orig_weights[agent] = {}

                    if aggregate_actor:
                        orig_weights[agent]['actor'] = []
                        for weight in self.agents[agent].actor.parameters():
                            orig_weights[agent]['actor'].append(weight.detach())

                    if aggregate_critic:
                        orig_weights[agent]['critic_Q1'] = []
                        for weight in self.agents[agent].critic.Q1.parameters():
                            orig_weights[agent]['critic_Q1'].append(weight.detach())

                        orig_weights[agent]['critic_Q2'] = []
                        for weight in self.agents[agent].critic.Q2.parameters():
                            orig_weights[agent]['critic_Q2'].append(weight.detach())
                    
                    if aggregate_target:
                        orig_weights[agent]['target_Q1'] = []
                        for weight in self.agents[agent].critic_target.Q1.parameters():
                            orig_weights[agent]['target_Q1'].append(weight.detach())

                        orig_weights[agent]['target_Q2'] = []
                        for weight in self.agents[agent].critic_target.Q2.parameters():
                            orig_weights[agent]['target_Q2'].append(weight.detach())

                    if aggregate_alpha:
                        orig_weights[agent]['log_alpha'] = self.agents[agent].log_alpha.detach()


                temp_weights = {}

                # aggregate weights
                for id, agent in enumerate(self.aggregatable_agents):

                    temp_weights[agent] = {}

                    # if first follower    
                    if id == 0:
                        
                        if aggregate_actor:
                            temp_weights[agent]['actor'] = []
                            for weight_id in range(len(orig_weights[agent]['actor'])):
                                temp_weights[agent]['actor'].append((1 - first_post_weight) * orig_weights[agent]['actor'][weight_id] + first_post_weight * orig_weights[self.aggregatable_agents[id+1]]['actor'][weight_id])
                            
                        if aggregate_critic:
                            temp_weights[agent]['critic_Q1'] = []
                            for weight_id in range(len(orig_weights[agent]['critic_Q1'])):
                                temp_weights[agent]['critic_Q1'].append((1 - first_post_weight) * orig_weights[agent]['critic_Q1'][weight_id] + first_post_weight * orig_weights[self.aggregatable_agents[id+1]]['critic_Q1'][weight_id])

                            temp_weights[agent]['critic_Q2'] = []
                            for weight_id in range(len(orig_weights[agent]['critic_Q2'])):
                                temp_weights[agent]['critic_Q2'].append((1 - first_post_weight) * orig_weights[agent]['critic_Q2'][weight_id] + first_post_weight * orig_weights[self.aggregatable_agents[id+1]]['critic_Q2'][weight_id])

                        if aggregate_target:
                            temp_weights[agent]['target_Q1'] = []
                            for weight_id in range(len(orig_weights[agent]['target_Q1'])):
                                temp_weights[agent]['target_Q1'].append((1 - first_post_weight) * orig_weights[agent]['target_Q1'][weight_id] + first_post_weight * orig_weights[self.aggregatable_agents[id+1]]['target_Q1'][weight_id])

                            temp_weights[agent]['target_Q2'] = []
                            for weight_id in range(len(orig_weights[agent]['target_Q2'])):
                                temp_weights[agent]['target_Q2'].append((1 - first_post_weight) * orig_weights[agent]['target_Q2'][weight_id] + first_post_weight * orig_weights[self.aggregatable_agents[id+1]]['target_Q2'][weight_id])

                        if aggregate_alpha:
                            temp_weights[agent]['log_alpha'] = (1 - first_post_weight) * orig_weights[agent]['log_alpha'] + first_post_weight * orig_weights[self.aggregatable_agents[id+1]]['log_alpha']
                        

                    # if last follower
                    elif id == len(self.aggregatable_agents) - 1:
                        
                        if aggregate_actor:
                            temp_weights[agent]['actor'] = []
                            for weight_id in range(len(orig_weights[agent]['actor'])):
                                temp_weights[agent]['actor'].append((1 - last_pre_weight) * orig_weights[agent]['actor'][weight_id] + last_pre_weight * orig_weights[self.aggregatable_agents[id-1]]['actor'][weight_id])
                            
                        if aggregate_critic:
                            temp_weights[agent]['critic_Q1'] = []
                            for weight_id in range(len(orig_weights[agent]['critic_Q1'])):
                                temp_weights[agent]['critic_Q1'].append((1 - last_pre_weight) * orig_weights[agent]['critic_Q1'][weight_id] + last_pre_weight * orig_weights[self.aggregatable_agents[id-1]]['critic_Q1'][weight_id])

                            temp_weights[agent]['critic_Q2'] = []
                            for weight_id in range(len(orig_weights[agent]['critic_Q2'])):
                                temp_weights[agent]['critic_Q2'].append((1 - last_pre_weight) * orig_weights[agent]['critic_Q2'][weight_id] + last_pre_weight * orig_weights[self.aggregatable_agents[id-1]]['critic_Q2'][weight_id])

                        if aggregate_target:
                            temp_weights[agent]['target_Q1'] = []
                            for weight_id in range(len(orig_weights[agent]['target_Q1'])):
                                temp_weights[agent]['target_Q1'].append((1 - last_pre_weight) * orig_weights[agent]['target_Q1'][weight_id] + last_pre_weight * orig_weights[self.aggregatable_agents[id-1]]['target_Q1'][weight_id])

                            temp_weights[agent]['target_Q2'] = []
                            for weight_id in range(len(orig_weights[agent]['target_Q2'])):
                                temp_weights[agent]['target_Q2'].append((1 - last_pre_weight) * orig_weights[agent]['target_Q2'][weight_id] + last_pre_weight * orig_weights[self.aggregatable_agents[id-1]]['target_Q2'][weight_id])

                        if aggregate_alpha:
                            temp_weights[agent]['log_alpha'] = (1 - last_pre_weight) * orig_weights[agent]['log_alpha'] + last_pre_weight * orig_weights[self.aggregatable_agents[id-1]]['log_alpha']

                    else:
                        
                        if aggregate_actor:
                            temp_weights[agent]['actor'] = []
                            for weight_id in range(len(orig_weights[agent]['actor'])):
                                temp_weights[agent]['actor'].append((1 - pre_weight - post_weight) * orig_weights[agent]['actor'][weight_id] + pre_weight * orig_weights[self.aggregatable_agents[id-1]]['actor'][weight_id] + post_weight * orig_weights[self.aggregatable_agents[id+1]]['actor'][weight_id])
                            
                        if aggregate_critic:
                            temp_weights[agent]['critic_Q1'] = []
                            for weight_id in range(len(orig_weights[agent]['critic_Q1'])):
                                temp_weights[agent]['critic_Q1'].append((1 - pre_weight - post_weight) * orig_weights[agent]['critic_Q1'][weight_id] + pre_weight * orig_weights[self.aggregatable_agents[id-1]]['critic_Q1'][weight_id] + post_weight * orig_weights[self.aggregatable_agents[id+1]]['critic_Q1'][weight_id])

                            temp_weights[agent]['critic_Q2'] = []
                            for weight_id in range(len(orig_weights[agent]['critic_Q2'])):
                                temp_weights[agent]['critic_Q2'].append((1 - pre_weight - post_weight) * orig_weights[agent]['critic_Q2'][weight_id] + pre_weight * orig_weights[self.aggregatable_agents[id-1]]['critic_Q2'][weight_id] + post_weight * orig_weights[self.aggregatable_agents[id+1]]['critic_Q2'][weight_id])

                        if aggregate_target:
                            temp_weights[agent]['target_Q1'] = []
                            for weight_id in range(len(orig_weights[agent]['target_Q1'])):
                                temp_weights[agent]['target_Q1'].append((1 - pre_weight - post_weight) * orig_weights[agent]['target_Q1'][weight_id] + pre_weight * orig_weights[self.aggregatable_agents[id-1]]['target_Q1'][weight_id] + post_weight * orig_weights[self.aggregatable_agents[id+1]]['target_Q1'][weight_id])

                            temp_weights[agent]['target_Q2'] = []
                            for weight_id in range(len(orig_weights[agent]['target_Q2'])):
                                temp_weights[agent]['target_Q2'].append((1 - pre_weight - post_weight) * orig_weights[agent]['target_Q2'][weight_id] + pre_weight * orig_weights[self.aggregatable_agents[id-1]]['target_Q2'][weight_id] + post_weight * orig_weights[self.aggregatable_agents[id+1]]['target_Q2'][weight_id])

                        if aggregate_alpha:
                            temp_weights[agent]['log_alpha'] = (1 - pre_weight - post_weight) * orig_weights[agent]['log_alpha'] + pre_weight * orig_weights[self.aggregatable_agents[id-1]]['log_alpha'] + post_weight * orig_weights[self.aggregatable_agents[id+1]]['log_alpha']
                
                # update weights
                
                for agent in self.aggregatable_agents:
                    
                    if aggregate_actor:
                        for param, weight in zip(self.agents[agent].actor.parameters(), temp_weights[agent]['actor']):
                            param.copy_(weight)

                    if aggregate_critic:
                        for param, weight in zip(self.agents[agent].critic.Q1.parameters(), temp_weights[agent]['critic_Q1']):
                            param.copy_(weight)

                        for param, weight in zip(self.agents[agent].critic.Q2.parameters(), temp_weights[agent]['critic_Q2']):
                            param.copy_(weight)

                    if aggregate_target:
                        for param, weight in zip(self.agents[agent].critic_target.Q1.parameters(), temp_weights[agent]['target_Q1']):
                            param.copy_(weight)

                        for param, weight in zip(self.agents[agent].critic_target.Q2.parameters(), temp_weights[agent]['target_Q2']):
                            param.copy_(weight)

                    if aggregate_alpha:
                        self.agents[agent].log_alpha.copy_(temp_weights[agent]['log_alpha'])

                    

    def add_to_buffer(self, obs, actions, rewards, next_obs, done, dones_no_max):

        for agent in self.agent_ids:
            self.replay_buffers[agent].add(obs[agent], actions[agent], rewards[agent], next_obs[agent], done, dones_no_max[agent])


    def load_checkpoint(self, dir, checkpoint, device, replay_buffer_capacity):

        print("started loading checkpoint " + checkpoint)

        checkpoint_dir = os.path.join(dir, checkpoint)

        #load model parameters
        model_checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'), map_location=device)
        self.agents.load_state_dict(model_checkpoint['models'])
        step = model_checkpoint['step']
        episode = model_checkpoint['episode']
        min_step_num = model_checkpoint['min_step_num']
      
        for agent in self.agent_ids:
            self.agents[agent].critic_optimizer.load_state_dict(model_checkpoint['optims'][agent]['critic'])
            self.agents[agent].actor_optimizer.load_state_dict(model_checkpoint['optims'][agent]['actor'])
            self.agents[agent].log_alpha_optimizer.load_state_dict(model_checkpoint['optims'][agent]['alpha'])


        #load replay buffer entries
        if self.mode == 'train':
            
            rep_dir = os.path.join(checkpoint_dir, 'replay_buffers')

            for agent in self.agent_ids:
                data = np.load(os.path.join(rep_dir, agent + '.npz'))
                
                for i in range(min(step, replay_buffer_capacity)):

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

        print("loading checkpoint " + checkpoint + " finished")

        return step, episode, min_step_num
    

    def save_checkpoint(self, dir, step, episode, min_step_num):

        checkpoint = 'cp_{:d}'.format(step)

        print("started saving checkpoint " + checkpoint)
        
        checkpoint_dir = os.path.join(dir, checkpoint)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            
        #save model parameters
        state = {}
        state['step'] = step
        state['episode'] = episode
        state['min_step_num'] = min_step_num
        state['models'] = self.agents.state_dict()
        state['optims'] = {}
        
        for agent in self.agent_ids:
            state['optims'][agent] = {}
            state['optims'][agent]['critic'] = self.agents[agent].critic_optimizer.state_dict()
            state['optims'][agent]['actor'] = self.agents[agent].actor_optimizer.state_dict()
            state['optims'][agent]['alpha'] = self.agents[agent].log_alpha_optimizer.state_dict()

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
            
        print("saving checkpoint " + checkpoint + " finished")

    

class SharedMultiAgent(MultiAgent):

    def __init__(self, cfg, agent_ids, obs_space, act_space, act_range, replay_buffer_cap, device, mode, learning_model, randomizer):
        
        self.agent_ids = agent_ids
        self.device = device
        self.mode = mode
        self.randomizer = randomizer

        #instantiate agent and buffer
        cfg.agent.params.obs_dim = obs_space[0]
        cfg.agent.params.action_dim = act_space[0]
        cfg.agent.params.action_range = act_range
    
        self.agent = hydra.utils.instantiate(learning_model)
        self.replay_buffer = ReplayBuffer(  obs_space,
                                            act_space,
                                            replay_buffer_cap,
                                            self.device,
                                            self.randomizer)
        

    def reset(self):
        self.agent.reset()


    def act(self, obs, sample=False, mode=None):
        
        actions = {}

        if mode == "eval":
            for agent in self.agent_ids:
                with utils.eval_mode(self.agent):
                    actions[agent] = self.agent.act(obs[agent], sample)
        
        elif mode == "train":
            for agent in self.agent_ids:
                with utils.train_mode(self.agents[agent]):
                    actions[agent] = self.agent.act(obs[agent], sample)

        elif mode == None:
            for agent in self.agent_ids:
                actions[agent] = self.agent.act(obs[agent], sample)

        else:
            raise Exception("ERROR: NO VALID ACTING MODE!")
        
        return actions


    def update(self, logger, step):
        
        self.agent.update(self.replay_buffer, logger, step)


    def add_to_buffer(self, obs, actions, rewards, next_obs, done, dones_no_max):
        
        for agent in self.agent_ids:
            self.replay_buffer.add(obs[agent], actions[agent], rewards[agent], next_obs[agent], done, dones_no_max[agent])


    def save_checkpoint(self, dir, step, episode, min_step_num):
        
        checkpoint = 'cp_{:d}'.format(step)

        print("started saving checkpoint " + checkpoint)
        
        checkpoint_dir = os.path.join(dir, checkpoint)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            
        #save model parameters
        state = {}
        state['step'] = step
        state['episode'] = episode
        state['min_step_num'] = min_step_num
        state['models'] = self.agent.state_dict()
        state['optims'] = {}
    
        state['optims'] = {}
        state['optims']['critic'] = self.agent.critic_optimizer.state_dict()
        state['optims']['actor'] = self.agent.actor_optimizer.state_dict()
        state['optims']['alpha'] = self.agent.log_alpha_optimizer.state_dict()

        torch.save(state, os.path.join(checkpoint_dir, 'checkpoint.pt'))

        #save replay_buffer
        
        #save numpy arrays
        np_file = os.path.join(checkpoint_dir, 'replay_buffer.npz')
        
        np.savez(np_file,
                    obses = self.replay_buffer.obses,
                    next_obses = self.replay_buffer.next_obses,
                    actions = self.replay_buffer.actions,
                    rewards = self.replay_buffer.rewards,
                    not_dones = self.replay_buffer.not_dones,
                    not_dones_no_max = self.replay_buffer.not_dones_no_max
                    )
            
        print("saving checkpoint " + checkpoint + " finished")


    def load_checkpoint(self, dir, checkpoint, device, replay_buffer_capacity):
        
        print("started loading checkpoint " + checkpoint)

        checkpoint_dir = os.path.join(dir, checkpoint)

        #load model parameters
        model_checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'), map_location=device)
        self.agent.load_state_dict(model_checkpoint['models'])
        step = model_checkpoint['step']
        episode = model_checkpoint['episode']
        min_step_num = model_checkpoint['min_step_num']
      
        self.agent.critic_optimizer.load_state_dict(model_checkpoint['optims']['critic'])
        self.agent.actor_optimizer.load_state_dict(model_checkpoint['optims']['actor'])
        self.agent.log_alpha_optimizer.load_state_dict(model_checkpoint['optims']['alpha'])


        #load replay buffer entries
        if self.mode == 'train':

            data = np.load(os.path.join(checkpoint_dir, 'replay_buffer.npz'))
            
            for i in range(min(step, replay_buffer_capacity) * len(self.agent_ids)):

                obs = data['obses'][i]
                next_obs = data['next_obses'][i]
                action = data['actions'][i]
                reward = data['rewards'][i]
                not_done = data['not_dones'][i]
                not_done_no_max = data['not_dones_no_max'][i]

                self.replay_buffer.add(obs=obs,
                                        action=action,
                                        reward=reward,
                                        next_obs=next_obs,
                                        done=not not_done,
                                        done_no_max=not not_done_no_max)
        
            data.close()

        print("loading checkpoint " + checkpoint + " finished")

        return step, episode, min_step_num
            
            

            
            

