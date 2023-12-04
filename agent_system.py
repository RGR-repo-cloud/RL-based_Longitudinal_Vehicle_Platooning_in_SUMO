from replay_buffer import ReplayBuffer
import hydra
import utils
import torch
import time


class MultiAgent:

    def __init__(self, cfg, agent_ids, obs_spaces, act_spaces, act_ranges, replay_buffer_caps, device):

        self.agent_ids = agent_ids
        self.device = device

        #instantiate agents and buffers
        self.replay_buffers = {}
        agents = {}
        optim_params_actor = []
        optim_params_critic = []
        optim_params_alpha = []

        for agent in self.agent_ids:

            #quick fix
            cfg.agent.params.obs_dim = obs_spaces[agent][0]
            cfg.agent.params.action_dim = act_spaces[agent][0]
            cfg.agent.params.action_range = act_ranges[agent]
        
            agents[agent] = hydra.utils.instantiate(cfg.agent)
            optim_params_actor.append({"params":agents[agent].actor.parameters(), "lr":agents[agent].actor_lr, "betas":agents[agent].actor_betas})
            optim_params_critic.append({"params":agents[agent].critic.parameters(), "lr":agents[agent].critic_lr, "betas":agents[agent].critic_betas})
            optim_params_alpha.append({"params":[agents[agent].log_alpha], "lr":agents[agent].alpha_lr, "betas":agents[agent].alpha_betas})

            self.replay_buffers[agent] = ReplayBuffer(  obs_spaces[agent],
                                                        act_spaces[agent],
                                                        replay_buffer_caps,
                                                        self.device)
            
        self.actors_optimizer = torch.optim.Adam(optim_params_actor)
        self.critics_optimizer = torch.optim.Adam(optim_params_critic)
        self.log_alphas_optimizer = torch.optim.Adam(optim_params_alpha)
        
        for agent in self.agent_ids:
            agents[agent].train()
        
        self.agents = torch.nn.ModuleDict(agents)


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

        obs, actions, rewards, next_obs, not_done, not_done_no_max = {}, {}, {}, {}, {}, {}

        for agent in self.agent_ids:
            obs[agent], actions[agent], rewards[agent], next_obs[agent], not_done[agent], not_done_no_max[agent] = self.replay_buffers[agent].sample(self.agents[agent].batch_size)
            loggers[agent].log('train/batch_reward', rewards[agent].mean(), step)

        #update critics
        critics_loss = 0
        for agent in self.agent_ids:
            critics_loss += self.agents[agent].update_critic(   obs[agent],
                                                                actions[agent],
                                                                rewards[agent],
                                                                next_obs[agent],
                                                                not_done[agent],
                                                                loggers[agent],
                                                                step)
            self.agents[agent].critic.log(loggers[agent], step)
        
            


        #update actors and alphas
        actors_loss = 0
        alphas_loss = 0
        if step % self.agents[self.agent_ids[0]].actor_update_frequency == 0: #quick fix
            log_probs = {}
            
            for agent in self.agent_ids:
                actor_loss, log_probs[agent] = self.agents[agent].update_actor( obs[agent],
                                                                                loggers[agent],
                                                                                step)
                actors_loss += actor_loss
                self.agents[agent].actor.log(loggers[agent], step)
                
            if self.agents[self.agent_ids[0]].learnable_temperature: #quick fix
                for agent in self.agent_ids:
                    alphas_loss += self.agents[agent].update_alpha(loggers[agent], step, log_probs[agent])
    
        
        loss = critics_loss + actors_loss + alphas_loss

        self.critics_optimizer.zero_grad()
        self.actors_optimizer.zero_grad()
        self.log_alphas_optimizer.zero_grad()
        
        loss.backward()

        self.critics_optimizer.step()
        self.actors_optimizer.step()
        self.log_alphas_optimizer.step()



        #update targets
        if step % self.agents[self.agent_ids[0]].critic_target_update_frequency == 0: #quick fix
            for agent in self.agent_ids:
                utils.soft_update_params(self.agents[agent].critic,
                                         self.agents[agent].critic_target,
                                         self.agents[agent].critic_tau)
            


        


    def add_to_buffers(self, obs, actions, rewards, next_obs, done, done_no_max):

        for agent in self.agent_ids:
            self.replay_buffers[agent].add(obs[agent], actions[agent], rewards[agent], next_obs[agent], done, done_no_max)