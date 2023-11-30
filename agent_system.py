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