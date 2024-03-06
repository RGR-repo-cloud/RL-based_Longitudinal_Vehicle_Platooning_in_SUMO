#!/usr/bin/env python3
import datetime
from pathlib import Path
import os
from external_controllers.controllers import Fastbed, Ploeg

import utils
import hydra
import numpy as np


class Evaluator(object):
    def __init__(self, cfg):

        # create new working directory
        dir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d'), datetime.datetime.now().strftime('%H-%M'))
        Path(dir).mkdir(parents=True, exist_ok=True)
        os.chdir(dir)
            
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        
        #register environment
        self.env = utils.import_flow_env(env_name=self.cfg.env, render=self.cfg.render, evaluate=True)
        self.agent_ids = self.env.agents

        self.act_range = [  float(self.env.action_space[self.agent_ids[0]].low.min()),
                            float(self.env.action_space[self.agent_ids[0]].high.max())]

        self.controller = None
        if self.cfg.controller == "Fastbed":
            self.controller = Fastbed()
        elif self.cfg.controller == "Ploeg":
            self.controller = Ploeg()
        

    def evaluate(self):
        average_episode_rewards = {}
        for agent in self.agent_ids:
            average_episode_rewards[agent] = 0
        for episode in range(self.cfg.num_eval_episodes):
            episode_step = 0
            
            self.env.wrapped_env.set_mode('eval')
            self.env.wrapped_env.set_scenario(episode % self.env.wrapped_env.env_params.additional_params['num_scenarios'])
            obs = self.env.reset()
            
            done = False
            episode_rewards = {}
            for agent in self.agent_ids:
                    episode_rewards[agent] = 0

            while not done:

                actions = {}
                for agent in self.agent_ids:
                    actions[agent] = np.clip(self.controller.get_accel(obs[agent]), a_min=self.act_range[0], a_max=self.act_range[1])
                
                obs, rewards, dones, _ = self.env.step(actions)
                
                done = dones['__all__']
                for agent in self.agent_ids:
                    episode_rewards[agent] += rewards[agent]
                episode_step += 1

            for agent in self.agent_ids:
                average_episode_rewards[agent] += episode_rewards[agent]

        for agent in self.agent_ids:
            average_episode_rewards[agent] /= self.cfg.num_eval_episodes
            print("---------------------------------")
            print(agent)
            print(average_episode_rewards[agent])
        
        utils.print_accumulated_rewards(average_episode_rewards)
        utils.log_eval_data(self.work_dir, self.env.wrapped_env.eval_state_dict, self.env.wrapped_env.eval_reward_dict, self.env.wrapped_env.eval_leader_dict, self.agent_ids)

    
@hydra.main(config_path='config/external_control_eval.yaml', strict=True)
def main(cfg):
    evaluator = Evaluator(cfg)
    
    evaluator.evaluate()


if __name__ == '__main__':
    main()