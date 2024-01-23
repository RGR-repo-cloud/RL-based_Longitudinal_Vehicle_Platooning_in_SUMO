#!/usr/bin/env python3
import datetime
from pathlib import Path
import os
from external_controllers.controllers import Fastbed

import utils
import hydra


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
        self.env = utils.import_flow_env(env_name=self.cfg.env, render=self.cfg.render, evaluate=False)
        self.agent_ids = self.env.agents

        self.controller = None
        if self.cfg.controller == "Fastbed":
            self.controller = Fastbed()
        

    def evaluate(self):
        average_episode_rewards = {}
        for agent in self.agent_ids:
            average_episode_rewards[agent] = 0
        for episode in range(self.cfg.num_eval_episodes):
            episode_step = 0
            
            obs = self.env.reset()
            
            done = False
            episode_rewards = {}
            for agent in self.agent_ids:
                    episode_rewards[agent] = 0

            while not done:

                actions = {}
                for agent in self.agent_ids:
                    actions[agent] = self.controller.get_accel(obs[agent])
                
                obs, rewards, dones, _ = self.env.step(actions)
                
                done = dones['__all__']
                for agent in self.agent_ids:
                    episode_rewards[agent] += rewards[agent]
                episode_step += 1

            for agent in self.agent_ids:
                average_episode_rewards[agent] += episode_rewards[agent]

        for agent in self.agent_ids:
            print("---------------------------------")
            print(agent)
            print(average_episode_rewards[agent])

    
@hydra.main(config_path='config/external_control_eval.yaml', strict=True)
def main(cfg):
    evaluator = Evaluator(cfg)
    
    evaluator.evaluate()


if __name__ == '__main__':
    main()