from smac.env import StarCraft2Env
import numpy as np
import qmix
import torch
import os
import argparse
from time import gmtime, strftime
from torch.utils.tensorboard import SummaryWriter
import runner

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def main(arglist):
    current_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    writer = SummaryWriter(log_dir='./logs/' + current_time + '-snake')
    actors = 6
    if arglist.train == False:
        actors = 1
    env_runner = runner.Runner(arglist, arglist.scenario, actors)


    while arglist.train or env_runner.episode < 1:
        env_runner.reset()
        replay_buffers = env_runner.run()
        for replay_buffer in replay_buffers:
            env_runner.qmix_algo.episode_batch.add(replay_buffer)
        env_runner.qmix_algo.train()
        for episode in env_runner.episodes:
            env_runner.qmix_algo.update_targets(episode)

        for episode in env_runner.episodes:
            if episode % 500 == 0 and arglist.train:
                env_runner.qmix_algo.save_model('./saved/agents_' + str(episode))

        print(env_runner.win_counted_array)
        for idx, episode in enumerate(env_runner.episodes):
            print("Total reward in episode {} = {} and global step: {}".format(episode, env_runner.episode_reward[idx], env_runner.episode_global_step))

            if arglist.train:
                writer.add_scalar('Reward', env_runner.episode_reward[idx], episode)
                writer.add_scalar('Victory', env_runner.win_counted_array[idx], episode)


    if arglist.train == False:
        env_runner.save()
    
    env_runner.close()

def parse_args():
    parser = argparse.ArgumentParser('Reinforcement Learning parser for DQN')

    parser.add_argument('--train', action='store_true') #"3m"
    parser.add_argument('--load-episode-saved', type=int, default=8000)
    parser.add_argument('--scenario', type=str, default="3m")

    return parser.parse_args()

if __name__ == "__main__":
    try:
        os.mkdir('./saved')
    except OSError:
        print ("Creation of the directory failed")
    else:
        print ("Successfully created the directory")
    arglist = parse_args()
    main(arglist)