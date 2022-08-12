import torch
import gym

import argparse

parser = argparse.ArgumentParser(description='Process some information.')
parser.add_argument('--env_name', type=str, default='CartPole-v0', help='name of the environment to run')
parser.add_argument('--seed', type=int, default=1234, metavar='N', help='random seed (default: 1234)')

args = parser.parse_args()
env = gym.make(args.env_name)
env.seed(args.seed)
torch.manual_seed(args.seed)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.n



