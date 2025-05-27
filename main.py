# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import bz2
from datetime import datetime
import os
import pickle

import atari_py
import numpy as np
import torch
from tqdm import trange
import random

from agent import Agent
from env import Env
from memory import ReplayMemory
from test import test
from utils import REWARD_LOGGER

from glob import glob


"""
Train code for Action Q-Transformer
"""


# Example code

## baseline model：Rainbow
# python main.py --id **_rainbow --game ** --T-max 50000000 --architecture canonical --evaluation-interval 100000 --save-interval 500000 --cuda-device cuda:0 --memory

## AQT (patch size: 7*7)
# python main.py --id **_aqt --game ** --architecture aqt --T-max 50000000 --evaluation-interval 100000 --save-interval 500000 --cuda-device cuda:0 --memory


# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='AQT')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--game', type=str, default='space_invaders', choices=atari_py.list_games(), help='ATARI game')
parser.add_argument('--T-max', type=int, default=int(30e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient', 'aqt'], metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--save-interval', type=int, default=500000, help='a')
# Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint-interval', default=0, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
parser.add_argument('--memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
parser.add_argument('--cuda-device', type=str, default='cuda:0', choices=['cuda:0', 'cuda:1'], metavar='CUDA', help='Cuda Device')


# Setup
args = parser.parse_args()

# Make save directry
results_dir = os.path.join('results', args.id)
if not os.path.exists(results_dir):
  os.makedirs(results_dir)
models_dir = os.path.join(results_dir, "models")
if not os.path.exists(models_dir):
  os.makedirs(models_dir)
metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf'), 't_epi': 0, 't_scores': []}
memory_path = os.path.join(results_dir, "memory.pickle")

# Print options
print(' ' * 26 + 'Options')
confing_txt = os.path.join(results_dir, "config.txt")
with open(confing_txt, 'w') as f:
  for k, v in vars(args).items():
    f.write(k + ': ' + str(v) + '\n')
    print(' ' * 26 + k + ': ' + str(v))

# Setup logger
logger = REWARD_LOGGER(results_dir + "/")

# Setup seed and device
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
#torch.manual_seed(np.random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
  print("Use GPU: {}".format(args.cuda_device))
  args.device = torch.device(args.cuda_device)
  #torch.cuda.manual_seed(np.random.randint(1, 10000))
  torch.cuda.manual_seed(args.seed)
  torch.backends.cudnn.enabled = args.enable_cudnn
else:
  print("Use CPU")
  args.device = torch.device('cpu')


# Simple ISO 8601 timestamped logger
def log(s):
  print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


def load_memory(path, disable_bzip):
  if disable_bzip:
    with open(path, 'rb') as pickle_file:
      return pickle.load(pickle_file)
  else:
    with bz2.open(path, 'rb') as zipped_pickle_file:
      return pickle.load(zipped_pickle_file)


def save_memory(memory, path, disable_bzip):
  if disable_bzip:
    with open(path, 'wb') as pickle_file:
      pickle.dump(memory, pickle_file)
  else:
    with bz2.open(path, 'wb') as zipped_pickle_file:
      pickle.dump(memory, zipped_pickle_file)


# Environment
env = Env(args)
env.train()
action_space = env.action_space()

# Agent
dqn = Agent(args, env)

# If a model is provided, and evaluate is false, presumably we want to resume, so try to load memory
load_step = None
if args.model is not None:
  if not args.memory:
    raise ValueError('Cannot resume training without memory save path. Aborting...')
  elif not os.path.exists(memory_path):
    raise ValueError('Could not find memory file at {path}. Aborting...'.format(path=memory_path))

  print("Loading memory...")
  mem = load_memory(memory_path, args.disable_bzip_memory)

  print("Loading model...")
  models_list = glob(models_dir+"/model_*.pth")
  for idx in range(len(models_list)):
    models_list[idx] = int(models_list[idx].replace(models_dir+"/model_", "").replace(".pth", ""))
  load_model_path = os.path.join(models_dir, "model_{}.pth".format(max(models_list)))
  load_step = dqn.model_load(load_model_path)

  print("Loading metrics...")
  metrics_path = os.path.join(results_dir, 'metrics.pth')
  load_metrics = torch.load(metrics_path)
  metrics["steps"] = load_metrics["steps"]
  metrics["rewards"] = load_metrics["rewards"]
  metrics["Qs"] = load_metrics["Qs"]
  metrics["best_avg_reward"] = load_metrics["best_avg_reward"]
  metrics["t_epi"] = load_metrics["t_epi"]
  metrics["t_scores"] = load_metrics["t_scores"]

  print("#"*10 + " Load parameter " + "#"*10)
  print("load model: {}".format(load_model_path))
  print("metrics: {}".format(metrics_path))
  print("load step: {}".format(load_step))
  print("load episode: {}".format(metrics["t_epi"]))
  print("#"*36)

else:
  mem = ReplayMemory(args, args.memory_capacity)

priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)


# Construct validation memory
val_mem = ReplayMemory(args, args.evaluation_size)
memory_T, done = 0, True
print("Collecting experience...")
while memory_T < args.evaluation_size:
  if done:
    state = env.reset()

  next_state, _, done = env.step(np.random.randint(0, action_space))
  val_mem.append(state, -1, 0.0, done)
  state = next_state
  memory_T += 1
print("Filled replay buffer")

if args.evaluate:
  print("Evaluating...")
  dqn.eval()  # Set DQN (online network) to evaluation mode
  avg_reward, avg_Q = test(args, 0, dqn, val_mem, metrics, results_dir, evaluate=True)  # Test
  print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
else:
  print("Training...")
  # record
  t_reward_sum = 0.0
  # Training loop
  dqn.train()
  done = True

  if load_step == None:
    start_step = 1
  else:
    start_step = load_step+1

  for T in trange(start_step, args.T_max + 1):
    if done:
      metrics['t_epi'] += 1
      state = env.reset()

    if T % args.replay_frequency == 0:
      dqn.reset_noise()  # Draw a new set of noisy weights

    action = dqn.act(state)  # Choose an action greedily (with noisy weights)
    next_state, reward, done = env.step(action)  # Step
    t_reward_sum += reward
    # record score
    if done:
      metrics['t_scores'].append(t_reward_sum)
      t_mean_score = sum(metrics['t_scores'][-100:]) / len(metrics['t_scores'][-100:])
      logger.record_train_log(T, t_reward_sum, t_mean_score, metrics['t_epi'])
      log('Train: T = {0} / {1} | score: {2} | mean {3:.2f}'.format(T, args.T_max, t_reward_sum, t_mean_score))
      t_reward_sum = 0.0

    if args.reward_clip > 0:
      reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
    mem.append(state, action, reward, done)  # Append transition to memory

    # Train and test
    if T >= args.learn_start:
      mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1
      
      if T % args.replay_frequency == 0:
        dqn.learn(mem)  # Train with n-step distributional double-Q learning

      if T % args.evaluation_interval == 0:
        dqn.eval()  # Set DQN (online network) to evaluation mode
        avg_reward, avg_Q = test(args, T, dqn, val_mem, metrics, results_dir, models_dir)  # Test
        logger.record_eval_log(T, avg_reward, metrics['t_epi'])
        log('Eval: T = {0} / {1} | Avg. reward: {2:.2f} | Avg. Q: {3:.3f}'.format(T, args.T_max, avg_reward, avg_Q))
        dqn.train()  # Set DQN (online network) back to training mode

        # If memory path provided, save it
        if args.memory:
          save_memory(mem, memory_path, args.disable_bzip_memory)

      # Update target network
      if T % args.target_update == 0:
        dqn.update_target_net()
    
    state = next_state

env.close()
