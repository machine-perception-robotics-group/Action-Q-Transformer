# -*- coding: utf-8 -*-
from __future__ import division
import os
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import torch
import cv2
import numpy as np
import csv

from env import Env

from utils import make_en_attention, make_en_img, make_de_attention, make_de_img, make_en_sal, make_de_sal, min_max, make_en_attimg, make_de_attimg
from tqdm import tqdm

from saliency import *


"""
Classes and functions related to testing, evaluation and attention visualization, etc.
"""


# Test model
def test(args, T, dqn, val_mem, metrics, results_dir, models_dir=None, evaluate=False):
  env = Env(args)
  env.eval()
  metrics['steps'].append(T)
  T_rewards, T_Qs = [], []

  # Test performance over several episodes
  done = True
  for _ in range(args.evaluation_episodes):
    while True:
      if done:
        state, reward_sum, done = env.reset(), 0, False

      action = dqn.act_e_greedy(state)  # Choose an action ε-greedily
      state, reward, done = env.step(action)  # Step
      reward_sum += reward
      if args.render:
        env.render()

      if done:
        T_rewards.append(reward_sum)
        break
  env.close()

  # Test Q-values over validation memory
  for state in val_mem:  # Iterate over valid states
    T_Qs.append(dqn.evaluate_q(state))

  avg_reward, avg_Q = sum(T_rewards) / len(T_rewards), sum(T_Qs) / len(T_Qs)
  if not evaluate:
    # Append to results and save metrics
    metrics['rewards'].append(T_rewards)
    metrics['Qs'].append(T_Qs)

    # Plot
    _plot_line(metrics['steps'], metrics['rewards'], 'Reward', path=results_dir)
    _plot_line(metrics['steps'], metrics['Qs'], 'Q', path=results_dir)

    # Save model parameters
    if (T % args.save_interval) == 0:
      dqn.save(models_dir, T, name="model_{}.pth".format(T))
      torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))
    # Save model parameters if improved
    if avg_reward > metrics['best_avg_reward']:
      metrics['best_avg_reward'] = avg_reward
      dqn.save(models_dir, T, name="best_model.pth")

  # Return average reward and Q-value
  return avg_reward, avg_Q


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
  max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

  ys = torch.tensor(ys_population, dtype=torch.float32)
  ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(1).squeeze()
  ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

  trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
  trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
  trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
  trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
  trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

  plotly.offline.plot({
    'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
    'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)


# eval model
def eval_model(args, dqn):
  env = Env(args)
  env.eval()
  T_rewards, T_Qs = [], []

  # Test performance over several episodes
  done = True
  for epi in tqdm(range(args.evaluation_episodes)):
    while True:
      if done:
        state, reward_sum, done = env.reset(), 0, False

      action = dqn.act_e_greedy(state, epsilon=0.0)  # Choose an action ε-greedily
      state, reward, done = env.step(action)  # Step
      reward_sum += reward
      if args.render:
        env.render()

      if done:
        T_rewards.append(reward_sum)
        break
  env.close()

  avg_reward = sum(T_rewards) / len(T_rewards)
  max_reward = max(T_rewards)
  min_reward = min(T_rewards)
  std_reward = np.std(T_rewards)

  return avg_reward, max_reward, min_reward, std_reward


# eval random action
def eval_random(args, action_space):
  env = Env(args)
  env.eval()
  T_rewards, T_Qs = [], []

  # Test performance over several episodes
  done = True
  for epi in tqdm(range(args.evaluation_episodes)):
    while True:
      if done:
        state, reward_sum, done = env.reset(), 0, False

      action = np.random.randint(0, action_space)
      state, reward, done = env.step(action)  # Step
      reward_sum += reward
      if args.render:
        env.render()

      if done:
        T_rewards.append(reward_sum)
        break
  env.close()

  avg_reward = sum(T_rewards) / len(T_rewards)
  max_reward = max(T_rewards)
  min_reward = min(T_rewards)
  std_reward = np.std(T_rewards)

  return avg_reward, max_reward, min_reward, std_reward



# Test model and Visualize attention
def test_att(args, dqn, results_dir, models_dir=None):
  env = Env(args)
  env.eval()
  action_num = env.action_space()
  print("action space: {}".format(action_num))
  T_rewards = []

  # Test performance over several episodes
  done = True
  for epi in range(args.evaluation_episodes):
    # make directry
    epi_dir = os.path.join(results_dir, "epi{}".format(epi+1))
    if not os.path.exists(epi_dir):
      os.makedirs(epi_dir)
    if not os.path.exists(epi_dir + "/raw_img/"):
      os.makedirs(epi_dir + "/raw_img/")
    if not os.path.exists(epi_dir + "/encoder/"):
      os.mkdir(epi_dir + "/encoder/")
    for i in range(env.action_space()):
      if not os.path.exists(epi_dir + "/decoder_act{}/".format(i)):
        os.mkdir(epi_dir + "/decoder_act{}/".format(i))
      
    
    step = 0
    while True:
      if done:
        raw_list, en_list, de_list, T_Qs = [], [], [], []
        state, reward_sum, done = env.reset(), 0, False
        qdif, max_qdif_step = 0.0, 0

      action, q_val, en_atts, de_atts = dqn.eval_act(state)  # Choose an action ε-greedily

      # range of Q-values for all actions
      if qdif <= (max(q_val).numpy() - min(q_val).numpy()):
        max_qdif_step = step
        qdif = (max(q_val).numpy() - min(q_val).numpy())

      # get raw image
      raw_img = env.render('rgb_array')
      raw_list.append(raw_img)

      # calculate attention
      en_attention = make_en_attention(args, en_atts)
      en_list.append(en_attention)
      de_attention = make_de_attention(args, de_atts, action_num)
      de_list.append(de_attention)

      T_Qs.append(q_val)

      # 1 step
      state, reward, done = env.step(action)
      reward_sum += reward
      step += 1

      if done:
        print("Episode {}/{}: {}".format(epi+1, args.evaluation_episodes, reward_sum))
        print("Frame: {}, Q diff: {}".format(max_qdif_step, qdif))
        T_rewards.append(reward_sum)
        break
    
    # save raw image
    cv2.imwrite(epi_dir + "/raw_img/raw_{0:06d}.png".format(0), raw_list[0])

    # calculate max and min for normalize attention
    raw_list, en_list, de_list = np.array(raw_list), np.array(en_list), np.array(de_list)
    en_max, en_min = en_list.reshape(-1).max(), en_list.reshape(-1).min()
    de_max, de_min = de_list.reshape(-1).max(), de_list.reshape(-1).min()
    

    # encoder attention (mean)
    en_mean = np.zeros((en_list.shape[1], en_list.shape[2]))
    for idx in range(en_list.shape[0]):
      en_mean += en_list[idx]
    en_mean_max = max(en_mean.flatten())
    en_mean_min = min(en_mean.flatten())
    for x in range(en_mean.shape[0]):
      for y in range(en_mean.shape[1]):
        en_mean[x][y] = en_mean[x][y] / en_mean_max
    # imaging encoder attention (mean)
    make_en_img(en_mean * 255, raw_list[0], 0, epi_dir, mode="mean")


    # encoder attention and decoder attention 
    for idx in tqdm(range(len(raw_list))):
      q_val = T_Qs[idx]
      raw_img = raw_list[idx]
      en_att = en_list[idx]
      en_att = min_max(en_att, en_min, en_max)
      de_att = de_list[idx]
      de_att = min_max(de_att, de_min, de_max)

      # imaging encoder attention and decoder attention
      cv2.imwrite(epi_dir + "/raw_img/raw_{0:06d}.png".format(idx), raw_img)
      make_en_img(en_att * 255, raw_img, idx, epi_dir)
      make_de_img(args, q_val, de_att * 255, idx, action_num, epi_dir)

  env.close()

  avg_reward = sum(T_rewards) / len(T_rewards)

  # Return average reward and Q-value
  return avg_reward





# Test model and Visualize attention for Language explanation
def test_att_lang(args, dqn, results_dir, models_dir=None):
  env = Env(args)
  env.eval()
  action_num = env.action_space()
  print("action space: {}".format(action_num))
  T_rewards = []

  # Test performance over several episodes
  done = True
  for epi in range(args.evaluation_episodes):
    # make directry
    epi_dir = os.path.join(results_dir, "epi{}".format(epi+1))
    if not os.path.exists(epi_dir):
      os.makedirs(epi_dir)
    if not os.path.exists(epi_dir + "/raw_img/"):
      os.makedirs(epi_dir + "/raw_img/")
    if not os.path.exists(epi_dir + "/encoder/"):
      os.mkdir(epi_dir + "/encoder/")
    for i in range(env.action_space()):
      if not os.path.exists(epi_dir + "/decoder_act{}/".format(i)):
        os.mkdir(epi_dir + "/decoder_act{}/".format(i))
    if not os.path.exists(epi_dir + "/encoder_att/"):
      os.mkdir(epi_dir + "/encoder_att/")
    for i in range(env.action_space()):
      if not os.path.exists(epi_dir + "/decoder_att_act{}/".format(i)):
        os.mkdir(epi_dir + "/decoder_att_act{}/".format(i))
    
    step = 0
    while True:
      if done:
        raw_list, en_list, de_list, T_Qs = [], [], [], []
        state, reward_sum, done = env.reset(), 0, False
        qdif, max_qdif_step = 0.0, 0

      action, q_val, en_atts, de_atts = dqn.eval_act(state)  # Choose an action ε-greedily

      # range of Q-values for all actions
      if qdif <= (max(q_val).numpy() - min(q_val).numpy()):
        max_qdif_step = step
        qdif = (max(q_val).numpy() - min(q_val).numpy())

      # get raw image
      raw_img = env.render('rgb_array')
      raw_list.append(raw_img)

      # calculate attention
      en_attention = make_en_attention(args, en_atts)
      en_list.append(en_attention)
      de_attention = make_de_attention(args, de_atts, action_num)
      de_list.append(de_attention)

      T_Qs.append(q_val)

      # 1 step
      state, reward, done = env.step(action) 
      reward_sum += reward
      step += 1

      if done:
        print("Episode {}/{}: {}".format(epi+1, args.evaluation_episodes, reward_sum))
        print("Frame: {}, Q diff: {}".format(max_qdif_step, qdif))
        T_rewards.append(reward_sum)
        break
    
    # save raw image
    cv2.imwrite(epi_dir + "/raw_img/raw_{0:06d}.png".format(0), raw_list[0])

    # calculate max and min for normalize attention
    raw_list, en_list, de_list = np.array(raw_list), np.array(en_list), np.array(de_list)
    en_max, en_min = en_list.reshape(-1).max(), en_list.reshape(-1).min()
    de_max, de_min = de_list.reshape(-1).max(), de_list.reshape(-1).min()
    
    # encoder attention (mean)
    en_mean = np.zeros((en_list.shape[1], en_list.shape[2]))
    for idx in range(en_list.shape[0]):
      en_mean += en_list[idx]
    en_mean_max = max(en_mean.flatten())
    en_mean_min = min(en_mean.flatten())
    for x in range(en_mean.shape[0]):
      for y in range(en_mean.shape[1]):
        en_mean[x][y] = en_mean[x][y] / en_mean_max
    # imaging encoder attention (mean)
    make_en_img(en_mean * 255, raw_list[0], 0, epi_dir, mode="mean")

    # encoder attention and decoder attention 
    action_list = []
    for idx in tqdm(range(len(raw_list))):
      q_val = T_Qs[idx]
      raw_img = raw_list[idx]
      en_att = en_list[idx]
      en_att = min_max(en_att, en_min, en_max)
      de_att = de_list[idx]
      de_att = min_max(de_att, de_min, de_max)

      # imaging encoder attention and decoder attention
      cv2.imwrite(epi_dir + "/raw_img/raw_{0:06d}.png".format(idx), cv2.resize(raw_img, dsize=(200, 200)))
      make_en_img(en_att * 255, raw_img, idx, epi_dir)
      make_de_img(args, q_val, de_att * 255, idx, action_num, epi_dir)
      make_en_attimg(en_att * 255, raw_img, idx, epi_dir)
      make_de_attimg(args, q_val, de_att * 255, idx, action_num, epi_dir)

      # record action
      action_list.append(q_val.argmax().item())

    # save action list
    with open(os.path.join(epi_dir,'actions.csv'), 'w') as f:
      writer = csv.writer(f)
      for idx in range(len(raw_list)):
        writer.writerow([idx, action_list[idx]])

  env.close()

  avg_reward = sum(T_rewards) / len(T_rewards)

  # Return average reward and Q-value
  return avg_reward





# Test model and Visualize saliency
def test_sal(args, dqn, results_dir, density, radius, meta, models_dir=None):
  env = Env(args)
  env.eval()
  action_num = env.action_space()
  print("action space: {}".format(action_num))
  T_rewards = []

  # Test performance over several episodes
  done = True
  for epi in range(args.evaluation_episodes):
    # make directry
    epi_dir = os.path.join(results_dir, "epi{}".format(epi+1))
    if not os.path.exists(epi_dir):
      os.makedirs(epi_dir)
    if not os.path.exists(epi_dir + "/raw_img/"):
      os.makedirs(epi_dir + "/raw_img/")
    if not os.path.exists(epi_dir + "/encoder_sal/"):
      os.mkdir(epi_dir + "/encoder_sal/")
    for i in range(env.action_space()):
      if not os.path.exists(epi_dir + "/decoder_sal{}/".format(i)):
        os.mkdir(epi_dir + "/decoder_sal{}/".format(i))
    
    step = 0
    while True:
      if done:
        raw_list, en_list, de_list, T_Qs = [], [], [], []
        state, reward_sum, done = env.reset(), 0, False
        qdif, max_qdif_step = 0.0, 0

      action, q_val, _, __ = dqn.eval_act(state)  # Choose an action ε-greedily

      if qdif <= (max(q_val).numpy() - min(q_val).numpy()):
        max_qdif_step = step
        qdif = (max(q_val).numpy() - min(q_val).numpy())

      raw_img = env.render('rgb_array')
      raw_list.append(raw_img)

      en_sal = score_frame(args, dqn, state, radius, density, interp_func=occlude, mode="V")
      en_sal = saliency_preprocess(en_sal, fudge_factor=meta['critic_ff'])
      en_list.append(en_sal)

      de_sals = []
      for act_idx in range(action_num):
        de_sal = score_frame(args, dqn, state, radius, density, interp_func=occlude, mode="Q", act=act_idx)
        de_sal = saliency_preprocess(de_sal, fudge_factor=meta['actor_ff'])
        de_sals.append(de_sal)
      de_list.append(de_sals)

      T_Qs.append(q_val)

      state, reward, done = env.step(action)  # Step
      reward_sum += reward
      step += 1

      if done:
        print("Episode {}/{}: {}".format(epi+1, args.evaluation_episodes, reward_sum))
        print("Frame: {}, Q diff: {}".format(max_qdif_step, qdif))
        T_rewards.append(reward_sum)
        break
    
    # save raw image
    cv2.imwrite(epi_dir + "/raw_img/raw_{0:06d}.png".format(0), raw_list[0])

    # calculate max and min for normalize saliency
    raw_list, en_list, de_list = np.array(raw_list), np.array(en_list), np.array(de_list)
    en_max, en_min = en_list.reshape(-1).max(), en_list.reshape(-1).min()
    de_max, de_min = de_list.reshape(-1).max(), de_list.reshape(-1).min()
    

    for idx in tqdm(range(len(raw_list))):
      q_val = T_Qs[idx]
      raw_img = raw_list[idx]

      # normalize saliency map
      en_sal = en_list[idx]
      en_sal = min_max(en_sal, en_min, en_max)
      de_sal = de_list[idx]
      de_sal = min_max(de_sal, de_min, de_max)

      # save saliency map
      cv2.imwrite(epi_dir + "/raw_img/raw_{0:06d}.png".format(idx), raw_img)
      make_en_sal(en_sal * 255, idx, epi_dir)
      make_de_sal(args, q_val, de_sal * 255, idx, action_num, epi_dir)

  env.close()

  avg_reward = sum(T_rewards) / len(T_rewards)

  # Return average reward and Q-value
  return avg_reward