# -*- coding: utf-8 -*-

# Visualizing and Understanding Atari Agents | Sam Greydanus | 2017 | MIT License
from __future__ import print_function
import warnings ; warnings.filterwarnings('ignore') # mute warnings, live dangerously ;)

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import cv2


"""
Classes and functions related to saliency map calculation
"""


# prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.
prepro = lambda img: np.array(Image.fromarray(img[35:195].mean(2)).resize(size=(84,84))).astype(np.float32).reshape(1,80,80)/255.
searchlight = lambda I, mask: I*mask + gaussian_filter(I, sigma=3)*(1-mask) # choose an area NOT to blur
occlude = lambda I, mask: I*(1-mask) + gaussian_filter(I, sigma=3)*mask # choose an area to blur

def get_mask(center, size, r):
    y,x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
    keep = x*x + y*y <= 1
    mask = np.zeros(size) ; mask[keep] = 1 # select a circle of pixels
    mask = gaussian_filter(mask, sigma=r) # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    return mask/mask.max()

def run_through_model(args, dqn, state, act, interp_func=None, mask=None, blur_memory=None, mode='actor'):
    if mask is not None:
        assert(interp_func is not None, "interp func cannot be none")
        mask = np.repeat(mask[np.newaxis, :, :], 4, axis=0)
        state = interp_func(state.cpu(), mask).reshape(4,84,84) # perturb input I -> I'
        state = state.to(torch.float32).to(device=args.device)
    # if blur_memory is not None: cx.mul_(1-blur_memory) # perturb memory vector
    # return player.action_sliency(state, hx_cx)[0] if mode == 'V' else player.action_sliency(state, hx_cx)[1]
    _, q_val, __, ___, state_v, adv =  dqn.eval_act_sal(state)
    return q_val[act] if mode == 'Q' else state_v


def score_frame(args, dqn, state, r, d, interp_func, mode='Q', act=None):
    # r: radius of blur
    # d: density of scores (if d==1, then get a score for every pixel...
    #    if d==2 then every other, which is 25% of total pixels for a 2D image)
    assert mode in ['Q', 'V'], 'mode must be either "Q" or "V"'
    L = run_through_model(args, dqn, state, act, interp_func, mask=None, mode=mode)
    scores = np.zeros((int(84/d)+1,int(84/d)+1)) # saliency scores S(t,i,j)
    for i in range(0,84,d):
        for j in range(0,84,d):
            mask = get_mask(center=[i,j], size=[84,84], r=r)
            l = run_through_model(args, dqn, state, act, interp_func, mask=mask, mode=mode)
            # scores[int(i/d),int(j/d)] = (L-l).pow(2).sum().mul_(.5).data[0]
            scores[int(i/d),int(j/d)] = (L-l).pow(2).sum().mul_(.5).data
    pmax = scores.max()
    # scores = imresize(scores, size=[80,80], interp='bilinear').astype(np.float32)
    scores = np.array(Image.fromarray(scores).resize(size=(84,84), resample=Image.BILINEAR)).astype(np.float32)
    return pmax * scores / scores.max()

def saliency_preprocess(saliency, fudge_factor, sigma=0):
    pmax = saliency.max()
    # S = imresize(saliency, size=[160,160], interp='bilinear').astype(np.float32)
    # S = np.array(Image.fromarray(saliency).resize(size=(160,160), resample=Image.BILINEAR)).astype(np.float32)
    # S = np.array(Image.fromarray(saliency).resize(size=(160,(crops[1]+160)-crops[0]), resample=Image.BILINEAR)).astype(np.float32)
    S = np.array(Image.fromarray(saliency).resize(size=(210,160), resample=Image.BILINEAR)).astype(np.float32)
    S = S if sigma == 0 else gaussian_filter(S, sigma=sigma)
    S -= S.min() ; S = fudge_factor*pmax * S / S.max()
    return S

def saliency_on_atari_frame(saliency, atari, crops, fudge_factor, channel=2, sigma=0, max_sal=0, min_sal=0):
    # sometimes saliency maps are a bit clearer if you blur them
    # slightly...sigma adjusts the radius of that blur
    # pmax = saliency.max()
    # # S = imresize(saliency, size=[160,160], interp='bilinear').astype(np.float32)
    # S = np.array(Image.fromarray(saliency).resize(size=(160,160), resample=Image.BILINEAR)).astype(np.float32)
    # S = S if sigma == 0 else gaussian_filter(S, sigma=sigma)
    # S -= S.min() ; S = fudge_factor*pmax * S / S.max()
    S = saliency
    I = atari.astype('uint8')

    # S = min_max_normalization(S, l_max=max_sal, l_min=min_sal) * 255.0
    S_jet = np.zeros((210,160))
    # S_jet[35:195,:] = S
    S_jet[crops[0]:crops[1] + 160,:] = S
    S_jet = S_jet.clip(1,255)
    # S_jet = cv2.cvtColor(cv2.applyColorMap(S_jet.astype('uint8'), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    S_jet = cv2.applyColorMap(S_jet.astype('uint8'), cv2.COLORMAP_JET)
    S_jet = cv2.addWeighted(I, 0.4, S_jet, 0.6, 0.0)
    # I[35:195,:] = S_jet
    I = S_jet
    I = I.clip(1,255).astype('uint8')

    # I[35:195,:,channel] += S.astype('uint16')
    # I = I.clip(1,255).astype('uint8')
    return I

def saliency_on_atari_frame_def(saliency, atari, crops, fudge_factor, channel=2, sigma=0):
    S = saliency
    I = atari.astype('uint16')
    # I[35:195,:,channel] += S.astype('uint16')
    I[crops[0]:crops[1] + 160,:,channel] += S.astype('uint16')
    I = I.clip(1,255).astype('uint8')
    return I



def min_max_normalization(l, l_max, l_min):
    # l_min = l.min()
    # l_max = l.max()
    return np.array([(i - l_min) / (l_max - l_min) for i in l])