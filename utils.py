import csv
import os
import cv2
import torch
import numpy as np

"""
Other functions and class (logger, make attention map, make saliency map, etc.)
"""

class REWARD_LOGGER:
    def __init__(self, path):
        self.save_path = path
        self.train_record_reward = []
        self.train_record_step = []
        self.test_record_reward = []
        self.test_record_step = []
        
        header = ['Step', 'Reward', 'Mean Reward', 'Episode']
        if not os.path.exists(self.save_path+'train_log.csv'):
            with open(self.save_path + 'train_log.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
        
        header = ['Step', 'Mean Reward', 'Episode']
        if not os.path.exists(self.save_path + 'test_log.csv'):
            with open(self.save_path + 'test_log.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
    
    def record_train_log(self, step, reward, mean_reward, epi):
        self.train_record_reward.append(reward)
        self.train_record_step.append(step)
        with open(self.save_path+'train_log.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([step, reward, mean_reward, epi])
        
    def record_eval_log(self, step, mean_reward, epi):
        self.test_record_reward.append(mean_reward)
        self.test_record_step.append(step)
        with open(self.save_path+'test_log.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([step, mean_reward, epi])


class TTQ_REWARD_LOGGER:
    def __init__(self, path):
        self.save_path = path
        self.train_record_reward = []
        self.train_record_step = []
        self.test_record_reward = []
        self.test_record_step = []
        
        header = ['Step', 'Reward', 'Mean Reward', 'Episode', 'Alpha']
        if not os.path.exists(self.save_path+'train_log.csv'):
            with open(self.save_path + 'train_log.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
        
        header = ['Step', 'Mean Reward', 'Episode', 'Alpha']
        if not os.path.exists(self.save_path + 'test_log.csv'):
            with open(self.save_path + 'test_log.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
    
    def record_train_log(self, step, reward, mean_reward, epi, alpha):
        self.train_record_reward.append(reward)
        self.train_record_step.append(step)
        with open(self.save_path+'train_log.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([step, reward, mean_reward, epi, alpha])
        
    def record_eval_log(self, step, mean_reward, epi, alpha):
        self.test_record_reward.append(mean_reward)
        self.test_record_step.append(step)
        with open(self.save_path+'test_log.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([step, mean_reward, epi, alpha])



def min_max(x, mins, maxs, axis=None):
    result = (x - mins)/(maxs - mins)
    return result


# Normalization of encoder-attention
def make_en_attention(args, attns):
    patch_size = 7
    reshaped_attns = attns[0].view((patch_size * patch_size, patch_size, patch_size)) # rainbow aqt
    reshaped_attns = torch.mean(reshaped_attns, axis=0)

    if args.game == "breakout":
        mask_att = torch.ones(patch_size, patch_size, device=args.device)
        for x in range(patch_size):
            mask_att[0][x] = 0
    elif args.game == "ms_pacman":
        mask_att = torch.ones(patch_size, patch_size, device=args.device)
        for x in range(patch_size):
            mask_att[patch_size-1][x] = 0
        reshaped_attns = reshaped_attns * mask_att

    return reshaped_attns.cpu().detach().numpy()

# Imaging encoder-attention (overlaid with raw img)
def make_en_img(attns, raw_img, step, epi_dir, mode="normal"):
    if mode != "mean":
        raw_img = cv2.imread(epi_dir + "/raw_img/raw_{0:06d}.png".format(step))
        mask = cv2.resize(attns, dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_NEAREST) # resize of attention map
        #mask = cv2.resize(attns, dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET) # jet mapping
        #mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_CIVIDIS)
        masked_img = cv2.addWeighted(raw_img, 0.4, mask, 0.6, 0)
        masked_img = cv2.resize(masked_img, dsize=(200, 200))
        cv2.imwrite(epi_dir + "/encoder/en_{0:06d}.png".format(step), masked_img)
    else:
        raw_img = cv2.imread(epi_dir + "/raw_img/raw_{0:06d}.png".format(step))
        mask = cv2.resize(attns, dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_NEAREST) # resize of attention map
        #mask = cv2.resize(attns, dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET) # jet mapping
        #mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_CIVIDIS) 
        masked_img = cv2.addWeighted(raw_img, 0.4, mask, 0.6, 0)
        masked_img = cv2.resize(masked_img, dsize=(200, 200))
        cv2.imwrite(epi_dir + "/en_mean.png", masked_img)
    return

# Imaging encoder-attention
def make_en_attimg(attns, raw_img, step, epi_dir, mode="normal"):
    if mode != "mean":
        raw_img = cv2.imread(epi_dir + "/raw_img/raw_{0:06d}.png".format(step))
        mask = cv2.resize(attns, dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_NEAREST) # resize of attention map
        #mask = cv2.resize(attns, dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        masked_img = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET) # jet mapping
        #mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_CIVIDIS)
        masked_img = cv2.resize(masked_img, dsize=(200, 200))
        cv2.imwrite(epi_dir + "/encoder_att/en_{0:06d}.png".format(step), masked_img)
    else:
        raw_img = cv2.imread(epi_dir + "/raw_img/raw_{0:06d}.png".format(step))
        mask = cv2.resize(attns, dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_NEAREST) # resize of attention map
        #mask = cv2.resize(attns, dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        masked_img = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET) # jet mapping
        #mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_CIVIDIS)
        masked_img = cv2.resize(masked_img, dsize=(200, 200))
        cv2.imwrite(epi_dir + "/en_mean.png", masked_img)
    return


# Normalization of decoder-attention
def make_de_attention(args, attns, action_num):
    patch_size = 7

    action_attns = []
    for action in range(action_num):
        ac_attn = attns[0, action].view(patch_size, patch_size)
        
        # normalizarion
        if args.game == "breakout":
            mask_att = torch.ones(patch_size, patch_size, device=args.device)
            for x in range(patch_size):
                mask_att[0][x] = 0
            ac_attn = ac_attn * mask_att
        elif args.game == "ms_pacman":
            mask_att = torch.ones(patch_size, patch_size, device=args.device)
            for x in range(patch_size):
                mask_att[patch_size-1][x] = 0
            ac_attn = ac_attn * mask_att

        action_attns.append(ac_attn.cpu().detach().numpy())
    return action_attns

# Imaging decoder-attention (overlaid with raw img)
def make_de_img(args, q, attns, step, action_num, epi_dir):
    raw_img = cv2.imread(epi_dir + "/raw_img/raw_{0:06d}.png".format(step))

    max_q = q.argmax().item()

    for action in range(action_num):
        if max_q == action:
            txt_color = (0,0,255)
        else:
            txt_color = (0,0,0)

        mask = cv2.resize(attns[action], dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_NEAREST) # resize of attention map
        #mask = cv2.resize(attns[action], dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET) # jet mapping

        masked_img = cv2.addWeighted(raw_img, 0.4, mask, 0.6, 0)
        masked_img = cv2.resize(masked_img, dsize=(200, 200))

        # Q-value action label
        label_img = cv2.imread("./label_img.png")
        action_name = ac_name_search(args.game, action)
        cv2.putText(label_img, text='{0} Q:{1:.3f}'.format(action_name, q[action]), org=(10,15), fontScale=0.5, 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=txt_color, thickness=1, lineType=cv2.LINE_4)
        masked_img = cv2.vconcat([masked_img, label_img])

        cv2.imwrite(epi_dir + "/decoder_act{0}/de{1}-{2:06d}.png".format(action, action, step), masked_img)
    return 

# Imaging decoder-attention
def make_de_attimg(args, q, attns, step, action_num, epi_dir):
    raw_img = cv2.imread(epi_dir + "/raw_img/raw_{0:06d}.png".format(step))

    max_q = q.argmax().item()

    for action in range(action_num):
        if max_q == action:
            txt_color = (0,0,255)
        else:
            txt_color = (0,0,0)

        mask = cv2.resize(attns[action], dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_NEAREST) # resize of attention map
        #mask = cv2.resize(attns[action], dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        masked_img = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET) # jet mapping
        masked_img = cv2.resize(masked_img, dsize=(200, 200))

        # Q-value action label
        label_img = cv2.imread("./label_img.png")
        action_name = ac_name_search(args.game, action)
        cv2.putText(label_img, text='{0} Q:{1:.3f}'.format(action_name, q[action]), org=(10,15), fontScale=0.5, 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=txt_color, thickness=1, lineType=cv2.LINE_4)
        masked_img = cv2.vconcat([masked_img, label_img])

        cv2.imwrite(epi_dir + "/decoder_att_act{0}/de{1}-{2:06d}.png".format(action, action, step), masked_img)
    return 




# Conversion from action index to action name
def ac_name_search(env_name, action):
    action_name = None
    if env_name == "pong":
        if action == 0: action_name = "NOOP"
        elif action == 1: action_name = "NOOP"
        elif action == 2: action_name = "UP"
        elif action == 3: action_name = "DOWN"
        elif action == 4: action_name = "UP"
        elif action == 5: action_name = "DOWN"
    elif env_name == "breakout":
        if action == 0: action_name = "NOOP"
        elif action == 1: action_name = "NOOP"
        elif action == 2: action_name = "RIGHT"
        elif action == 3: action_name = "LEFT"
    elif env_name == "ms_pacman":
        if action == 0: action_name = "NOOP"
        elif action == 1: action_name = "UP"
        elif action == 2: action_name = "RIGHT"
        elif action == 3: action_name = "LEFT"
        elif action == 4: action_name = "DOWN"
        elif action == 5: action_name = "UPRIGHT"
        elif action == 6: action_name = "UPLEFT"
        elif action == 7: action_name = "DOWNRIGHT"
        elif action == 8: action_name = "DOWNLEFT"
    elif env_name == "space_invaders":
        if action == 0: action_name = "NOOP"
        elif action == 1: action_name = "FIRE"
        elif action == 2: action_name = "RIGHT"
        elif action == 3: action_name = "LEFT"
        elif action == 4: action_name = "RIGHTFIRE"
        elif action == 5: action_name = "LEFTFIRE"
    elif env_name == "seaquest":
        if action == 0: action_name = "NOOP"
        elif action == 1: action_name = "FIRE"
        elif action == 2: action_name = "UP"
        elif action == 3: action_name = "RIGHT"
        elif action == 4: action_name = "LEFT"
        elif action == 5: action_name = "DOWN"
        elif action == 6: action_name = "UPRIGHT"
        elif action == 7: action_name = "UPLEFT"
        elif action == 8: action_name = "DOWNRIGHT"
        elif action == 9: action_name = "DOWNLEFT"
        elif action == 10: action_name = "UPFIRE"
        elif action == 11: action_name = "RIGHTFIRE"
        elif action == 12: action_name = "LEFTFIRE"
        elif action == 13: action_name = "DOWNFIRE"
        elif action == 14: action_name = "UPRIGHTFIRE"
        elif action == 15: action_name = "UPLEFTFIRE"
        elif action == 16: action_name = "DOWNRIGHTFIRE"
        elif action == 17: action_name = "DOWNLEFTFIRE"
    elif env_name == "kung_fu_master":
        if action == 0: action_name = "NOOP"
        elif action == 1: action_name = "UP"
        elif action == 2: action_name = "RIGHT"
        elif action == 3: action_name = "LEFT"
        elif action == 4: action_name = "DOWN"
        elif action == 5: action_name = "DOWNRIGHT"
        elif action == 6: action_name = "DOWNLEFT"
        elif action == 7: action_name = "RIGHTFIRE"
        elif action == 8: action_name = "LEFTFIRE"
        elif action == 9: action_name = "DOWNFIRE"
        elif action == 10: action_name = "UPRIGHTFIRE"
        elif action == 11: action_name = "UPLEFTFIRE"
        elif action == 12: action_name = "DOWNRIGHTFIRE"
        elif action == 13: action_name = "DOWNLEFTFIRE"
    elif env_name == "fishing_derby":
        if action == 0: action_name = "NOOP"
        elif action == 1: action_name = "FIRE"
        elif action == 2: action_name = "UP"
        elif action == 3: action_name = "RIGHT"
        elif action == 4: action_name = "LEFT"
        elif action == 5: action_name = "DOWN"
        elif action == 6: action_name = "UPRIGHT"
        elif action == 7: action_name = "UPLEFT"
        elif action == 8: action_name = "DOWNRIGHT"
        elif action == 9: action_name = "DOWNLEFT"
        elif action == 10: action_name = "UPFIRE"
        elif action == 11: action_name = "RIGHTFIRE"
        elif action == 12: action_name = "LEFTFIRE"
        elif action == 13: action_name = "DOWNFIRE"
        elif action == 14: action_name = "UPRIGHTFIRE"
        elif action == 15: action_name = "UPLEFTFIRE"
        elif action == 16: action_name = "DOWNRIGHTFIRE"
        elif action == 17: action_name = "DOWNLEFTFIRE"
    elif env_name == "bowling":
        if action == 0: action_name = "NOOP"
        elif action == 1: action_name = "FIRE"
        elif action == 2: action_name = "UP"
        elif action == 3: action_name = "DOWN"
        elif action == 4: action_name = "UPFIRE"
        elif action == 5: action_name = "DOWNFIRE"
    elif env_name == "beam_rider":
        if action == 0: action_name = "NOOP"
        elif action == 1: action_name = "FIRE"
        elif action == 2: action_name = "UP"
        elif action == 3: action_name = "RIGHT"
        elif action == 4: action_name = "LEFT"
        elif action == 5: action_name = "UPRIGHT"
        elif action == 6: action_name = "UPLEFT"
        elif action == 7: action_name = "RIGHTFIRE"
        elif action == 8: action_name = "LEFTFIRE"
    elif env_name == "boxing":
        if action == 0: action_name = "NOOP"
        elif action == 1: action_name = "FIRE"
        elif action == 2: action_name = "UP"
        elif action == 3: action_name = "RIGHT"
        elif action == 4: action_name = "LEFT"
        elif action == 5: action_name = "DOWN"
        elif action == 6: action_name = "UPRIGHT"
        elif action == 7: action_name = "UPLEFT"
        elif action == 8: action_name = "DOWNRIGHT"
        elif action == 9: action_name = "DOWNLEFT"
        elif action == 10: action_name = "UPFIRE"
        elif action == 11: action_name = "RIGHTFIRE"
        elif action == 12: action_name = "LEFTFIRE"
        elif action == 13: action_name = "DOWNFIRE"
        elif action == 14: action_name = "UPRIGHTFIRE"
        elif action == 15: action_name = "UPLEFTFIRE"
        elif action == 16: action_name = "DOWNRIGHTFIRE"
        elif action == 17: action_name = "DOWNLEFTFIRE"
    elif env_name == "freeway":
        if action == 0: action_name = "NOOP"
        elif action == 1: action_name = "UP"
        elif action == 2: action_name = "DOWN"
    elif env_name == "chopper_command":
        if action == 0: action_name = "NOOP"
        elif action == 1: action_name = "FIRE"
        elif action == 2: action_name = "UP"
        elif action == 3: action_name = "RIGHT"
        elif action == 4: action_name = "LEFT"
        elif action == 5: action_name = "DOWN"
        elif action == 6: action_name = "UPRIGHT"
        elif action == 7: action_name = "UPLEFT"
        elif action == 8: action_name = "DOWNRIGHT"
        elif action == 9: action_name = "DOWNLEFT"
        elif action == 10: action_name = "UPFIRE"
        elif action == 11: action_name = "RIGHTFIRE"
        elif action == 12: action_name = "LEFTFIRE"
        elif action == 13: action_name = "DOWNFIRE"
        elif action == 14: action_name = "UPRIGHTFIRE"
        elif action == 15: action_name = "UPLEFTFIRE"
        elif action == 16: action_name = "DOWNRIGHTFIRE"
        elif action == 17: action_name = "DOWNLEFTFIRE"
    return action_name



# Imaging encoder-saliency (overlaid with raw img)
def make_en_sal(sal, step, epi_dir, mode="normal"):
    if mode != "mean":
        raw_img = cv2.imread(epi_dir + "/raw_img/raw_{0:06d}.png".format(step))
        mask = cv2.resize(sal, dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_NEAREST) # resize of attention map
        #mask = cv2.resize(sal, dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        #mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_CIVIDIS)
        mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET) # jet mapping
        masked_img = cv2.addWeighted(raw_img, 0.4, mask, 0.6, 0)
        masked_img = cv2.resize(masked_img, dsize=(200, 200))
        cv2.imwrite(epi_dir + "/encoder_sal/en_{0:06d}.png".format(step), masked_img)
    else:
        raw_img = cv2.imread(epi_dir + "/raw_img/raw_{0:06d}.png".format(step))
        mask = cv2.resize(sal, dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_NEAREST) # resize of attention map
        #mask = cv2.resize(sal, dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET) # jet mapping
        #mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_CIVIDIS)
        masked_img = cv2.addWeighted(raw_img, 0.4, mask, 0.6, 0)
        masked_img = cv2.resize(masked_img, dsize=(200, 200))
        cv2.imwrite(epi_dir + "/encoder_sal/en_{0:06d}.png".format(step), masked_img)
    return

# Imaging decoder-saliency (overlaid with raw img)
def make_de_sal(args, q, sals, step, action_num, epi_dir):
    raw_img = cv2.imread(epi_dir + "/raw_img/raw_{0:06d}.png".format(step))

    max_q = q.argmax().item()

    for action in range(action_num):
        if max_q == action:
            txt_color = (0,0,255)
        else:
            txt_color = (0,0,0)

        mask = cv2.resize(sals[action], dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_NEAREST) # resize of attention map
        #mask = cv2.resize(attns[action], dsize=(raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET) # jet mapping

        masked_img = cv2.addWeighted(raw_img, 0.4, mask, 0.6, 0)
        masked_img = cv2.resize(masked_img, dsize=(200, 200))

        # Q-value action label
        label_img = cv2.imread("./label_img.png")
        action_name = ac_name_search(args.game, action)
        cv2.putText(label_img, text='{0} Q:{1:.3f}'.format(action_name, q[action]), org=(10,15), fontScale=0.5, 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=txt_color, thickness=1, lineType=cv2.LINE_4)
        masked_img = cv2.vconcat([masked_img, label_img])

        cv2.imwrite(epi_dir + "/decoder_sal{0}/de{1}-{2:06d}.png".format(action, action, step), masked_img)
    return 