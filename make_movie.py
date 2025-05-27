import os
from tqdm import tqdm
import cv2
import numpy as np
import csv
import argparse

import warnings
warnings.simplefilter('ignore')

"""
Code to convert attention map or raw image to video
"""

# Example code

# python make_movie.py --game game_name --load-dir folder_path --mode raw_or_encoder_or_decoder


parser = argparse.ArgumentParser(description='Make movie')
parser.add_argument('--mode', type=str, default='raw', choices=['raw', 'encoder', 'decoder', 'encoder_sal', 'decoder_sal', ], metavar='CUDA', help='Cuda Device')
parser.add_argument('--load-dir', type=str, default='visuals/breakout_rainbow_aqt/epi1/', help='Load data')
parser.add_argument('--game', type=str, default='pong', help='env name')

# Setup
args = parser.parse_args()

# Print options
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))



print("Make movie: {}".format(args.mode))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
if args.mode == "raw":
  movie_path = os.path.join(args.load_dir, "raw_movie.mp4")
  video = cv2.VideoWriter(movie_path, fourcc, 10.0, (160, 210))

  file_num = sum(os.path.isfile(os.path.join(args.load_dir + "raw_img/", name)) for name in os.listdir(args.load_dir + "raw_img/"))
  for idx in tqdm(range(file_num)):
    raw_img = cv2.imread(args.load_dir + "raw_img/raw_{0:06d}.png".format(idx))
    video.write(raw_img)
  video.release()

elif args.mode == "encoder":
  movie_path = os.path.join(args.load_dir, "encoder_movie.mp4")
  video = cv2.VideoWriter(movie_path, fourcc, 10.0, (200, 200))

  file_num = sum(os.path.isfile(os.path.join(args.load_dir + "encoder/", name)) for name in os.listdir(args.load_dir + "encoder/"))
  for idx in tqdm(range(file_num)):
    en_att = cv2.imread(args.load_dir + "encoder/en_{0:06d}.png".format(idx))
    video.write(en_att)
  video.release()

elif args.mode == "decoder":
  label_img = np.ones((220,5,3)) * 125
  cv2.imwrite("./mv_img.png", label_img)

  if not os.path.exists(os.path.join(args.load_dir, "movie_act_img")):
    os.makedirs(os.path.join(args.load_dir, "movie_act_img"))

  if args.game == "pong":
    max_act_n = 6
  elif args.game == "breakout":
    max_act_n = 4
  elif args.game == "mspacman":
    max_act_n = 9
  elif args.game == "spaceinvaders":
    max_act_n = 6
  elif args.game == "seaquest":
    max_act_n = 18
  elif args.game == "kungfu":
    max_act_n = 14
  elif args.game == "fishingderby":
    max_act_n = 18
  elif args.game == "bowling":
    max_act_n = 6
  elif args.game == "beamrider":
    max_act_n = 9
  elif args.game == "boxing":
    max_act_n = 18
  elif args.game == "freeway":
    max_act_n = 3
  elif args.game == "choppercommand":
    max_act_n = 18
  
  actions = [i for i in range(1, max_act_n)]
  # actions = [3, 6, 11, 14, 4, 7, 12, 15]
  # actions = [2, 3, 4, 5]
  act_n = len(actions) + 1

  movie_path = os.path.join(args.load_dir, "decoder_movie.mp4")
  video = cv2.VideoWriter(movie_path, fourcc, 10.0, (200*act_n + 5*(act_n-1), 220))

  file_num = sum(os.path.isfile(os.path.join(args.load_dir + "decoder_act0/", name)) for name in os.listdir(args.load_dir + "decoder_act0/"))
  mv_img = cv2.imread("./mv_img.png")

  for idx in tqdm(range(file_num)):
    att_img = cv2.imread(args.load_dir + "decoder_act0/de0-{0:06d}.png".format(idx))
    for act in actions:
      att_img = cv2.hconcat([att_img, mv_img])
      act_img = cv2.imread(args.load_dir + "decoder_act{0}/de{1}-{2:06d}.png".format(act,act,idx))
      att_img = cv2.hconcat([att_img, act_img])
    cv2.imwrite(args.load_dir + "movie_act_img/act-{0:06d}.png".format(idx), att_img)
    #print(att_img.shape)
    video.write(att_img)
  video.release()


elif args.mode == "encoder_sal":
  movie_path = os.path.join(args.load_dir, "encoder_sal_movie.mp4")
  video = cv2.VideoWriter(movie_path, fourcc, 10.0, (200, 200))

  file_num = sum(os.path.isfile(os.path.join(args.load_dir + "encoder_sal/", name)) for name in os.listdir(args.load_dir + "encoder_sal/"))
  for idx in tqdm(range(file_num)):
    en_att = cv2.imread(args.load_dir + "encoder_sal/en_{0:06d}.png".format(idx))
    video.write(en_att)
  video.release()


elif args.mode == "decoder_sal":
  label_img = np.ones((220,5,3)) * 125
  cv2.imwrite("./mv_img.png", label_img)

  if not os.path.exists(os.path.join(args.load_dir, "movie_act_sal")):
    os.makedirs(os.path.join(args.load_dir, "movie_act_sal"))

  if args.game == "pong":
    max_act_n = 6
  elif args.game == "breakout":
    max_act_n = 4
  elif args.game == "mspacman":
    max_act_n = 9
  elif args.game == "spaceinvaders":
    max_act_n = 6
  elif args.game == "seaquest":
    max_act_n = 18
  elif args.game == "kungfu":
    max_act_n = 14
  elif args.game == "fishingderby":
    max_act_n = 18
  elif args.game == "bowling":
    max_act_n = 6
  elif args.game == "beamrider":
    max_act_n = 9
  elif args.game == "boxing":
    max_act_n = 18
  elif args.game == "freeway":
    max_act_n = 3
  elif args.game == "choppercommand":
    max_act_n = 18
  
  actions = [i for i in range(1, max_act_n)]
  # actions = [3, 6, 11, 14, 4, 7, 12, 15]
  # actions = [2, 3, 4, 5]
  act_n = len(actions) + 1

  movie_path = os.path.join(args.load_dir, "decoder_sal_movie.mp4")
  video = cv2.VideoWriter(movie_path, fourcc, 10.0, (200*act_n + 5*(act_n-1), 220))

  file_num = sum(os.path.isfile(os.path.join(args.load_dir + "decoder_sal0/", name)) for name in os.listdir(args.load_dir + "decoder_sal0/"))
  mv_img = cv2.imread("./mv_img.png")

  for idx in tqdm(range(file_num)):
    att_img = cv2.imread(args.load_dir + "decoder_sal0/de0-{0:06d}.png".format(idx))
    for act in actions:
      att_img = cv2.hconcat([att_img, mv_img])
      act_img = cv2.imread(args.load_dir + "decoder_sal{0}/de{1}-{2:06d}.png".format(act,act,idx))
      att_img = cv2.hconcat([att_img, act_img])
    cv2.imwrite(args.load_dir + "movie_act_sal/act-{0:06d}.png".format(idx), att_img)
    #print(att_img.shape)
    video.write(att_img)
  video.release()