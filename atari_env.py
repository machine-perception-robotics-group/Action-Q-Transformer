import numpy as np
import gym
import gym.spaces

import cv2
import collections


"""
Classes and functions for atari environment
"""

def make_env(env_name, seed, visual=False):
    env = gym.make(env_name, frameskip=1, repeat_action_probability=0.0, full_action_space=False)
    env.seed(seed)
    env.action_space.seed(seed)
    
    env = ClipRewardEnv(env)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(visual, env)
    env = ImageToPyTorch(visual, env)
    env = BufferWrapper(visual, env, 4)
    env = ScaledFloatFrame(visual, env)
    return env

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, visual, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.visual_flag = visual
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        if self.visual_flag:
            #return ProcessFrame84.process(obs), obs[35:210-15, :, :, np.newaxis]
            return ProcessFrame84.process(obs), obs[:, :, :, np.newaxis]
        else:
            return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        x_t = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, visual, env):
        super(ImageToPyTorch, self).__init__(env)
        self.visual_flag = visual
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        if self.visual_flag:
            observation, raw_img = observation
            return np.moveaxis(observation, 2, 0), np.moveaxis(raw_img, 3, 0)
        else:
            return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, visual, env):
        super(ScaledFloatFrame, self).__init__(env)
        self.visual_flag = visual
        
    def observation(self, obs):
        if self.visual_flag:
            obs, raw_img = obs
            return np.array(obs).astype(np.float32) / 255.0, np.array(raw_img)
        else:
            return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, visual, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.visual_flag = visual
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)
        
        self.raw_obs_space = np.zeros((4, 210, 160, 3))

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        self.raw_buffer = np.zeros_like(self.raw_obs_space, dtype=np.uint8)
        return self.observation(self.env.reset())

    def observation(self, observation):
        if self.visual_flag:
            observation, raw_img = observation
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = observation
            self.raw_buffer[:-1] = self.raw_buffer[1:]
            self.raw_buffer[-1] = raw_img
            return self.buffer, self.raw_buffer
        else:
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = observation
            return self.buffer


