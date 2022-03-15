import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from copy import deepcopy
from joblib import Parallel, delayed
from IPython.display import clear_output
from IPython import display
from copy import deepcopy
import gym
import joblib
from joblib import Parallel, delayed
from six import BytesIO
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY")) == 0:
    os.system("bash ../xvfb start")
    os.environ['DISPLAY'] = ':1'

tmp_env = gym.make("CartPole-v0")
tmp_env.reset()
n_states = tmp_env.observation_space.shape[0]
n_actions = tmp_env.action_space.n

def display_session(env, agent, t_max=500):
    total_reward = 0
    plt.figure(figsize=(4, 3))
    display.clear_output(wait=True)

    s = env.reset()
    
    for t in range(t_max):
        plt.gca().clear()
        
        a = agent.get_action(torch.tensor(s).float())[0]
        new_s, r, done, info = env.step(a)
        s = new_s
        total_reward += r
        # Draw game image on display.
        plt.imshow(env.render('rgb_array'))

        display.display(plt.gcf())
        display.clear_output(wait=True)
        
        if done:
            break
            
    return total_reward

def generate_session(env, agent, t_max=500):
    total_reward = 0
    s = env.reset()
    
    for t in range(t_max):
        a = agent.get_action(torch.tensor(s).float())[0]
        new_s, r, done, info = env.step(a)
        total_reward += r
        s = new_s
        
        if done:
            break
            
    return total_reward

def add_noise_to_model(model, noise, copy=False):
    if copy:
        model = deepcopy(model)
    
    new_weight = model.fc.weight.detach() + noise.T
    model.fc.weight = nn.Parameter(new_weight.float())
    return model

class MLPPolicy(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.fc = nn.Linear(n_states, n_actions)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        out = self.fc(x)
        out = self.softmax(out)
        return out
    
    def get_action(self, states):
        probs = self(states)
        probs = probs.detach().numpy()
        action = np.random.choice(np.arange(n_actions), size = 1, p = probs)
        return action

def get_env_function():
    env = gym.make('CartPole-v0').env
    return env

class EvolutionManagerParallel:
    def __init__(self, get_env_function, lr=0.001, std=0.01, n_samples = 64, normalize=True):
        super().__init__()
        
        self.lr = lr
        self.std = std
        self.normalize = normalize
        self.n_samples = n_samples
        self.mean_reward_history = []
        
        self.env = get_env_function()
        
    def get_noised_model(self, model, noise):
        return add_noise_to_model(model, noise, True)
    
    def get_reward(self, model, noise):
        noised_model = self.get_noised_model(model, noise)
        reward = generate_session(self.env, noised_model)
        return reward
    
    @torch.no_grad()
    def optimize(self, model, noises):
        rewards = np.array(
            Parallel(n_jobs=4)
            (delayed(self.get_reward)(model, self.std * noise) for noise in noises)
        )
        
        A = (rewards - np.mean(rewards)) / np.std(rewards)
        new_noise = self.lr/(self.n_samples*self.std) * np.dot(noises.T, A)
        new_model = add_noise_to_model(model, new_noise.T)
        return new_model, rewards
    
    def step(self, model):
        noises = np.random.randn(self.n_samples, n_states, n_actions)  
        new_model, rewards = self.optimize(model, noises)
        self.update_log(rewards)
        
    def update_log(self, rewards):
        mean_reward = np.mean(rewards)
        # self.mean_reward_history.append(mean_reward)

        # clear_output(True)
        print("last mean reward = %.3f" % mean_reward)
        # plt.figure(figsize=[8, 4])
        # plt.subplot(1, 2, 1)
        # plt.plot(self.mean_reward_history, label='Mean rewards')
        # plt.legend()
        # plt.grid()

        # plt.subplot(1, 2, 2)
        # plt.hist(rewards)
        # plt.grid()

        # plt.show()


if __name__ == '__main__':
    model = MLPPolicy(n_states, n_actions)
    algorithm = EvolutionManagerParallel(get_env_function, lr=0.01, std=0.1, n_samples = 64)

    t = time.time()
    for i in range(1000):
        algorithm.step(model)
        
    print(time.time() - t)