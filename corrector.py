#generate dataset
import gym
import numpy as np
from stable_baselines import SAC

#we should get normal transitions, and add noise as samples
#pendulum obs: cos(theta), sin(theta), thetadot, action: 1

#config page
total_episodes = 10000
ac_dim = 1
ob_dim = 3
x_dim = 2*ob_dim+ac_dim
#config page end

env = gym.make('NoisyPendulum-v0')
model=SAC.load("sac_pendulum")
obs = env.reset()
acvariant = np.rand(total_episodes)
cor_dataset_xraw = np.zeros((total_episodes,x_dim))
cor_dataset_y = np.zeros((total_episodes,ob_dim))

subcount = 0
for dd in range(total_episodes):
    action, states = model.predict(obs)
    action = action + acvariant[dd]
    action = np.clip(action, -2, 2)[0]
    cor_dataset_xraw[dd][0:ob_dim] = obs
    cor_dataset_xraw[dd][ob_dim:ob_dim+ac_dim] = action
    obs, rewards, dones, info = env.step(action)
    cor_dataset_xraw[dd][ob_dim+ac_dim:2*ob_dim+ac_dim] = obs
    cor_dataset_y[dd][:] = obs

    subcount = subcount + 1
    if subcount >= 100:
        subcount = 0
        obs = env.reset()

np.save("cor_x_raw.npy",cor_dataset_xraw)
np.save("cor_y.npy",cor_dataset_y)

from winsound import Beep
Beep(3000, 500)