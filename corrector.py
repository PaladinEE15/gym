#generate dataset
import gym
import numpy as np
from stable_baselines import SAC

#we should get normal transitions, and add noise as samples
#pendulum obs: cos(theta), sin(theta), thetadot, action: 1

#config page
total_episodes = 100000
ac_dim = 1
ob_dim = 3
x_dim = 2*ob_dim+ac_dim
#config page end

env = gym.make('NoisyPendulum-v0')
model=SAC.load("sac_pendulum")
obs = env.reset()
acvariant = np.random.rand(total_episodes)
cor_dataset_xraw = np.zeros((total_episodes,x_dim))
cor_dataset_y = np.zeros((total_episodes,ob_dim))

subcount = 0
for dd in range(total_episodes):
    action, states = model.predict(obs)
    action = action + acvariant[dd]
    cor_dataset_xraw[dd][0:ob_dim] = obs
    cor_dataset_xraw[dd][ob_dim:ob_dim+ac_dim] = action
    obs, rewards, dones, info = env.step(action)
    cor_dataset_xraw[dd][ob_dim+ac_dim:2*ob_dim+ac_dim] = obs
    cor_dataset_y[dd][:] = obs

    subcount = subcount + 1
    if subcount >= 100:
        subcount = 0
        obs = env.reset()

noiseangle = np.random.rand(1000)-0.5
noisespeed = 0.5*np.random.rand(1000)-0.25
Xraw = cor_dataset_xraw
ac_dim = 1
ob_dim = 3
x_dim = 2*ob_dim+ac_dim
X = np.zeros((total_episodes,x_dim))

for outer in range(total_episodes/100):
    cosb = np.cos(noiseangle[outer])
    sinb = np.sin(noiseangle[outer])
    bias = noisespeed[outer]
    for inner in range(100):
        count = outer*100 + inner
        X[count][0] = Xraw[count][0]*cosb - Xraw[count][1]*sinb
        X[count][1] = Xraw[count][1]*cosb + Xraw[count][0]*sinb
        X[count][2] = Xraw[count][2] + bias
        X[count][4] = Xraw[count][4]*cosb - Xraw[count][5]*sinb
        X[count][5] = Xraw[count][5]*cosb + Xraw[count][4]*sinb
        X[count][6] = Xraw[count][6] + bias


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

input_size = x_dim
output_size = ob_dim
hidden_size = [64,64]

x_train = X
y_train = cor_dataset_y

model = Sequential()
model.add(Dense(hidden_size[0], input_dim = input_size, activation = 'relu'))
model.add(Dense(hidden_size[1], activation = 'relu'))
model.add(Dense(output_size,activation = None))
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train,y_train,epochs = 500, batch_size = 128)

env = gym.make('NoisyPendulum-v0')
dcs_model=SAC.load("sac_pendulum")
obs_fix = env.reset()
obs_raw = obs_fix
env.set_noise(0,0.18)
sub_reward_fix=[]
for dd in range(400):
    action, _states = dcs_model.predict(obs_fix)
    obac = np.append(obs_raw,action)
    obs_raw, rewards, dones, info = env.step(action)
    inputx = np.append(obac, obs_raw)
    inputx = inputx.reshape((1,7))
    obs_fix = model.predict(inputx)
    sub_reward_fix.append(rewards)

sub_reward_raw=[]
obs = env.reset()
for dd in range(400):
    action, _states = dcs_model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    sub_reward_raw.append(rewards)

import matplotlib.pyplot as plt

plt.plot(range(400),np.array(sub_reward_raw))
plt.plot(range(400),np.array(sub_reward_fix))
plt.show()
