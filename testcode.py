#test well-trained model performance in common scenerio and noisy scenerio 
import gym
import numpy as np
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import SAC, DDPG

acn=[0,0,0,0,0]
obn=[0,0.1,0.15,0.2,0.3]

env = gym.make('NoisyPendulum-v0')
model=SAC.load("sac_pendulum")
obs = env.reset()

SACpendu_reward=[]
for dd in range(5):
    env.set_noise(acn[dd],obn[dd])
    obs = env.reset()
    sub_reward=[]
    for dd in range(400):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        sub_reward.append(rewards)
    SACpendu_reward.append(sub_reward)
SACpendu_total=np.array(SACpendu_reward)

del model

model=DDPG.load("ddpg_pendulum")
obs = env.reset()

DDPGpendu_reward=[]
for dd in range(5):
    env.set_noise(acn[dd],obn[dd])
    obs = env.reset()
    sub_reward=[]
    for dd in range(200):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        sub_reward.append(rewards)
    DDPGpendu_reward.append(sub_reward)
DDPGpendu_total=np.array(SACpendu_reward)

print('finish!')

#draw section
import matplotlib.pyplot as plt
fig=plt.figure()
fig.suptitle("SAC vs noisy observation")

sbp1=plt.subplot(5,1,1)
plt.plot(range(400),SACpendu_total[0,:])
sbp1.set_title("observe noise=0.0")

sbp2=plt.subplot(5,1,2)
plt.plot(range(400),SACpendu_total[1,:])
sbp2.set_title("observe noise=0.1")

sbp3=plt.subplot(5,1,3)
plt.plot(range(400),SACpendu_total[2,:])
sbp3.set_title("observe noise=0.15")

sbp4=plt.subplot(5,1,4)
plt.plot(range(400),SACpendu_total[3,:])
sbp4.set_title("observe noise=0.2")

sbp5=plt.subplot(5,1,5)
plt.plot(range(400),SACpendu_total[4,:])
sbp5.set_title("observe noise=0.3")

plt.subplots_adjust(hspace=3)
plt.show()