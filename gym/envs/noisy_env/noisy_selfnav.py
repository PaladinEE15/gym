import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class NoisySelfnav(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    #action: turn left or right
    #observe: 7 distances, speed direction, target distance, target direction, x, y 
    #direction is [-1,1],corresbonding to [-pi,pi]

    def __init__(self):
        
        
        self.low_state=np.array([0,0,0,0,0,0,0,-1,0,0,0],dtype=np.float32)
        





    


def angle_norm(x):
    if x>1:
        x=x-2
    if x<-1:
        x=x+2
    return x

def getdistance(x,y,direc):#distance without normalize.
    delta_x=0.1*np.cos(direc)
    delta_y=0.1*np.sin(direc)
    cur_x=x
    cur_y=y
    for ccount in range(200):
        cur_x=cur_x+delta_x
        cur_y=cur_y+delta_y
        if cur_y>=60 and cur_y<=0:
            return ccount*0.1
        if cur_y<40:
            if cur_x>10 and cur_x<50:
                return ccount*0.1
        if cur_y>20:
            if cur_x>70 and cur_x<110:
                return ccount*0.1
    return 20
            
