import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import math


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


        #this space is reserved, about boundaries




        self.seed()
        self.reset()
    

    def reset(self):
        self.start_loc = np.array([0,self.np_random.uniform(low=0,high=30)])
        self.target_loc = np.array([120,self.np_random.uniform(low=30,high=60)])
        self.cur_x_real = self.start_loc[0]
        self.cur_y_real = self.start_loc[1]
        self.speed = 2
        self.speed_direc = 0.5
        self.x_dist = self.target_loc[0]-self.start_loc[0]
        self.y_dist = self.target_loc[1]-self.start_loc[1]
        self.real_dist = math.sqrt(self.x_dist**2+self.y_dist**2)
        normed_dist = (self.real_dist-100)/100
        normed_xstart = (self.start_loc[0]-60)/30
        normed_ystart = (self.start_loc[1]-30)/30
        self.target_direc = math.atan(self.y_dist/self.x_dist)/math.pi
        self.state = np.array([1,1,1,1,1,1,1,self.speed_direc,normed_dist,self.target_direc,normed_xstart,normed_ystart])
        self.detect_direc = [1/2,1/3,1/6,0,-1/6,-1/3,-1/2]
        return self.state

    def seed(self, seed=None): #i don't know what's this for actually
        self.np_random, seed =seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #perform move, and detect obstacles
        self.speed_direc=angle_norm(self.state[7]+action)
        self.cur_x_real=self.state[10]*30+60+self.speed*math.cos(self.speed_direc)
        self.cur_y_real=self.state[11]*30+30+self.speed*math.sin(self.speed_direc)
        #now get observations
        self.x_dist=self.target_loc[0]-self.cur_x_real
        self.y_dist=self.target_loc[1]-self.cur_y_real
        self.target_direc=math.atan(self.y_dist/self.x_dist)/math.pi
        self.real_dist = math.sqrt(self.x_dist**2+self.y_dist**2)
        #now detect obstacles. this can be parallized later
        kk=0
        for bias in self.detect_direc:
            direc = angle_norm(self.speed_direc+bias)
            self.state[kk] = getdistance(self.cur_x_real,self.cur_y_real,direc)
            kk = kk+1
        #now renew other states
        self.state[7] = self.speed_direc
        self.state[8] = (self.real_dist-100)/100
        self.state[9] = self.target_direc
        self.state[10] = (self.cur_x_real-60)/30
        self.state[11] = (self.cur_y_real-30)/30





        
        


    

        





    


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
            
