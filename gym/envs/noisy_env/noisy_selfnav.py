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
    #the detection range should be [0,20], and we normalize it to [-1,1]
    #the x and y scale should be [0,120] and [0,60], and we normalize it to [-2,2] and [-1,1]
    #the distance to target should be [0,200], we normalize it to [-1,1]

    def __init__(self):
        self.max_detec_dis=1
        self.min_detec_dis=-1
        self.max_direction=1
        self.min_direction=-1
        self.max_x=2
        self.min_x=-2
        self.max_y=1
        self.min_y=-1
        self.max_targ_dis=1
        self.min_targ_dis=-1

        self.max_action=0.5 #can only turn 90
        self.min_action=-0.5

        self.low_state=np.array([self.min_detec_dis,self.min_detec_dis,self.min_detec_dis,\
        self.min_detec_dis,self.min_detec_dis,self.min_detec_dis,self.min_detec_dis,\
        self.min_direction,self.min_targ_dis,self.min_direction,self.min_x=0,\
        self.min_y],dtype=np.float32)

        self.high_state=np.array([self.max_detec_dis,self.max_detec_dis,self.max_detec_dis,\
        self.max_detec_dis,self.max_detec_dis,self.max_detec_dis,self.max_detec_dis,\
        self.max_direction,self.max_targ_dis,self.max_direction,self.max_x=0,\
        self.max_y],dtype=np.float32)

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=self.low_state
            high=self.high_state
            dtype=np.float32
        )

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
        self.old_dist = self.real_dist 
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
        #perform move first
        self.speed_direc=angle_norm(self.state[7]+action)
        self.cur_x_real=self.state[10]*30+60+self.speed*math.cos(self.speed_direc)
        self.cur_y_real=self.state[11]*30+30+self.speed*math.sin(self.speed_direc)
        #get observations
        self.x_dist=self.target_loc[0]-self.cur_x_real
        self.y_dist=self.target_loc[1]-self.cur_y_real
        self.target_direc=math.atan(self.y_dist/self.x_dist)/math.pi
        self.old_dist = self.real_dist
        self.real_dist = math.sqrt(self.x_dist**2+self.y_dist**2)
        #detect obstacles. this can be parallized later
        kk=0
        for bias in self.detect_direc:
            direc = angle_norm(self.speed_direc+bias)
            self.state[kk] = getdistance(self.cur_x_real,self.cur_y_real,direc)
            kk = kk+1
        #renew other states
        self.state[7] = self.speed_direc
        self.state[8] = (self.real_dist-100)/100
        self.state[9] = self.target_direc
        self.state[10] = (self.cur_x_real-60)/30
        self.state[11] = (self.cur_y_real-30)/30
        #judge whether done
        if self.real_dist < 10 :
            isdone = True
        else :
            isdone = False
        #calculate reward
        #obstacle penalty
        min_dis = 10*np.min(self.state[0:7])+10
        obs_pny = -8*np.exp(-25*min_dis)
        #step penalty
        sep_pny = -0.6
        #transition reward
        trans_reward = 2*(self.old_dist-self.real_dist)
        #success reward
        if isdone:
            success_reward = 100
        else:
            success_reward = 0
        total_reward = obs_pny+sep_pny+trans_reward+success_reward

        return self.state, total_reward , isdone, {}

    
    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(1200,600)
            obs_a_loc = [[10,0],[10,40],[50,0],[50,40]]
            obs_b_loc = [[70,20],[70,60],[110,20],[110,60]]
            self.obs_a = rendering.make_polygon(obs_a_loc)
            self.obs_b = rendering.make_polygon(obs_b_loc)
            self.viewer.add_geom(self.obs_a)
            self.viewer.add_geom(self.obs_b)

            car = rendering.make_circle(2)
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
        
        pos_x = self.state[10]*30 + 60
        pos_y = self.state[11]*30 + 30
        self.cartrans.set_translation(pos_x,pos_y)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')






    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None




def angle_norm(x):
    if x>1:
        x=x-2
    if x<-1:
        x=x+2
    return x

def getdistance(x,y,direc):
    delta_x=0.1*np.cos(direc)
    delta_y=0.1*np.sin(direc)
    #recover normal axis
    cur_x = x
    cur_y = y
    for ccount in range(200):
        cur_x=cur_x+delta_x
        cur_y=cur_y+delta_y
        if cur_y>=60 and cur_y<=0:
            return ccount*0.01-1
        if cur_y<40:
            if cur_x>10 and cur_x<50:
                return ccount*0.01-1
        if cur_y>20:
            if cur_x>70 and cur_x<110:
                return ccount*0.01-1
    return 1
            
