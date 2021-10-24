#!/usr/bin/env python
#-*- coding: utf-8 -*-

#     ______         _      _             
#    /_  __/______ _(_)__  (_)__  ___ _   
#     / / / __/ _ `/ / _ \/ / _ \/ _ `/   
#    /_/_/_/  \_,_/_/_//_/_/_//_/\_, /    
#     / ___/__ _  _____ _______ /___/ ____
#    / (_ / _ \ |/ / -_) __/ _ \/ _ \/ __/
#    \___/\___/___/\__/_/ /_//_/\___/_/   
#    Training Governor                                      

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import DaseonTypesNtf_V3 as Daseon
import time
import random as rd
import os
import math as m
from CraftDynamics import Craft
from DaseonTypesNtf_V3 import Vector3, DCM5DOF, lineEQN
from BattleField import BATTLEFIELD
from RPbuffer import ReplayBuffer
from collections import namedtuple
from TD3Core import TD3

def LearningSequence_1():
    dt              = 0.1
    TMinDist        = 5000
    TMaxDist        = 10000
    MViewMax        = 100
    MSpd            = 280
    MaxNofly        = 5
    MaxStruct       = 30
    NoflySizeRng    = (500,3000)
    structSizeRng   = (50,400)
    timeScale       = 0
    cmdScale        = 30

    ######### Hyperparameters #########
    env_name = "Vref"
    log_interval = 10           # print avg reward after interval
    random_seed = 0
    gamma = 0.9999              # discount for future rewards
    batch_size = 100            # num of transitions sampled from replay buffer
    alr = 0.0001
    clr1 = 0.0001
    clr2 = 0.0001
    exploration_noise = 0.1 
    polyak = 0.995              # target policy update parameter (1-tau)
    policy_noise = 0.2          # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2            # delayed policy updates parameter
    max_episodes = 100000       # max num of episodes
    max_t        = 200          # max timesteps in one episode
    directory = "./params/{}".format(env_name) # save trained models
    filename_header = "TD3_{}".format(env_name)
    ###################################

    gpu_num = int(sys.argv[1])
    device = ('cuda'+':'+ str(gpu_num)) if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if device == ('cuda'+ ':' + str(gpu_num)) :
        torch.cuda.manual_seed_all(random_seed)

    TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])
    RewardRecord = namedtuple('RewardRecord', ['ep', 'reward'])

    state_dim = 32
    action_dim  = 1
    max_action  = 99.
    
    policy = TD3(alr, clr1, clr2, state_dim, action_dim, max_action, gpu_num)
    replay_buffer = ReplayBuffer()

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    # logging variables:
    avg_reward = 0
    ep_reward = 0
    log_f = open("log.txt","w+")

    Battlefield = BATTLEFIELD(dt, TMaxDist, TMinDist, MViewMax,\
                                        (200,400), MaxNofly, MaxStruct, NoflySizeRng, structSizeRng)
    Battlefield.render(dt, 'Initialization')
    for episode in range(1, max_episodes+1):
        state   = Battlefield.reset((200,400), MaxNofly, MaxStruct, NoflySizeRng, structSizeRng)
        t       = 0
        cnt     = 1
        while( t <= (max_t + 1)):
            action = policy.select_action(state)
            action = action + np.random.normal(0, exploration_noise, action_dim)
            action = action.clip(-max_action, max_action)

            next_state, reward, done = Battlefield.step(Vector3(0,action[0],0), t, (t >= max_t))
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state
            Battlefield.render(dt)
            avg_reward += reward
            ep_reward += reward
            
            if done:
                policy.update(replay_buffer, cnt, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                break

            t = t + dt
            cnt = cnt + 1

        # logging updates:
        log_f.write('{},{}\n'.format(episode, ep_reward))
        log_f.flush()
        ep_reward = 0

        # if avg reward > 300 then save and stop traning:
        if (avg_reward/log_interval) >= 200:
            print("########## Solved! ###########")
            name = filename_header + '_solved'
            policy.save(directory, name)
            log_f.close()
            #break
        
        if (episode%100)==0:
            policy.save(directory, filename_header+str(episode))
        
        # print avg reward every log interval:
        if episode % log_interval == 0:
            avg_reward = int(avg_reward / log_interval)
            print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
            avg_reward = 0


if __name__ == '__main__':
    LearningSequence_1()
            



    

