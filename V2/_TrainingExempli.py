#!/usr/bin/env python

#-*- coding: utf-8 -*-
#                                           
# ███████ ██████   ██████     ██       █████  ██████  
# ██      ██   ██ ██          ██      ██   ██ ██   ██ 
# █████   ██   ██ ██          ██      ███████ ██████  
# ██      ██   ██ ██          ██      ██   ██ ██   ██ 
# ██      ██████   ██████     ███████ ██   ██ ██████                                            
# FDC Laboratory
#    _____ ____     __  ____           _ __        ____  __ 
#   |__  // __ \   /  |/  (_)_________(_) /__     / __ \/ / 
#    /_ </ / / /  / /|_/ / / ___/ ___/ / / _ \   / /_/ / /  
#  ___/ / /_/ /  / /  / / (__  |__  ) / /  __/  / _, _/ /___
# /____/_____/  /_/  /_/_/____/____/_/_/\___/  /_/ |_/_____/
# _    ___                  __   ____        __  __              
#| |  / (_)______  ______ _/ /  / __ \__  __/ /_/ /_  ____  ____ 
#| | / / / ___/ / / / __ `/ /  / /_/ / / / / __/ __ \/ __ \/ __ \
#| |/ / (__  ) /_/ / /_/ / /  / ____/ /_/ / /_/ / / / /_/ / / / /
#|___/_/____/\__,_/\__,_/_/  /_/    \__, /\__/_/ /_/\____/_/ /_/ 
#                                  /____/                        
# ████████ ██████  ██████  ██████  ██████   ██████  
#    ██    ██   ██ ██   ██ ██   ██ ██   ██ ██       
#    ██    ██   ██ ██   ██ ██   ██ ██████  ██   ███ 
#    ██    ██   ██ ██   ██ ██   ██ ██      ██    ██ 
#    ██    ██████  ██████  ██████  ██       ██████                            
# Designed_by_Daseon_Hong
# Twin Delayed Deep Deterministic Policy Gradient Version
# RL for a Missile

import sys
                                    
import PSpincalc as spin
from pyquaternion import Quaternion
import numpy as np
import torch
import torch.nn as nn
import math as m
import torch.nn.functional as F
import torch.optim as optim
import time
import copy
import pickle
import Missile_Env_6DOF as MissileEnv
import AirCraft_ENV_3D as AirCrftEnv
import csv

from DaseonTypes import Vector3, ASCIIart

from torch.distributions import Normal
                             
from collections import namedtuple

from TDDDPG_Core import TD3, ReplayBuffer

import matplotlib.pyplot as plt
import random as rd        
import vpython as vp       
import pdb                             

# Argument List
#   0. 
#   1. GPU number
#   2. DataStream Discription
#   3. simulation on(1) off(otherwise) verbous(777)


# basic initialization+++++++++++++++++++++++++++++++++++++++++++++
ASCIIart.FDCLAB()
ASCIIart.DDDMissileRL()
print("\n\t\t\t\tWITH")
ASCIIart.VisualPython()
FilePath = "."

# Switches +++++++++++++++++++++++++++++++++++++++++++++++++++++#
realtimeSim     = False                                         #
rospublish      = False                                         #
gazebosim       = False                                         #
Unitysim        = False                                         #
vpythonsim      = (sys.argv[3]=='1') or (sys.argv[3]=='777')    #
verbous         = sys.argv[3]=='777'                            #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

SessionIdentificationNumber = int(rd.random()*10000)

# ++++++++++++++++++++++++ Hyperparameters ++++++++++++++++++++++++
state_dim = 6               # Input dim for actor and critic
allmighty_dim = 3           # Input dim for critic
action_dim = 3              # Output of actor dim
action_limit_abs = 99.0     # Saturation point for action
log_interval = 10           # Print avg reward after interval
random_seed = 777           # Seed
gamma = 0.999999            # Discount for future rewards
batch_size = 1000           # Num of transitions sampled from replay buffer
aclr = 1e-5                 # Learning Rate for actor
cr1lr = 1e-5                # Learning Rate for critic no1
cr2lr = 1e-5                # Learning Rate for critic no2
exploration_noise = 1       # Action Noise Probably
polyak = 0.995              # target policy update parameter (1-tau)
policy_noise = 0.2          # target policy smoothing noise
noise_clip = 5
policy_delay = 2            # delayed policy updates parameter
max_episodes = 5000000      # max num of episodes
max_step_count = 500        # max timesteps in one episode
updatePerStep = False       # True updates policy every step, False updates policy at the end of the episode
directory = "./params/" # save trained models
filename_header = "TD3_"
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
dtSim = 0.1

# nn param
gpu_num = int(sys.argv[1])
mu_now = 0
hitCount = 0
saveCount = 0
RWDAnticipation1 = 0
RWDAnticipation2 = 0
solved = False
slept = False
firstfull = True
LearningCurveFlag = False
t = 0
LearningCurveDataset = []
graph_step_count = 0


# is GPU Available?
device = ('cuda'+':'+ str(gpu_num)) if torch.cuda.is_available() else 'cpu'

torch.manual_seed(random_seed)
np.random.seed(random_seed)
if device == ('cuda'+ ':' + str(gpu_num)) :
    torch.cuda.manual_seed_all(random_seed)

TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])
RewardRecord = namedtuple('RewardRecord', ['ep', 'reward'])

print(device)

# Function Definitions ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def read_CSV():
    global LearningCurveDataset
    count_row = 0
    LearningCurveDataset = []
    with open('./img/'+'LC_'+str(sys.argv[2])+'_'+str(SessionIdentificationNumber)+'.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if(count_row >=1):
                LearningCurveDataset.append(row)
            count_row += 1
        
    LearningCurveDataset = np.array(LearningCurveDataset, dtype=float)
    print('FileReadDone')


def processing_fnc(msg):        #                                                     
                                #
    global att_x                #                                                    
    global att_y                #
    global att_z                #
    global att_w                #
                                #
    att_x = msg.orientation.x   #                                                    
    att_y = msg.orientation.y   #                                                    
    att_z = msg.orientation.z   #                                                    
    att_w = msg.orientation.w   #

def PPNG(N,Vr,Vm,Om,Qnb):
    Acc = ( - N * Vr.mag * (np.cross(Qnb.inverse.rotate(Vm.direction.vec), Om.vec)) )
    return Acc

def write_LearningCurve(chunk, override = 'none'):
    global spamwriter
    if override == 'none' :
        with open('./img/'+'LC_'+str(sys.argv[2])+'_'+str(SessionIdentificationNumber)+'.csv', 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            spamwriter.writerow(chunk)
    else:
        with open('./img/'+'LC_'+str(sys.argv[2])+'_'+str(SessionIdentificationNumber)+'.csv', 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            spamwriter.writerow(override)

def init_picker():
    minradius = 10000
    maxradius = 20000
    minSpeed  = 200
    maxSpeed  = 600
    
    rand_R   = Vector3(rd.random()*(maxradius-minradius) + minradius,0.,0.)
    rand_lam = Vector3(0., (rd.random()-0.5)*m.pi, rd.random()*2*m.pi)
    init_pos = Vector3.cast(RotationQuaternion(rand_lam).rotate(rand_R.vec))
    head_vec = Vector3.cast(init_pos.vec*-1)
    gazeazim, gazeelev = azimNelev(head_vec)
    head_ang = Vector3(0., gazeelev, gazeazim)
    Qhead = RotationQuaternion(head_ang)
    hed_seed = Vector3(0., (rd.random()-0.5)*m.pi, (rd.random()-0.5)*m.pi)
    Qseed = RotationQuaternion(hed_seed)
    Qrotation = Qhead*Qseed
    rand_hed = Qrotation.rotate(rand_R.vec)
    headazim, headelev = azimNelev(Vector3.cast(rand_hed))
    init_hed = Vector3(0., headelev, headazim)
    
    init_spd = (maxSpeed - minSpeed)*rd.random() + minSpeed

    return init_pos, init_hed, init_spd  

def Transmit2Vpython(MissileInstance, MissileSeekerInstance, t, td_reward):
    global LOOK_elev_graph, LOOK_azim_graph, R_graph, accy_graph, accz_graph
    global rwd_graph, missile_visual, xy_traj, xz_traj
    global graph_step_count
    global muy_graph, muz_graph, ExpectedRWD1, ExpectedRWD2

    if vpythonsim & (not M_done):
        R, LOOK, dLOS, Scalvel, _ = MissileSeekerInstance.seek(t)

        LOOK_azim_graph.plot(t,LOOK.z)
        LOOK_elev_graph.plot(t,LOOK.y)        
        R_graph.plot(t,R)
        accx_graph.plot(t,MissileInstance.acc.x)
        accy_graph.plot(t,MissileInstance.acc.y)
        accz_graph.plot(t,MissileInstance.acc.z)
        mux_graph.plot(t, mu_now[0])
        muy_graph.plot(t, mu_now[1])
        muz_graph.plot(t, mu_now[2])
        missile_visual.pos  = MissileInstance.pos.VPvec
        missile_visual.axis = MissileInstance.dpos.VPvec
        if graph_step_count == 20:
            #xy_traj.plot(missile_visual.pos.x, missile_visual.pos.y)
            #xz_traj.plot(missile_visual.pos.x, -missile_visual.pos.z)
            graph_step_count = 0
        graph_step_count +=1
        scene1.center = vp.vec(0,0,0)
        ExpectedRWD1.plot(t,RWDAnticipation1)
        ExpectedRWD2.plot(t,RWDAnticipation2)

def VpythonStatusGraph(epsd, impacR):
    global stt_graph
    if vpythonsim:stt_graph.plot(epsd, impacR)

def VpythonClearGraph():
    LOOK_azim_graph.delete()
    LOOK_elev_graph.delete()
    R_graph.delete()
    accx_graph.delete()
    accy_graph.delete()
    accz_graph.delete()
    mux_graph.delete()
    muy_graph.delete()
    muz_graph.delete()
    rwd_graph.delete()
    ExpectedRWD1.delete()
    ExpectedRWD2.delete()

def statPurge():
    stt_graph.delete()
    return 'purged'

def learningCurve():
    global LearningCurveFlag
    LearningCurveFlag = True

def VpythonClearTraj():
    global xy_traj
    global xz_traj
    #xy_traj.delete
    #xz_traj.delete
    pass

def azimNelev(vec):
        azim = m.atan2( vec.y, vec.x)
        elev = m.atan2( -vec.z, m.sqrt( vec.x**2 + vec.y**2))
        return azim, elev

def euler_to_quaternion(att):

        roll = att.x
        pitch = att.y
        yaw = att.z

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]

def RotationQuaternion(att, option="n_b"):
    _qx = Quaternion(axis=[1, 0, 0], angle=att.x)
    _qy = Quaternion(axis=[0, 1, 0], angle=att.y)
    _qz = Quaternion(axis=[0, 0, 1], angle=att.z)
    return _qz*_qy*_qx

#=======================================================================

# Visual Python Initialize
if vpythonsim:
    scene1 = vp.canvas(title = "close_look"+sys.argv[2],
                    x=0,y=0, width=500, height=300,
                    range=10, background=vp.color.black,
                    center = vp.vec(10,10,0))

    g1 = vp.graph(title = "LOOKazim",width = 500, height = 200, ymin = -1.57, ymax = 1.57)
    g2 = vp.graph(title = "LOOKelev",width = 500, height = 200, ymin = -1.57, ymax = 1.57)
    g3 = vp.graph(title = "R", width = 500, height = 200)
    g35= vp.graph(title = "accx", width = 500, height = 200)
    g4 = vp.graph(title = "accy", width = 500, height = 200)
    g5 = vp.graph(title = "accz", width = 500, height = 200)
    g6 = vp.graph(title = "reward", width = 500, height = 200)
    g7 = vp.graph(title = "training_status", width = 500, height = 500)
    g8 = vp.graph(title = "xytrajectory", width = 500, height = 500, xmin = -20000, xmax = 20000, ymin = -20000, ymax = 20000)
    g9 = vp.graph(title = "xztrajectory", width = 500, height = 500, xmin = -20000, xmax = 20000, ymin = -20000, ymax = 20000)

    verb1 = vp.graph(title = "Expected Reward", width=500, height= 200)

    StatPurgeButton = vp.button(text = 'PURGE STATUS', bind = statPurge, width=200, height = 20)
    LearningCurveButton = vp.button(text = 'VIEW CURVES', bind = learningCurve, width=200, height=20)

#=======================================================================

#=======================================================================
def norm_R_reward(sacrifice):
    loged = 9.91-m.log(-sacrifice)
    normV = loged/10
    return normV

def norm_A_reward(sacrifice,Vm,initLOS,t_f):
    stand = sacrifice/Vm/t_f/m.sqrt(abs(initLOS)+0.5)
    normV = ((-m.log(-stand))-3)/4
    return normV

def norm_Acc(sacrifice, Vm, initLOS):
    stand = (sacrifice/Vm/m.cos(initLOS)/m.sqrt(abs(initLOS)+0.5))*dtSim
    normV = ((-m.log(-stand))-3)/8
    return normV
#=======================================================================

# Instances +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MissileModel_1 = MissileEnv.Missile_3D(0, Vector3.cast([0,0,0]), Vector3.cast([0,0,0]), dtSim)
print(MissileModel_1)
FighterModel_1 = AirCrftEnv.Craft_3D(0, Vector3.cast([0,0,0]), Vector3.cast([0,0,0]), dtSim)
print(FighterModel_1)
MissileSeeker_1 = MissileEnv.Seeker(MissileModel_1, FighterModel_1)
FighterSeeker_1 = AirCrftEnv.Seeker(FighterModel_1, MissileModel_1)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#=======================================================================
# Visual Graph Initilize
if vpythonsim:
    LOOK_azim_graph  = vp.gdots(graph = g1, color = vp.color.black, radius=1)
    LOOK_elev_graph  = vp.gdots(graph = g2, color = vp.color.black, radius=1)

    R_graph     = vp.gdots(graph = g3, color = vp.color.magenta, radius=1)
    accx_graph  = vp.gdots(graph = g35, color = vp.color.black, radius=1)
    accy_graph  = vp.gdots(graph = g4, color = vp.color.black, radius=1)
    accz_graph  = vp.gdots(graph = g5, color = vp.color.black, radius=1)
    mux_graph   = vp.gdots(graph = g35, color = vp.color.red, radius=1)
    muy_graph   = vp.gdots(graph = g4, color = vp.color.red, radius=1)
    muz_graph   = vp.gdots(graph = g5, color = vp.color.red, radius=1)
    rwd_graph   = vp.gdots(graph = g6, color = vp.color.blue, radius=1)
    stt_graph   = vp.gdots(graph = g7, color = vp.color.green, radius=1)
    #xy_traj     = vp.gdots(graph = g8, color = vp.vec(rd.random(),rd.random(),rd.random()), radius = 1)
    #xz_traj     = vp.gdots(graph = g9, color = vp.vec(rd.random(),rd.random(),rd.random()), radius = 1)

    missile_visual = vp.sphere(axis = MissileModel_1.dpos.VPvec, size = vp.vec(2.01,0.5,0.5), color = vp.color.white,
                            accaxis = MissileModel_1.acc.VPvec, make_trail = True, retain = 10)
    fighter_visual  = vp.box(length = 50, width = 15, height = 15, pos = FighterModel_1.pos.VPvec)

    missile_visual.pos = MissileModel_1.pos.VPvec

    ExpectedRWD1 = vp.gdots(graph = verb1, color = vp.color.blue, radius=1)
    ExpectedRWD2 = vp.gdots(graph = verb1, color = vp.color.blue, radius=1)
#missile_visual.v   = MissileModel_1.dpos.VPvec
#=======================================================================

# ++++++++++++++++++++ Init RL Core ++++++++++++++++++++
agent = TD3(aclr, cr1lr, cr2lr, state_dim, action_dim, action_limit_abs, device)
replay_buffer = ReplayBuffer()

training_records    = []
Reward_records      = []



# logging
avg_reward = 0
ep_reward = 0
log_f = open("log.txt","w+")

write_LearningCurve(None, ['ep', 'rwd', 'lpf'])

#     __                          _                _____ __             __      
#    / /   ___  ____ __________  (_)___  ____ _   / ___// /_____ ______/ /______
#   / /   / _ \/ __ `/ ___/ __ \/ / __ \/ __ `/   \__ \/ __/ __ `/ ___/ __/ ___/
#  / /___/  __/ /_/ / /  / / / / / / / / /_/ /   ___/ / /_/ /_/ / /  / /_(__  ) 
# /_____/\___/\__,_/_/  /_/ /_/_/_/ /_/\__, /   /____/\__/\__,_/_/   \__/____/  
#                                     /____/                                    
ASCIIart.LearningStarts()
for i_ep in range(max_episodes):

    t = 0. ##sec
    Cassette_tape = []
    score = 0
    
    posM, hedM, spdM = init_picker()
    print('Session' + str(sys.argv[2]) +'  '+str(SessionIdentificationNumber) + ' GPU:' + str(gpu_num))
    print('===================================================================================')
    print('EP : ', i_ep, '| Init : ','\n\t\t',posM,'\n\t\t',hedM,'\n\t\t',spdM)
    
    MissileSeeker_1.impactR = 50000
    integral_acc = np.array([0.,0.])
    step_count = 0
    
    print('--------------------------------------RL---------------------------Replaying? ')

    MissileModel_1.reset(posM, hedM, spdM, reset_flag = True)
    t = 0 ##sec
    step_count = 0
    MissileSeeker_1.impactR = 50000
    _, initLook, LOSdot, _, state = MissileSeeker_1.seek(t)
    Look = initLook
    energy_term = []
    energy = 0
    pngtime = []
    pngacc  = []
    print(MissileSeeker_1.Look.z, MissileSeeker_1.Look.y)

    while t<max_step_count:
        # save prev LOS val
        MissileSeeker_1.newStepStarts(t)
        # Select Action
        action = agent.select_action(state)
        mu_now = action
        action = action + np.random.normal(0, exploration_noise, size=action_dim)
        action = action.clip(-action_limit_abs, action_limit_abs)
        # Apply action into the environment
        Acc_cmd = Vector3(action[0], action[1], action[2])
        MissileModel_1.simulate(Acc_cmd)
        # Get next state
        Rmag, Look_, LOSdot, _, state_ = MissileSeeker_1.seek(t)
        # Get Reward
        Mt_reward, Mm_reward, M_done, M_is_hit = MissileSeeker_1.spit_reward(Acc_cmd)
        Ct_reward, Cm_reward, C_done, C_is_hit = FighterSeeker_1.spit_reward(Vector3(0,0,0))
        
        if t == max_step_count-1:
            M_done = True
            C_done = True

        if M_is_hit:
            hitCount += 1
            print('\t'+'hit!!!++++++++++++++++++++++!!!!!!!!!!!!!!!!!'+str(hitCount))
            if not solved : 
                pass
        
        if vpythonsim: Transmit2Vpython(MissileModel_1, MissileSeeker_1, t, Mt_reward)
        
        integral_acc += Mt_reward*dtSim
        
        final_reward =    0.7*norm_R_reward(Mm_reward) \
                    + 0.0\
                    + 0.15*norm_Acc(Mt_reward[0],spdM, initLook.y)\
                    + 0.15*norm_Acc(Mt_reward[1],spdM, initLook.z)
        score = final_reward*M_done
        if verbous: print('Score : ',score)

        replay_buffer.add((state, action, score, state_, float(M_done)))
        
        state = state_
        Look  = Look_
        if verbous : print('state : ',state)
        with torch.no_grad():
            RWDAnticipation1 = agent.critic_1_target( torch.tensor([state], dtype=torch.float), \
                                                     agent.actor_target(torch.tensor([state], dtype=torch.float)))\
                                                    .to('cpu').numpy()[0][0]          
            RWDAnticipation2 = agent.critic_2_target( torch.tensor([state], dtype=torch.float), \
                                                     agent.actor_target(torch.tensor([state], dtype=torch.float)))\
                                                    .to('cpu').numpy()[0][0]   
        # if episode is done then update policy:
        if M_done or updatePerStep:
            agent.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay, dtSim)
            break

        t = t + dtSim
 
        if M_done:
            print(M_is_hit)
            break

    
    if vpythonsim: VpythonStatusGraph(i_ep, -Mm_reward)
    #for rowrow in Cassette_tape:
    #    agent.store_transition(Transition(rowrow[0], rowrow[1], score, rowrow[2]))
    #    print(rowrow[0], rowrow[1], score, rowrow[2])
    if not slept:
        running_reward = score
    slept = True

    if(i_ep % 2 == 0 and vpythonsim): VpythonClearGraph()

    running_reward = running_reward * 0.9 + score * 0.1
    training_records.append(TrainingRecord(i_ep, running_reward))
    Reward_records.append(RewardRecord(i_ep, score))
    #print(i_ep, running_reward, running_q)
    if LearningCurveFlag:
        read_CSV()
        plt.figure(figsize=(20, 8))
        plt.plot(LearningCurveDataset[:,0], LearningCurveDataset[:,1])
        plt.plot(LearningCurveDataset[:,0], LearningCurveDataset[:,2])
        plt.title('RWDs-LPF')
        plt.xlabel('Episode')
        plt.ylabel('reward sum')
        plt.savefig("./img/reward_lpf.png")
        LearningCurveFlag = False
    # Generating LearningCurve
    T1 = time.time()
    write_LearningCurve([i_ep, score, running_reward])
    print(time.time()-T1)
        #a_render = True
    if M_is_hit:
        print("Solved! Running reward is now {}!".format(running_reward))
        #env.close()
        saveCount += 1
 
        agent.save(directory, filename_header+str(i_ep))

        if running_reward > 0: solved = True
        #with open('log/ddpg_training_records.pkl', 'wb') as f:
        #    pickle.dump(training_records, f)
        #break
    
    print('===================================================================================\n\n')

quit()


#++++++++++++++++++++++++++++++Functions++++++++++++++++++++++++++++++

