#!/usr/bin/env python
#-*- coding: utf-8 -*-
#  
#  
# ██████   █████  ████████ ████████ ██      ███████ ███████ ██ ███████ ██      ██████  
# ██   ██ ██   ██    ██       ██    ██      ██      ██      ██ ██      ██      ██   ██ 
# ██████  ███████    ██       ██    ██      █████   █████   ██ █████   ██      ██   ██ 
# ██   ██ ██   ██    ██       ██    ██      ██      ██      ██ ██      ██      ██   ██ 
# ██████  ██   ██    ██       ██    ███████ ███████ ██      ██ ███████ ███████ ██████                                                                                                                                                    
#                                                                  
#   2D 3-DOF BATTLEFIELD
#   Version --3DOFfor inertial
#   Created by Hong Daseon

import numpy as np
import time
import random as rd
import DaseonTypesNtf_V3 as Daseon
import pyquaternion as Quaternion
from DaseonTypesNtf_V3 import Vector3, DCM5DOF
from NoFlyZone import NFZ
from Structs import STT
import CraftDynamics
import Lidar
import pdb
import torch
import copy
import math as m
import pygame
import sys
from GameVisual import VisualizationPygame

class BATTLEFIELD:

    def __init__(self, dt, TargetMaxDist, TargetMinDist, MissileViewMax, MissileSpdRNG,\
                                                            MaxNofly, MaxStruct, NoflySizeRng, structSizeRng):
        self.dt             = dt
        self.TargetMaxDist  = TargetMaxDist
        self.TargetMinDist  = TargetMinDist
        self.MaxNofly       = MaxNofly
        self.MaxStruct      = MaxStruct
        self.NoflySizeRng   = NoflySizeRng
        self.structSizeRng  = structSizeRng
        self.fovMaxMissile  = MissileViewMax/180*m.pi
        self.MissileSpdRNG  = MissileSpdRNG
        
        self.NoFlyZones     = []
        self.Structs        = []
        self.TargetInitPos  = Vector3(0.,0.,0.)
        self.Vm             = 0
        self.Target         = None
        self.Missile        = None
        self.Lidar          = None
        self.LidarInfo      = []
        self.init_picker()
        self.MissileSeeker  = CraftDynamics.Seeker(self.Missile, self.Target)

        self.VisualInterface = None

    def __repr__(self):
        dtstring            = 'dt '+ "{:.2f}".format(self.dt) 
        targetPosString     = 'Target Pos ' + "{:.2f}".format(self.TargetPos.x) + ", " + "{:.2f}".format(self.TargetPos.y)
        NoFlyZoneString     = 'No Fly Zone quant '+"{:.0f}".format(len(self.NoFlyZones))
        StructString        = 'Structure quant '+"{:.0f}".format(self.MaxStruct)
        return dtstring+'\n'+targetPosString+'\n'+NoFlyZoneString+'\n'+StructString+'\n'

    def init_picker(self):
        TargetDist              = self.TargetMinDist+ rd.random()*(self.TargetMaxDist-self.TargetMinDist)
        TargetDirec             = 2 * m.pi * rd.random()
        self.TargetInitPos.x    = TargetDist * m.cos(TargetDirec)
        self.TargetInitPos.y    = TargetDist * m.sin(TargetDirec)
        self.Target             = CraftDynamics.Craft(0, self.TargetInitPos, Vector3(0,0,0), self.dt)

        MissileHead             = TargetDirec + (rd.random()-0.5)*self.fovMaxMissile
        self.Vm                 = 200 + rd.random()*200
        self.Missile            = CraftDynamics.Craft(self.Vm, Vector3(0,0,0), Vector3(0,0,MissileHead), self.dt)

        NoflyQuant          = rd.randrange(0,self.MaxNofly)
        for cnt in range(NoflyQuant):
            self.NoFlyZones.append( NFZ(self.NoflySizeRng, self.Target.pos, 20))

        StructQuant         = rd.randrange(0,self.MaxStruct)
        for cnt in range(StructQuant):
            self.Structs.append( STT(rd.random()*3, self.structSizeRng, self.Target.pos, 20) )
        self.Lidar = Lidar.LidarModule(self.Missile, 30, 999999, self.fovMaxMissile)

    def CalcZEM(self, Rvec, Vvec, dt):
        t_minimum        = np.clip( -np.dot(Rvec.vec, Vvec.vec) / np.dot(Vvec.vec, Vvec.vec), 0, dt)
        VecMinimum       = Vector3.cast(Rvec.vec + t_minimum * Vvec.vec)
        return VecMinimum.mag

    def step(self, M_cmd, t, timeOut):
        #print(M_cmd)
        self.Missile.simulate(M_cmd, cmdtype='acc')
        self.LidarInfo   = self.Lidar.StepNSense(self.Missile.pos, self.Missile.att.z, self)
        seekerdat   = self.MissileSeeker.seek(t)
        lidardat    = np.array(self.LidarInfo)
        lidardat    = np.array(lidardat[:,0], dtype=float)
        normlidardat = (lidardat - 5000)/5000
        normlidardat = normlidardat.clip(-1,1)
        #print(np.concatenate([seekerdat, normlidardat], axis = 0))
        RWD, done   = self.referee(M_cmd, timeOut)
        
        return np.concatenate([seekerdat, lidardat], axis = 0), RWD, done

    
    def reset(self, MissileSpdRNG, MaxNofly, MaxStruct, NoflySizeRng, structSizeRng):
        self.MaxNofly       = MaxNofly
        self.MaxStruct      = MaxStruct
        self.NoflySizeRng   = NoflySizeRng
        self.structSizeRng  = structSizeRng
        self.MissileSpdRNG  = MissileSpdRNG
        
        self.NoFlyZones     = []
        self.Structs        = []
        self.TargetInitPos  = Vector3(0.,0.,0.)
        self.Target         = None
        self.Missile        = None
        self.Lidar          = None
        
        self.init_picker()
        self.MissileSeeker  = CraftDynamics.Seeker(self.Missile, self.Target)
        self.LidarInfo      = self.Lidar.StepNSense(self.Missile.pos, self.Missile.att.z, self)

        seekerdat = self.MissileSeeker.seek(0)
        lidardat = np.array(self.LidarInfo)
        lidardat = np.array(lidardat[:,0], dtype=float)
        normlidardat = (lidardat - 5000)/5000
        normlidardat = normlidardat.clip(-1,1)
        #print(np.concatenate([seekerdat, normlidardat], axis = 0))
        return np.concatenate([seekerdat, normlidardat], axis = 0)

    def referee(self, M_cmd, timeOut):
        Rng         = self.MissileSeeker.Rvec.mag
        engy        = M_cmd.y**2 * self.dt
        isCollide   = False
        for stts in self.Structs:
            isCollide = ( isCollide | stts.checkOverrap(self.Missile) )
        rwdStreaming = -engy/2500
        rwd4Result = 0
        done = False
        hit = (Rng <= 30)
        farAway = (self.Missile.pos.mag > self.TargetMaxDist*1.2)

        if isCollide:
            rwd4Result = -100 + 300/Rng
            done = True
        if farAway:
            rwd4Result = -150 + 300/Rng
            done = True
        if timeOut:
            rwd4Result = -150 + 300/Rng
            done = True
        if hit:
            rwd4Result = 200 + 300/Rng
            done = True
        RWD = rwdStreaming + rwd4Result
        return RWD, done

    def render(self, dt, mode='run'):
        if(mode == 'Initialization'):
            self.VisualInterface = VisualizationPygame((800,800), 1/30, joy=False)
            Exit                = False
        
        self.VisualInterface.wipeOut()
        self.VisualInterface.draw_NFZ(self)
        self.VisualInterface.draw_STT(self)
        self.VisualInterface.draw_Spot(self.Missile, (50,100,200), 5)

        self.VisualInterface.draw_lidar(self.Missile, self)
        self.VisualInterface.draw_Spot(self.Target, (200,50,100), 5)
        self.VisualInterface.update()


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

def O3to2(vec3):
    return [vec3.x, vec3.y]

def Test(dt, timeScale, cmdScale):
    
    test_Battlefield = BATTLEFIELD(dt, TMaxDist, TMinDist, MViewMax,\
                                (200,400), MaxNofly, MaxStruct, NoflySizeRng, structSizeRng)#이거말고 리셋설정
    VisualInterface = VisualizationPygame((800,800), 1/30, joy=True)                            
    def visLoop():
        Exit                = False
        prevstepEndtime     = 0
        t                   = 0
        while not Exit:
            loopStartTime   = time.time()
            while time.time() <= (prevstepEndtime+(dt*timeScale)):
                pass
            M_rotRatey  = 0
            M_rotRatey  = VisualInterface.controller.get_axis(3)*cmdScale
            McmdVec     = Vector3(0, M_rotRatey, 0)
            zeroVec     = Vector3(0,0,0)
            test_Battlefield.step(McmdVec, t, False)
            #print(test_Battlefield.Missile.pos)
            VisualInterface.event_get()
            # 시각화
            #gameDisplay.scroll(int(controller.get_axis(1)*10), int(controller.get_axis(2)*10))
            visStartTime = time.time()
            VisualInterface.wipeOut()

            VisualInterface.draw_NFZ(test_Battlefield)
            VisualInterface.draw_STT(test_Battlefield)
            VisualInterface.draw_Spot(test_Battlefield.Missile, (50,100,200), 5)

            #print(test_Battlefield.LidarInfo)
            VisualInterface.draw_lidar(test_Battlefield.Missile, test_Battlefield)
            
            VisualInterface.draw_Spot(test_Battlefield.Target, (200,50,100), 5)
            VisualInterface.update()

            print('visdur : ' + str(time.time()- visStartTime))
            t = t+dt
            prevstepEndtime = time.time()
            if VisualInterface.controller.get_axis(5)>=0.5:
                test_Battlefield.reset(MSpd, MaxNofly, MaxStruct, NoflySizeRng, structSizeRng)
                print("RESET")
                visLoop()
            print('loop dur : ', str(time.time()-loopStartTime))

    while True:
        for event in pygame.event.get():
            if VisualInterface.controller.get_axis(5)>=0.5:
                test_Battlefield.reset(MSpd, MaxNofly, MaxStruct, NoflySizeRng, structSizeRng)
                print("RESET")
                visLoop()


if __name__ == "__main__":
    
    Test(dt, timeScale, cmdScale)
    pygame.joystick.quit()



