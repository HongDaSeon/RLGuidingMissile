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

class BATTLEFIELD:

    def __init__(self, dt, TargetMaxDist, TargetMinDist, MissileViewMax, MissileSpd,\
                                                            MaxNofly, MaxStruct, NoflySizeRng, structSizeRng):
        self.dt             = dt
        self.TargetMaxDist  = TargetMaxDist
        self.TargetMinDist  = TargetMinDist
        self.MaxNofly       = MaxNofly
        self.MaxStruct      = MaxStruct
        self.NoflySizeRng   = NoflySizeRng
        self.structSizeRng  = structSizeRng
        self.fovMaxMissile  = MissileViewMax/180*m.pi
        self.Vm             = MissileSpd
        
        self.NoFlyZones     = []
        self.Structs        = []
        self.TargetInitPos  = Vector3(0.,0.,0.)
        self.Target         = None
        self.Missile        = None
        self.Lidar          = None
        self.LidarInfo      = []

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

    def step(self, M_cmd, t, forcedQuit):
        #print(M_cmd)
        self.Missile.simulate(M_cmd, cmdtype='acc')
        self.LidarInfo   = self.Lidar.StepNSense(self.Missile.pos, self.Missile.att.z, self)
    
    def reset(self, MissileSpd, MaxNofly, MaxStruct, NoflySizeRng, structSizeRng):
        self.MaxNofly       = MaxNofly
        self.MaxStruct      = MaxStruct
        self.NoflySizeRng   = NoflySizeRng
        self.structSizeRng  = structSizeRng
        self.Vm             = MissileSpd
        
        self.NoFlyZones     = []
        self.Structs        = []
        self.TargetInitPos  = Vector3(0.,0.,0.)
        self.Target         = None
        self.Missile        = None
        self.Lidar          = None
        self.LidarInfo      = []
        self.init_picker()

dt              = 0.01
TMinDist        = 5000
TMaxDist        = 10000
MViewMax        = 100
MSpd            = 280
MaxNofly        = 5
MaxStruct       = 30
NoflySizeRng    = (500,3000)
structSizeRng   = (50,200)
timeScale       = 1
cmdScale        = 30

def O3to2(vec3):
    return [vec3.x, vec3.y]

def Test(dt, timeScale, cmdScale):
    
    test_Battlefield = BATTLEFIELD(dt, TMaxDist, TMinDist, MViewMax,\
                                MSpd, MaxNofly, MaxStruct, NoflySizeRng, structSizeRng)#이거말고 리셋설정
    pygame.init()
    pygame.joystick.init()

    try:
        controller = pygame.joystick.Joystick(0)
        controller.init()
        print ("Joystick_Paired: {0}".format(controller.get_name()))
    except pygame.error:
        print ("None of or Invalid joystick connected")

    display_width = 800
    display_height = 800
    gameDisplay = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption('Environment Test Environment')
    C_Missile       = (50, 100, 200)
    C_NoflyZone     = (200, 100, 200)
    C_lidarSens     = (150, 150, 200)
    C_Target        = (200, 50, 100)
    C_lGrey         = (200, 200, 200)
    centre          = (display_width/2, display_height/2)
    
    Lookscale       = 1/30
    def in2Dcenter(d2list):
        
        return [d2list[0]*Lookscale+display_width/2, d2list[1]*Lookscale+display_height/2 ]

    def visLoop():
        Exit                = False
        H_spotx             = display_width/2
        H_spoty             = display_height/2
        prevstepEndtime     = 0
        t                   = 0
        shift_x             = 0
        shift_y             = 0
        while not Exit:
            while time.time() <= (prevstepEndtime+(dt*timeScale)):
                pass
            M_rotRatex  = 0
            M_rotRatey  = 0
            M_rotRatex  = controller.get_axis(3)*cmdScale
            M_rotRatey  = controller.get_axis(3)*cmdScale
            shift_x     = shift_x + controller.get_axis(1)
            shift_y     = shift_y + controller.get_axis(2)
            #print(controller.get_axis(1))
            #print(controller.get_axis(2))
            #print(controller.get_axis(3))
            #print(controller.get_axis(4))

            McmdVec     = Vector3(0, M_rotRatey, 0)
            zeroVec     = Vector3(0,0,0)
            test_Battlefield.step(McmdVec, t, False)
            #print(test_Battlefield.Missile.pos)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print('goodbye')
                    sys.exit()
                if event.type == pygame.JOYAXISMOTION:  # Joystick
                    pass
                if event.type == pygame.JOYBUTTONDOWN:  
                    print("Joystick Button pressed")
            # 시각화
            #gameDisplay.scroll(int(controller.get_axis(1)*10), int(controller.get_axis(2)*10))
            
            pygame.draw.rect(gameDisplay, (0,0,0), [0, 0 , display_width, display_height])

            for nfzs in test_Battlefield.NoFlyZones:
                pygame.draw.circle(gameDisplay, C_NoflyZone, in2Dcenter(O3to2(nfzs.pos)), nfzs.radius*Lookscale)
            for stts in test_Battlefield.Structs:
                pygame.draw.polygon(gameDisplay, C_lGrey, [ in2Dcenter(stts.d2vertices[0]),\
                                                            in2Dcenter(stts.d2vertices[1]),\
                                                            in2Dcenter(stts.d2vertices[2]),
                                                            in2Dcenter(stts.d2vertices[3])], 2)
            M2Dpos = O3to2(test_Battlefield.Missile.pos)
            pygame.draw.circle(gameDisplay, C_Missile, in2Dcenter(M2Dpos), 5)
            #print(test_Battlefield.LidarInfo)
            for lidars in test_Battlefield.LidarInfo:
                pygame.draw.aaline(gameDisplay, C_lidarSens, in2Dcenter(M2Dpos), in2Dcenter(O3to2(lidars[1])), 1)
            pygame.draw.circle(gameDisplay, C_Target, in2Dcenter(O3to2(test_Battlefield.Target.pos)),5)
            pygame.display.update()

            t = t+dt
            prevstepEndtime = time.time()
            if controller.get_axis(5)>=0.5:
                test_Battlefield.reset(MSpd, MaxNofly, MaxStruct, NoflySizeRng, structSizeRng)
                print("RESET")
                visLoop()

    while True:
        for event in pygame.event.get():
            if controller.get_axis(5)>=0.5:
                test_Battlefield.reset(MSpd, MaxNofly, MaxStruct, NoflySizeRng, structSizeRng)
                print("RESET")
                visLoop()

if __name__ == "__main__":
    
    Test(dt, timeScale, cmdScale)
    pygame.joystick.quit()



