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
        randomTitleHolder   = Daseon.randomTitle()
        self.Title          = randomTitleHolder.title
        self.VisualInterface = None
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('++++++++++++++++++'+ self.Title +'+++++++++++++++++')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

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
        Vrel        = self.MissileSeeker.Vrel
        isCollide   = False
        for stts in self.Structs:
            isCollide = ( isCollide | stts.checkOverrap(self.Missile.pos) )
        rwdStreaming = -engy/2500 - Vrel/200
        rwd4Result = 0
        done = False
        
        farAway = (self.Missile.pos.mag > self.TargetMaxDist*1.2)
        ViewOut = abs(self.MissileSeeker.Look.z)>(50/180*m.pi)
        Rng = self.getDist(ViewOut)
        hit = (Rng <= 50)
        if isCollide:
            rwd4Result = -100 + 300/Rng
            done = True
        if farAway:
            rwd4Result = -150 + 300/Rng
            done = True
        if timeOut:
            rwd4Result = -150 + 300/Rng
            done = True
        if ViewOut:
            rwd4Result = -100 + 200/Rng
            done = True
        if hit:
            rwd4Result = 200 + 300/Rng
            done = True
        RWD = rwdStreaming + rwd4Result
        return RWD, done

    def render(self, dt, mode='run'):
        if(mode == 'Initialization'):
            self.VisualInterface = VisualizationPygame((800,800), 1/30, self.Title,joy=False)
            Exit                = False
        
        self.VisualInterface.wipeOut()
        self.VisualInterface.draw_NFZ(self)
        self.VisualInterface.draw_STT(self)
        self.VisualInterface.draw_Spot(self.Missile, (50,100,200), 5)

        self.VisualInterface.draw_lidar(self.Missile, self)
        self.VisualInterface.draw_Spot(self.Target, (200,50,100), 5)
        self.VisualInterface.update()

    def getDist(self, OOR):
        if OOR:
            Rf_1 = self.MissileSeeker.prev_Rm
            Rf = self.MissileSeeker.Yo.pos
            
            R3 = Rf - Rf_1
            A = R3
            B = (self.MissileSeeker.Tu.pos - Rf_1) - R3
            
            if self.MissileSeeker.Rvec.mag < 50:

                self.MissileSeeker.impactR = (Vector3.cast(np.cross(A.vec,B.vec)).mag) / A.mag 

            else:
                self.MissileSeeker.impactR = self.MissileSeeker.Rvec.mag

            rwdR = copy.deepcopy(self.MissileSeeker.impactR)
        else:
            self.MissileSeeker.prev_Rm = copy.deepcopy(self.MissileSeeker.Yo.pos)
            rwdR = copy.deepcopy(self.MissileSeeker.Rvec.mag)

        return rwdR





