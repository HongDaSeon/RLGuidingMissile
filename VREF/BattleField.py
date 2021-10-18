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
        self.TargetMaxDist  = TargetMinDist
        self.MaxNofly       = MaxNofly
        self.MaxStruct      = MaxStruct
        self.NoflySizeRng   = NoflySizeRange
        self.structSizeRng  = structSizeRng
        self.fovMaxMissile  = MissileViewMax
        self.Vm             = MissileSpd
        
        self.NoFlyZones     = []
        self.Structs        = []
        self.TargetInitPos  = Vector3(0.,0.,0.)
        self.Target         = None
        self.Missile        = None

    def __repr__(self):
        dtstring            = 'dt '+ "{:.2f}".format(self.dt) 
        targetPosString     = 'Target Pos ' + "{:.2f}".format(self.TargetPos.x) + ", " + "{:.2f}".format(self.TargetPos.y)
        NoFlyZoneString     = 'No Fly Zone quant '+"{:.0f}".format(len(self.NoFlyZones))
        StructString        = 'Structure quant '+"{:.0f}".format(self.MaxStruct)
        return dtstring+'\n'+targetMinMaxString+'\n'+MaxNoFlyZoneString+'\n'+MaxStructString+'\n'

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
            self.NoFlyZone.append( NFZ(self.NoflySizeRng, self.Target.pos, 200)

        StructQuant         = rd.randrange(0,self.MaxStruct)
        for cnt in range(StructQuant):
            self.NoFlyZone.append( STT(self.structSizeRng, self.Target.pos) )


    def CalcZEM(self, Rvec, Vvec, dt):
        t_minimum        = np.clip( -np.dot(Rvec.vec, Vvec.vec) / np.dot(Vvec.vec, Vvec.vec), 0, dt)
        VecMinimum       = Vector3.cast(Rvec.vec + t_minimum * Vvec.vec)
        return VecMinimum.mag
        
            




