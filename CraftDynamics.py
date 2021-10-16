#!/usr/bin/env python
#-*- coding: utf-8 -*-
#  
#  
# |         ███████ ██████   ██████     ██       █████  ██████       | # 
# |         ██      ██   ██ ██          ██      ██   ██ ██   ██      | # 
# |         █████   ██   ██ ██          ██      ███████ ██████       | # 
# |         ██      ██   ██ ██          ██      ██   ██ ██   ██      | # 
# |         ██      ██████   ██████     ███████ ██   ██ ██████       | # 
# |                                                                  | # 
# |                                                                  | # 
# |              ██████ ██████   █████  ███████ ████████             | # 
# |             ██      ██   ██ ██   ██ ██         ██                | # 
# |             ██      ██████  ███████ █████      ██                | # 
# |             ██      ██   ██ ██   ██ ██         ██                | # 
# |              ██████ ██   ██ ██   ██ ██         ██                | # 
# |                                                                  | # 
# |                                                                  | # 
# | ██████  ██    ██ ███    ██  █████  ███    ███ ██  ██████ ███████ | # 
# | ██   ██  ██  ██  ████   ██ ██   ██ ████  ████ ██ ██      ██      | # 
# | ██   ██   ████   ██ ██  ██ ███████ ██ ████ ██ ██ ██      ███████ | # 
# | ██   ██    ██    ██  ██ ██ ██   ██ ██  ██  ██ ██ ██           ██ | # 
# | ██████     ██    ██   ████ ██   ██ ██      ██ ██  ██████ ███████ | # 

                                                                 
#                                                                  
# Pseudo 5 DOF Craft Dynamics cover both missile and aircraft
#   Version --5DOF
#   Created by Hong Daseon
#   Input  : Complex coordinate of projected Look-angle
#   Output : desired angular velocity in body coordinate itself

import DaseonTypesNtf as Daseon
from pyquaternion import Quaternion
from DaseonTypesNtf import Vector3, DCM6DOF
import math as m
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import time
import copy
import pdb

#Missile coordinate trans Model
Debug = False
gAcc = 0. #9.806


class Craft:
    
    def __init__(self, scavel, initPosN, initAttN, dt):
        self.scavel         = scavel
        self.bodyVelDir     = Vector3(self.scavel,0.,0.)
        self.pos            = initPosN
        self.datt           = Vector3(0.,0.,0.)
        self.att            = initAttN
        self.Cnb            = DCM6DOF(self.att)

        self.ControllerActZ = Daseon.SecondOrder(5, 0.75, dt)
        self.ControllerActY = Daseon.SecondOrder(5, 0.75, dt)
        self.ControllerActX = Daseon.SecondOrder(5, 0.75, dt)
        #self.pqrD           = Vector3(0.,0.,0.)
        self.dpos           = self.Cnb.rotate(self.bodyVelDir)
        self.AngRateB       = Vector3(0.,0.,0.)
        
        self.dt             = dt

        self.reset_flag     = True

        self.IntegAtt_x     = Daseon.Integrator(self.att.x, dt)
        self.IntegAtt_y     = Daseon.Integrator(self.att.y, dt)
        self.IntegAtt_z     = Daseon.Integrator(self.att.z, dt)
        
        self.IntegPos_x     = Daseon.Integrator(self.pos.x, dt)
        self.IntegPos_y     = Daseon.Integrator(self.pos.y, dt)
        self.IntegPos_z     = Daseon.Integrator(self.pos.z, dt)

    def simulate(self, pqrD):
        self.AngRateB.x     = self.ControllerActX.step(pqrD.x)
        self.AngRateB.y     = self.ControllerActY.step(pqrD.y)
        self.AngRateB.z     = self.ControllerActZ.step(pqrD.z)
        self.datt           = self.Cnb.angRateTransformation(self.AngRateB, self.att)

        self.att.x          = self.IntegAtt_x.step(self.datt.x)
        self.att.y          = self.IntegAtt_y.step(self.datt.y)
        self.att.z          = self.IntegAtt_z.step(self.datt.z)
        
        self.Cnb.update(self.att)

        self.dpos           = self.Cnb.rotate(self.bodyVelDir)

        self.pos.x          = self.IntegPos_x.step(self.dpos.x)
        self.pos.y          = self.IntegPos_y.step(self.dpos.y)
        self.pos.z          = self.IntegPos_z.step(self.dpos.z)
        return self.dpos, self.pos

    def reset(self, _scavel, _initPosN, _initAttN, reset_flag):
        self.scavel         = _scavel
        self.bodyVelDir     = Vector3(self.scavel,0.,0.)
        self.pos            = _initPosN
        self.att            = _initAttN
        self.acc            = Vector3(0.,0.,0.) 

        self.Cnb.reset(_initAttN)
        self.ControllerActZ.reset()
        self.ControllerActY.reset()
        self.ControllerActX.reset()
        self.IntegAtt_x.reset(_initAttN.x)
        self.IntegAtt_y.reset(_initAttN.y)
        self.IntegAtt_z.reset(_initAttN.z)
        self.IntegPos_x.reset(_initPosN.x)
        self.IntegPos_y.reset(_initPosN.y)
        self.IntegPos_z.reset(_initPosN.z)

        self.dpos           = self.Cnb.rotate(self.bodyVelDir)
        
        #print('just after reset : ',self.Qnb)
        self.reset_flag     = reset_flag
    
    def __str__(self):
        nowpos = 'x : '+ format(self.pos.x,".2f")+ ' y : '+ format(self.pos.y,".2f")+ ' z : '+ format(self.pos.z,".2f")

        return nowpos

class Seeker:
    #prevR = 0
    def __init__(self, Yo, Tu):
        self.VAqztor    = Daseon.Differntiator(Yo.dt)
        self.Rvec       = Tu.pos - Yo.pos
        self.Vvec       = Tu.dpos - Yo.dpos
        self.Vrel       = -self.Vvec.mag
        self.direcVec   = Yo.dpos

        self.Tu         = Tu
        self.Yo         = Yo
        
        self.impactR    = 9999999

        Lookz, Looky    = self.azimNelev(Yo.Cnb.rotate(self.Rvec, stat='inv'))
        self.Look       = Vector3(0., Looky, Lookz)
        self.pLook      = copy.deepcopy(self.Look)
        self.ppLook     = copy.deepcopy(self.pLook)

        self.firstrun   = True

        self.prev_Rm  = Vector3(9999999, 9999999, 9999999)

        self.t2go       = 600

    def angle(self, vec1, vec2):
        dot = vec1[0]*vec2[0] + vec1[1]*vec2[1]      # dot product
        det = vec1[0]*vec2[1] - vec2[0]*vec1[1]      # determinant
        return m.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)

    def azimNelev(self, vec):
        azim = m.atan2( vec.y, vec.x)
        elev = m.atan2( -vec.z, m.sqrt( vec.x**2 + vec.y**2))
        return azim, elev

    def seek(self, t):
        def normL(LOSval):
            return (LOSval)/3.14

        def normLd(LOSdotval):
            return (LOSdotval)*10
        
        def normVm(Vval):
            return Vval/600

        def normLk(Vval):
            return Vval/1.57
        #pdb.set_trace()
        self.t2go       = 0
        #print('in seek : ',self.Missile.Qnb)
        self.Rvec       = self.Tu.pos - self.Yo.pos
        self.Vvec       = self.Tu.dpos - self.Yo.dpos
        self.Vrel       = self.VAqztor.step(self.Rvec.mag)
        self.direcVec   = self.Yo.dpos
                
        LOSz, LOSy      = self.azimNelev(self.direcVec)
        self.LOS        = Vector3(0.,LOSy, LOSz)
        if t == 0 : 
            self.prevLOS = copy.deepcopy(self.LOS)
            #print('t=0 detected')
        self.ppLook     = copy.deepcopy(self.pLook)
        self.pLook      = copy.deepcopy(self.Look) 
        Lookz, Looky    = self.azimNelev(self.Yo.Cnb.rotate(self.Rvec,'inv'))
        self.Look       = Vector3(0., Looky, Lookz)

        RjxVj = np.cross(self.Rvec.vec, self.Vvec.vec)
        RjdRj = np.dot(self.Rvec.vec, self.Rvec.vec)
        Ldotn = RjxVj/RjdRj
        
        Ldotb = self.Yo.Cnb.rotate(Vector3.cast(Ldotn),'inv')
        self.dLOS = Ldotb
        self.Yo.reset_flag = False
        Vvecb = self.Yo.Cnb.rotate(Vector3.cast(self.Vvec.vec), 'inv')
        return self.Rvec.mag, self.Look, self.dLOS, self.Yo.scavel,\
                                                    np.array([  normLk(self.ppLook.y), normLk(self.pLook.y), normLk(self.Look.y),\
                                                                normLk(self.ppLook.z), normLk(self.pLook.z), normLk(self.Look.z)])
                                                                                
    def newStepStarts(self, t):
        if t != 0:
            self.prevLOS = copy.deepcopy(self.LOS)
            #print('prevValEngaged')

    def spit_reward(self, acc):
        
        OOR         = (self.Look.y < -1.57)|(self.Look.y > 1.57)|(self.Look.z < -1.57)|(self.Look.z > 1.57)|(self.Rvec.mag>20000)  # Out of range
        if OOR:
            Rf_1 = self.prev_Rm
            Rf = self.Yo.pos
            
            R3 = Rf - Rf_1
            A = R3
            B = (self.Tu.pos - Rf_1) - R3
            
            if self.Rvec.mag < 50:

                self.impactR = (Vector3.cast(np.cross(A.vec,B.vec)).mag) / A.mag 

            else:
                self.impactR = self.Rvec.mag

            rwdR = copy.deepcopy(self.impactR)
            
            if Debug : pdb.set_trace()
            print(rwdR)
        else:
            self.prev_Rm = copy.deepcopy(self.Yo.pos)
            rwdR = copy.deepcopy(self.Rvec.mag)

        


        hit         = (self.impactR <2)

        step_reward  = np.array([-(acc.y**2),-(acc.z**2)])
         #0.01*-Rdot - 3*abs(self.LOS) - 1.2*abs(self.LOS)*500*abs(Ldot) + (2/(self.R / 8000))**2.5 - (self.R<10000)*self.R/5000 #-1000*abs(Ldot)   -self.R/10000  # - self.R/100
        
        mc_reward    =  - (rwdR)
        
        #reward = (not OOR)*reward -OOR*25
        
        return step_reward, mc_reward, (OOR), hit

