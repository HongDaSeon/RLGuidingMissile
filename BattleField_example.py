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
# Pseudo 6 DOF BATTLEFIELD
#   Version --6DOFfor inertial
#   Created by Hong Daseon

import numpy as np
import time
import random as rd
import DaseonTypesNtf as Daseon
import pyquaternion as Quaternion
from DaseonTypesNtf import Vector3, DCM5DOF
import CraftDynamics
import pdb
import torch
import copy
import math as m
import pygame
import sys

class BATTLEFIELD:

    def __init__(self, dt, radius, halfheight, distance, M_spd, F_spd, t_max):
        AttOrigin       = Vector3(0.,0.,m.pi/2.)
        M1_initpos      = Vector3(0.,0.,0.)
        M2_initpos      = Vector3(0.,distance,0.)
        self.tMax       = t_max
        self.radius     = radius
        self.halfheight = halfheight
        self.dt         = dt
        self.Missile1   = CraftDynamics.Craft(M1_spd, M1_initpos, AttOrigin, dt)
        self.Missile2   = CraftDynamics.Craft(M2_spd, M2_initpos, AttOrigin, dt)
        self.Target     = CraftDynamics.Craft(T_spd, T_initpos, initAttN, dt)
        self.M1Seeker   = CraftDynamics.Seeker(self.Missile1, self.Target)
        self.M2Seeker   = CraftDynamics.Seeker(self.Missile2, self.Target)
        self.SpdCalc    = Daseon.Differntiator(self.dt)
        self.AprchSp    = 0.
        self.prevRvec   = Vector3(0,0,0)
        self.prevVvec   = Vector3(0,0,0)
        self.done       = False
        
    def __repr__(self):
        return str('gogogo')

    def init_picker(max_g_missile1, max_g_missile2):
        fieldR      = 5000
        fieldH      = 1000
        distance    = 50 + rd.random()*300 
        M1spd       = 200 + rd.random()*100
        M2spd       = 200 + rd.random()*100
        Tspd        = (M1spd+M2spd)/2 - 50 - rd.random()*90
        M1Amax      = max_g_missile*9.806/Mspd
        M2Amax      = max_g_fighter*9.806/Fspd
        return fieldR, fieldH, distance, Mspd, Fspd, MAmax, FAmax  

    def Catesian2Cylinder(self, CatCoor):
        r               = m.sqrt(CatCoor.x**2 + CatCoor.y**2)
        t               = m.atan2(CatCoor.y, CatCoor.x)
        z               = CatCoor.z 
        CylCoor         = Vector3(r,t,z)
        return CylCoor

    def spatialRestrictionCheck(self, sacrifice):
        cylPos = self.Catesian2Cylinder(sacrifice.pos)
        isinBattleField = (cylPos.x < self.radius) & (abs(cylPos.z) < self.halfheight)
        return isinBattleField

    def CalcZEM(self, Rvec, Vvec, dt):
        t_minimum        = np.clip( -np.dot(Rvec.vec, Vvec.vec) / np.dot(Vvec.vec, Vvec.vec), 0, dt)
        VecMinimum       = Vector3.cast(Rvec.vec + t_minimum * Vvec.vec)
        return VecMinimum.mag

    def step(self, M_pqrD, F_pqrD, t, forcedQuit, TDratio=0.25, MCratio=0.75):
        self.Missile1.simulate(M_pqrD1)
        self.Missile2.simulate(M_pqrD2)
        self.M1Seeker.seek(t)
        self.M2Seeker.seek(t)
        self.AprchSp    = self.SpdCalc.step(self.MSeeker.Rvec.mag)
        ZEM             = 9999999
        M1SpatialExceed = not self.spatialRestrictionCheck(self.Missile1)
        M2SpatialExceed = not self.spatialRestrictionCheck(self.Missile2)
        MOutOfSight     = (self.MSeeker.Look.y < -1.57)|(self.MSeeker.Look.y > 1.57)|\
                          (self.MSeeker.Look.z < -1.57)|(self.MSeeker.Look.z > 1.57)
        timeout         = t >= self.tMax-1
        MissTDRWD1      = (-self.MSeeker.Vrel)/100*100
        MissTDRWD2      = ( self.FSeeker.Vrel)/100*100
        MissMCRWD1      = 0.
        MissMCRWD2      = 0.
        if (MOutOfSight | timeout):
            print(MOutOfSight*'MOutOfSight ' + timeout*'timeout ')
            ZEM = self.CalcZEM(self.prevRvec, self.prevVvec, self.dt)
            MissMCRWD   = -75*(m.log10(ZEM)/m.log10(m.sqrt(10000**2 + 2000**2)))
            FighMCRWD   = +75*(m.log10(ZEM)/m.log10(m.sqrt(10000**2 + 2000**2)))
            #print(ZEM)
            self.done   = True
        MRWD1 = MissTDRWD1 + MissMCRWD1
        MRWD2 = MissTDRWD2 + MissMCRWD2
        self.prevRvec   = copy.deepcopy(self.MSeeker.Rvec)
        self.prevVvec   = copy.deepcopy(self.MSeeker.Vvec)
        return MRWD, FRWD, ZEM, self.done

    def reset(self, distance, M_spd, F_spd, reset_flag):
        AttOrigin       = Vector3(0.,0.,m.pi/2.)
        M_initpos       = Vector3(0.,0.,0.)
        F_initpos       = Vector3(0.,distance,0.)
        self.Missile.reset(M_spd, M_initpos, AttOrigin, reset_flag)
        self.Fighter.reset(F_spd, F_initpos, AttOrigin, reset_flag)
        #del self.MSeeker, self.FSeeker
        self.MSeeker    = CraftDynamics.Seeker(self.Missile, self.Fighter)
        self.FSeeker    = CraftDynamics.Seeker(self.Fighter, self.Missile)
        self.SpdCalc.reset()
        self.AprchSp    = 0.
        self.done       = False
        
def Test(dt, timeScale, cmdScale):
    battleField_1 = BATTLEFIELD(dt, 5000, 1000, 500, 250, 200, 500)
    pygame.init()
    pygame.joystick.init()

    try:
        controller = pygame.joystick.Joystick(0)
        controller.init() # init instance
        print ("Joystick_Paired: {0}".format(controller.get_name()))
    except pygame.error:
	    print ("None of or Invalid joystick connected")
    
    display_width = 720
    display_height = 720
    gameDisplay = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption('EnvironmentTestEnvironment')
    C_Enemy     = (200, 100, 200)
    C_Enemyp    = (150, 150, 200)
    C_Enemypp   = (100, 200, 200)
    C_Headn     = (10,  100, 200)
    C_lGrey     = (200, 200, 200)
    centre      = (display_width/2, display_height/2)
    
    def azimelev2cartesianProjection(azimelev):
        y = m.sin(azimelev.z)*m.cos(azimelev.y)
        z = -m.sin(azimelev.y)
        return Vector3(0, y, z)

    def visLoop():
        Exit = False
        H_spotx = display_width/2
        H_spoty = display_height/2
        prevstepEndtime = 0
        t = 0
        while not Exit:
            while time.time() <= (prevstepEndtime+(dt*timeScale)): # 실시간맞추기
                pass 
            M_rotRatex  = 0
            M_rotRatey  = 0
            M_rotRatex  = controller.get_axis(3)*cmdScale
            M_rotRatey  = controller.get_axis(4)*cmdScale
            F_rotRatex  = 0
            F_rotRatey  = 0
            F_rotRatex  = controller.get_axis(0)*cmdScale
            F_rotRatey  = controller.get_axis(1)*cmdScale
            McmdVec     = Vector3.cast([0, M_rotRatey, M_rotRatex])
            FcmdVec     = Vector3.cast([0, F_rotRatey, F_rotRatex])
            zeroVec     = Vector3(0,0,0)
            MR,FR,ZEM,done  = battleField_1.step(McmdVec, FcmdVec, t, False, 1, 1)
            azimelev    = battleField_1.MSeeker.Look
            print(azimelev)
            pazimelev   = battleField_1.MSeeker.pLook
            ppazimelev  = battleField_1.MSeeker.ppLook
            prjcur      = azimelev2cartesianProjection(azimelev)
            prjp        = azimelev2cartesianProjection(pazimelev)
            prjpp       = azimelev2cartesianProjection(ppazimelev)
            #print(battleField_1.MSeeker.Rvec.mag)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print('goodbye')
                    sys.exit()
                if event.type == pygame.JOYAXISMOTION:  # Joystick
                    pass
                if event.type == pygame.JOYBUTTONDOWN:  
                    print("Joystick Button pressed")
            # 시각화 처리
            Enemylook   = ((prjcur.y+1)*360, (prjcur.z+1)*360)
            Enemylookp  = ((prjp.y+1)*360, (prjp.z+1)*360)
            Enemylookpp = ((prjpp.y+1)*360, (prjpp.z+1)*360)
            pygame.draw.rect(gameDisplay, (0,0,0), [0, 0 , display_width, display_height])
            pygame.draw.circle(gameDisplay, C_lGrey, centre, 360, width=1)
            pygame.draw.circle(gameDisplay, C_lGrey, centre, 240, width=1)
            pygame.draw.circle(gameDisplay, C_lGrey, centre, 180, width=1)
            pygame.draw.circle(gameDisplay, C_lGrey, centre, 120, width=1)
            pygame.draw.lines(gameDisplay, C_lGrey, True, [(display_width/2, 0), (display_width/2, display_height)])
            pygame.draw.lines(gameDisplay, C_lGrey, True, [(0, display_height/2), (display_width, display_height/2)])
            pygame.draw.circle(gameDisplay, C_Headn, centre, 10)
            pygame.draw.circle(gameDisplay, C_Enemypp, Enemylookpp, 5)
            pygame.draw.circle(gameDisplay, C_Enemyp, Enemylookp, 5)
            pygame.draw.circle(gameDisplay, C_Enemy, Enemylook, 5)
            pygame.display.update()

            if done:
                print("Done with ZEM of ", ZEM)
                break

            # 최신화
            t = t+dt
            prevstepEndtime = time.time()
    while True:
        for event in pygame.event.get():
            if controller.get_axis(5)>=0.5:
                battleField_1.reset(500, 250, 200, True)
                print("RESET")
                visLoop()

if __name__ == "__main__":
    Test(0.01, 1, 1)
    pygame.joystick.quit()


