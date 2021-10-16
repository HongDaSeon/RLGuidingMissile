#!/usr/bin/env python
#-*- coding: utf-8 -*-


# Types and useful classes Defined by Daseon

from pyquaternion import Quaternion
import math as m
import numpy as np
import torch
import vpython as vp


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


class ASCIIart:
    def FDCLAB():
        print(   "\t\t\t   ____  ____   ___  \
                \n\t\t\t  (  __)(    \ / __) \
                \n\t\t\t   ) _)  ) D (( (__  \
                \n\t\t\t  (__)  (____/ \___) \
                \n\t\t\t   __     __   ____  \
                \n\t\t\t  (  )   / _\ (  _ \ \
                \n\t\t\t  / (_/\/    \ ) _ ( \
                \n\t\t\t  \____/\_/\_/(____/")

    def LearningStarts():
        print(   "    __                          _                _____ __             __ \
                \n   / /   ___  ____ __________  (_)___  ____ _   / ___// /_____ ______/ /______ \
                \n  / /   / _ \/ __ `/ ___/ __ \/ / __ \/ __ `/   \__ \/ __/ __ `/ ___/ __/ ___/ \
                \n / /___/  __/ /_/ / /  / / / / / / / / /_/ /   ___/ / /_/ /_/ / /  / /_(__  )  \
                \n/_____/\___/\__,_/_/  /_/ /_/_/_/ /_/\__, /   /____/\__/\__,_/_/   \__/____/   \
                \n                                    /____/                                     ")

    def DDDMissileRL():
        print("        _____ ____     __  ____           _ __        ____  __ \
             \n       |__  // __ \   /  |/  (_)_________(_) /__     / __ \/ / \
             \n        /_ </ / / /  / /|_/ / / ___/ ___/ / / _ \   / /_/ / /  \
             \n      ___/ / /_/ /  / /  / / (__  |__  ) / /  __/  / _, _/ /___\
             \n     /____/_____/  /_/  /_/_/____/____/_/_/\___/  /_/ |_/_____/")

    def VisualPython():
        print("   _    ___                  __   ____        __  __ \
            \n  | |  / (_)______  ______ _/ /  / __ \__  __/ /_/ /_  ____  ____ \
            \n  | | / / / ___/ / / / __ `/ /  / /_/ / / / / __/ __ \/ __ \/ __ \ \
            \n  | |/ / (__  ) /_/ / /_/ / /  / ____/ /_/ / /_/ / / / /_/ / / / / \
            \n  |___/_/____/\__,_/\__,_/_/  /_/    \__, /\__/_/ /_/\____/_/ /_/  \
            \n                                    /____/                        ")

    def Unity_ROS():
        print("      __  __      _ __              ____  ____  _____ \
                \n  / / / /___  (_) /___  __      / __ \/ __ \/ ___/\
                \n / / / / __ \/ / __/ / / /_____/ /_/ / / / /\__ \ \
                \n/ /_/ / / / / / /_/ /_/ /_____/ _, _/ /_/ /___/ / \
                \n\____/_/ /_/_/\__/\__, /     /_/ |_|\____//____/  \
                \n                 /____/                           ")

    def ReplayStarts():
        print(    "    ____             __               _____ __             __      \
                 \n   / __ \___  ____  / /___ ___  __   / ___// /_____ ______/ /______\
                 \n  / /_/ / _ \/ __ \/ / __ `/ / / /   \__ \/ __/ __ `/ ___/ __/ ___/\
                 \n / _, _/  __/ /_/ / / /_/ / /_/ /   ___/ / /_/ /_/ / /  / /_(__  ) \
                 \n/_/ |_|\___/ .___/_/\__,_/\__, /   /____/\__/\__,_/_/   \__/____/  \
                 \n          /_/            /____/                                    ")

class Vector3:

    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
    
    def cast(lllist):
        return Vector3(lllist[0], lllist[1], lllist[2])

    def __repr__(self):
        return str(self.vec)

    def __add__(self, other):
        A = self.vec
        B = other.vec
        if len(A) != len(B):
            print("Error : Vector size"+str(len(A)+" and "+str(len(B)) + "missmatch" ))
            return None
        return Vector3(A[0]+B[0], A[1]+B[1], A[2]+B[2])

    def __sub__(self, other):
        A = self.vec
        B = other.vec
        if len(A) != len(B):
            print("Error : Vector size"+str(len(A)+" and "+str(len(B)) + "missmatch" ))
            return None
        return Vector3(A[0]-B[0], A[1]-B[1], A[2]-B[2])
    
    def __neg__(self):
        return Vector3(-self.x, -self.y, -self.z)
        
    @property
    def vec(self):
        return np.array([self.x,self.y,self.z])
    @vec.setter
    def vec(self, listvec):
        self.x = listvec[0]
        self.y = listvec[1]
        self.z = listvec[2]

    @property
    def VPvec(self):
        return vp.vec(self.x, self.y, self.z)

    @property
    def zyxvec(self):
        return np.array([self.z,self.y,self.x])

    @property
    def mag(self):
        return np.sqrt(sum(self.vec**2))

    @property
    def direction(self):
        return Vector3.cast(self.vec / np.sqrt(sum(self.vec**2)))


class Integrator:
    def __init__(self, IC, dt):
        self.IC             = IC
        self.dt             = dt
        self.prev           = IC
        self.previnput      = 0
        self.Result         = IC
    
    def __repr__(self):
        return str(self.Result)

    def step(self, nowval):
        self.Result         = self.prev + ( (nowval)*self.dt + (nowval + self.previnput)/2*self.dt )/2
        self.prev           = self.Result
        self.previnput      = nowval
        return self.Result

    def reset(self, IC):
        self.prev           = IC
        self.Result         = IC

class Differntiator:
    def __init__(self, dt):
        self.dt             = dt
        self.previnput      = 0
        self.Result         = 0
        self.sofarNorun     = True
    
    def __repr__(self):
        return str(self.Result)

    def step(self, nowval):
        if self.sofarNorun == True:
            self.sofarNorun = False
            self.Result = 0
            self.previnput      = nowval
        else:
            self.Result         = (nowval - self.previnput)/self.dt
            self.previnput      = nowval
        return self.Result

    def reset(self):
        self.previnput           = 0
        self.Result         = 0
        self.sofarNorun     = True

class FirstOrder:
    def __init__(self, num, tau, K, dt):
        self.xinteg         = Integrator(0, dt)
        self.yinteg         = Integrator(0, dt)
        self.prevy          = 0
        self.num, self.tau  = num, tau
        self.K              = K
        self.dt             = dt
    
    def step(self, cmd):
        y = ( self.num*self.xinteg.step(cmd) - self.K*self.yinteg.step(self.prevy) )/self.tau
        self.prevy = y
        return y

    def reset(self):
        self.xinteg         = Integrator(0, self.dt)
        self.yinteg         = Integrator(0, self.dt)
        self.prevy          = 0

class SecondOrder:
    def __init__(self, omega, zeta, dt):
        self.x2integ_1      = Integrator(0, dt)
        self.x2integ_2      = Integrator(0, dt)
        self.yinteg         = Integrator(0, dt)
        self.y2integ_1      = Integrator(0, dt)
        self.y2integ_2      = Integrator(0, dt)
        self.prevy          = 0
        self.omega          = omega
        self.zeta           = zeta
        self.dt             = dt
    
    def step(self, cmd):
        y = self.omega**2 * self.x2integ_2.step(self.x2integ_1.step(cmd))\
            - 2*self.zeta*self.omega * self.yinteg.step(self.prevy)\
            - self.omega**2 * self.y2integ_2.step(self.y2integ_1.step(self.prevy))
        self.prevy = y
        return y

    def reset(self):
        self.x2integ_1      = Integrator(0, self.dt)
        self.x2integ_2      = Integrator(0, self.dt)
        self.yinteg         = Integrator(0, self.dt)
        self.y2integ_1      = Integrator(0, self.dt)
        self.y2integ_2      = Integrator(0, self.dt)
        self.prevy          = 0

class DCM5DOF:
    # Put the Cnb Val
    def __init__(self, att):
        self.psi    = att.z
        self.the    = att.y
        self.dof    = 5
        Cps     = m.cos(att.z)
        Cth     = m.cos(att.y)
        Sps     = m.sin(att.z)
        Sth     = m.sin(att.y)
        self.MAT = np.array([   [Cps*Cth,  -Sps,   Cps*Sth],\
                                [Sps*Cth,   Cps,   Sps*Sth],\
                                [-Sth,        0,       Cth] ])

    def update(self, att):
        Cps     = m.cos(att.z)
        Cth     = m.cos(att.y)
        Sps     = m.sin(att.z)
        Sth     = m.sin(att.y)
        self.MAT = np.array([   [Cps*Cth,  -Sps,   Cps*Sth],\
                                [Sps*Cth,   Cps,   Sps*Sth],\
                                [-Sth,        0,       Cth] ])
    
    def rotate(self, sacri, stat='gen'):
        if stat == 'inv':
            Res = np.matmul(self.MAT.T, np.array(sacri.vec).T).tolist()
        elif stat == 'gen':
            Res = np.matmul(self.MAT, np.array(sacri.vec).T).tolist()
        else:
            print('Make sure if the stat name is right')
        return Vector3.cast(Res)

    def reset(self,att):
        self.psi    = att.z
        self.the    = att.y
        self.dof    = 5
        Cps     = m.cos(att.z)
        Cth     = m.cos(att.y)
        Sps     = m.sin(att.z)
        Sth     = m.sin(att.y)
        self.MAT = np.array([   [Cps*Cth,  -Sps,   Cps*Sth],\
                                [Sps*Cth,   Cps,   Sps*Sth],\
                                [-Sth,        0,       Cth] ])
                                
    def __repr__(self):
        return self.MAT

class DCM6DOF:
    # Put the Cnb Val
    def __init__(self, att):
        self.dof    = 6
        Cps     = m.cos(att.z)
        Cth     = m.cos(att.y)
        Cph     = m.cos(att.x)
        Sps     = m.sin(att.z)
        Sth     = m.sin(att.y)
        Sph     = m.sin(att.x)
        self.MAT = np.array([   [ Cps*Cth, Cps*Sph*Sth - Cph*Sps, Sph*Sps + Cph*Cps*Sth],\
                                [ Cth*Sps, Cph*Cps + Sph*Sps*Sth, Cph*Sps*Sth - Cps*Sph],\
                                [    -Sth,               Cth*Sph,               Cph*Cth]])

    def update(self, att):
        Cps     = m.cos(att.z)
        Cth     = m.cos(att.y)
        Cph     = m.cos(att.x)
        Sps     = m.sin(att.z)
        Sth     = m.sin(att.y)
        Sph     = m.sin(att.x)
        self.MAT = np.array([   [ Cps*Cth, Cps*Sph*Sth - Cph*Sps, Sph*Sps + Cph*Cps*Sth],\
                                [ Cth*Sps, Cph*Cps + Sph*Sps*Sth, Cph*Sps*Sth - Cps*Sph],\
                                [    -Sth,               Cth*Sph,               Cph*Cth]])
    
    def rotate(self, sacri, stat='gen'):
        if stat == 'inv':
            Res = np.matmul(self.MAT.T, np.array(sacri.vec).T).tolist()
        elif stat == 'gen':
            Res = np.matmul(self.MAT, np.array(sacri.vec).T).tolist()
        else:
            print('Make sure if the stat name is right')
        return Vector3.cast(Res)

    def angRateTransformation(self, scri, att):
        # pqr to phi theta psi dot
        Cps     = m.cos(att.z)
        Cth     = m.cos(att.y)
        Cph     = m.cos(att.x)
        Sps     = m.sin(att.z)
        Sth     = m.sin(att.y)
        Sph     = m.sin(att.x)
        Tth     = m.tan(att.y)
        RotMat  = np.array( [ [1, Sph*Tth, Cph*Tth],
                            [0, Cph,     -Sth   ],
                            [0, Sph/Cth, Cph/Cth]])

        return Vector3.cast(RotMat@scri.vec.T)      

    def reset(self, att):
        self.dof    = 6
        Cps     = m.cos(att.z)
        Cth     = m.cos(att.y)
        Cph     = m.cos(att.x)
        Sps     = m.sin(att.z)
        Sth     = m.sin(att.y)
        Sph     = m.sin(att.x)
        self.MAT = np.array([   [ Cps*Cth, Cps*Sph*Sth - Cph*Sps, Sph*Sps + Cph*Cps*Sth],\
                                [ Cth*Sps, Cph*Cps + Sph*Sps*Sth, Cph*Sps*Sth - Cps*Sph],\
                                [    -Sth,               Cth*Sph,               Cph*Cth]])
                                
    def __repr__(self):
        return self.MAT

