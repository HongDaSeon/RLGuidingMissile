#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import DaseonTypesNtf_V3 as Daseon
from DaseonTypesNtf_V3 import Vector3, DCM5DOF, lineEQN
import time, random
import math as m
import matplotlib.pyplot as plt

class LidarModule:

    def __init__(self, Missile, beamQuant, beamLength, beamViewDeg):
        if(beamQuant<2):
            print("beam quantity should be greater than 1, Session shut down")
            quit()
        self.MissileObject  = Missile
        self.beamQuant      = beamQuant
        self.beamLength     = beamLength
        self.beamView       = beamViewDeg
        self.beamTips       = []

        self.reset(self.beamQuant, self.beamLength, self.beamView)

    def StepNSense(self, BeamStartPos, OverralOrient, battlefield):
        #print(OverralOrient/m.pi*180)
        OverralRotator  = DCM5DOF(Vector3(0,0,OverralOrient))
        lidarInfo       = []
        for beamCNT in range(self.beamQuant):
            tipCoord    = OverralRotator.rotate(self.beamTips[beamCNT]) + BeamStartPos
            rootCoord   = BeamStartPos
            thisBeam    = lineEQN(rootCoord, tipCoord)
            intersecS   = [thisBeam.pnt2+battlefield.Missile.pos]
            distS       = [999999]
            for sttCnt in battlefield.Structs:
                intersec, dist = sttCnt.lidarReflection(thisBeam)
                intersecS.append(intersec)
                distS.append(dist)
            MinArg      = np.argmin(distS)
            dist        = distS[MinArg]
            intersec    = intersecS[MinArg]
            lidarInfo.append([dist, intersec])
        return np.array(lidarInfo)

    def reset(self, beamQuant, beamLength, beamView):
        #Define lidar beam tip when the missile is at 0,0 and the orientation of 0
        self.beamQuant      = beamQuant
        self.beamLength     = beamLength
        self.beamView       = beamView
        SmalAng             = self.beamView/(self.beamQuant-1)
        SmalRotationMAT     = DCM5DOF(Vector3(0,0,SmalAng))
        LargRotationMAT     = DCM5DOF(Vector3(0,0,-self.beamView/2))
        beamTips            = []
        for lidarCnt in range(self.beamQuant):
            if(lidarCnt==0):
                beamTips.append(Vector3(self.beamLength,0))
            else:
                beamTips.append(SmalRotationMAT.rotate(beamTips[-1], 'inv'))
        for lidarCnt in range(self.beamQuant):
            beamTips[lidarCnt] = LargRotationMAT.rotate(beamTips[lidarCnt], 'inv')
        self.beamTips = beamTips
        #beamTips = np.array(beamTips)


