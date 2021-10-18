#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import math as m
import DaseonTypesNtf_V3 as Daseon
from DaseonTypesNtf_V3 import Vector3, DCM5DOF
import random as rd

class STT

    def __init__(self, AspectRatio, sizeRange, TargetPos, MissileMargin):
        width_side  = sizeRange[0] + rd.random()*(sizeRange[1]-sizeRange[0])
        self.sides  = (width_side, width_side*AspectRatio)
        targetDist  = TargetPos.mag
        defined     = False
        while(not defined)
            DircCandi           = 2*m.pi*rd.random()
            DistCandi           = targetDist*rd.random()
            OrieCandi           = 0.5*m.pi*rd.random() # 90도까지만
            CenterPosCandi      = Vector3(DistCandi*m.cos(DircCandi), DistCandi*m.sin(DircDistCandi), 0)
            RotationMat         = DCM5DOF( Vector3(0,0,OrieCandi) )
            areaREF             = self.sides[0]*self.sides[1]
            vtx0                = CenterPosCandi + RotationMat.rotate(Vector3(self.sides[0]/2, self.sides[1]/2, 0),'inv')
            vtx1                = CenterPosCandi + RotationMat.rotate(Vector3(-self.sides[0]/2, self.sides[1]/2, 0),'inv')
            vtx2                = CenterPosCandi + RotationMat.rotate(Vector3(-self.sides[0]/2, -self.sides[1]/2, 0),'inv')
            vtx3                = CenterPosCandi + RotationMat.rotate(Vector3(self.sides[0]/2, -self.sides[1]/2, 0),'inv')
            
            targetwiseDefined   = not isInside(TargetPos, vtx0, vtx1, vtx2, vtx3, areaREF)
            missilewiseDefined  = not isInside(Vector3(0,0,0), vtx0, vtx1, vtx2, vtx3, areaREF)

            defined             = targetMinMaxString & missilewiseDefined

        self.pos    = posCandi
        self.orient
        self.vertices = (vtx0,vtx1,vtx2,vtx3)

    def __repr__(self):
        return  "Center Pos : " + "{:.2f}".format(self.pos.x)+", "+"{:.2f}".format(self.pos.y)+\
                "\n Radius : "+"{:.2f}".format(self.radius)

    def lidarReflection(self, beam):
                
        

    def isInside(theObject, vtx0, vtx1, vtx2, vtx3, areaREF):
        AreaT01             = TriAreaVtx(TargetPos, vtx0, vtx1)
        AreaT12             = TriAreaVtx(TargetPos, vtx1, vtx2)
        AreaT23             = TriAreaVtx(TargetPos, vtx2, vtx3)
        AreaT34             = TriAreaVtx(TargetPos, vtx3, vtx4)
        return ( (AreaT01 + AreaT12 + AreaT23 + AreaT34) <= (areaREF+1) )

    def TriAreaVtx(p1, p2, p3):
        MAT     = np.array([ [p1.x, p1.y, 1], [p2.x, p2.y, 1], [p3.x, p3.y, 1]])
        return 0.5*abs(np.linalg.det(MAT))