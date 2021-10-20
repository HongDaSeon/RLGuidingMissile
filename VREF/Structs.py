#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import math as m
import DaseonTypesNtf_V3 as Daseon
from DaseonTypesNtf_V3 import Vector3, DCM5DOF
import random as rd



class STT:

    def __init__(self, AspectRatio, sizeRange, TargetPos, Margin):
        width_side  = sizeRange[0] + rd.random()*(sizeRange[1]-sizeRange[0])
        self.sides  = (width_side, width_side*AspectRatio)
        targetDist  = TargetPos.mag
        defined     = False
        while(not defined):
            DircCandi           = 2*m.pi*rd.random()
            DistCandi           = targetDist*rd.random()
            OrieCandi           = 0.5*m.pi*rd.random() # 90도까지만
            CenterPosCandi      = Vector3(DistCandi*m.cos(DircCandi), DistCandi*m.sin(DircCandi), 0)
            
            self.areaREF        = self.sides[0]*self.sides[1]
            vtx0, vtx1, vtx2, vtx3  = getVertices(CenterPosCandi, self.sides, OrieCandi)

            BigSides            = [self.sides[0]+2*Margin, self.sides[1]+2*Margin]
            BigArea             = BigSides[0] * BigSides[1]
            VTX0, VTX1, VTX2, VTX3  = getVertices(CenterPosCandi, BigSides, OrieCandi)

            targetwiseDefined   = not isInside(TargetPos, VTX0, VTX1, VTX2, VTX3, BigArea)
            missilewiseDefined  = not isInside(Vector3(0,0,0), VTX0, VTX1, VTX2, VTX3, BigArea)

            defined             = targetwiseDefined & missilewiseDefined

        self.pos        = CenterPosCandi
        self.orient     = OrieCandi
        self.vertices   = (vtx0,vtx1,vtx2,vtx3)
        self.d2vertices = [[vtx0.x,vtx0.y],[vtx1.x,vtx1.y],[vtx2.x,vtx2.y],[vtx3.x,vtx3.y]]
        self.lines      = [ Daseon.lineEQN(self.vertices[0], self.vertices[1]),
                            Daseon.lineEQN(self.vertices[1], self.vertices[2]),
                            Daseon.lineEQN(self.vertices[2], self.vertices[3]),
                            Daseon.lineEQN(self.vertices[3], self.vertices[0]) ]

    def __repr__(self):
        return  "Center Pos : " + "{:.2f}".format(self.pos.x)+", "+"{:.2f}".format(self.pos.y)+\
                "\n Vertices : \n"+ "{:.2f}".format(self.vertices[0].x) +", "+ "{:.2f}".format(self.vertices[0].y)+"\n"\
                + "{:.2f}".format(self.vertices[1].x) +", "+ "{:.2f}".format(self.vertices[1].y)+"\n"\
                + "{:.2f}".format(self.vertices[2].x) +", "+ "{:.2f}".format(self.vertices[2].y)+"\n"\
                + "{:.2f}".format(self.vertices[3].x) +", "+ "{:.2f}".format(self.vertices[3].y)+"\n"
                
    def lidarReflection(self, beam):
        Intersecs   = []
        Dists       = []
        for lineNum in range(4):
            intersectionNow = self.lines[lineNum].FindIntersection(beam)
            if(self.lines[lineNum].Is_inMyRange(beam, intersectionNow)):
                Intersecs.append(intersectionNow)
                Dists.append( (beam.pnt1 - intersectionNow).mag )
        if Dists:
            MinArg      = np.argmin(Dists)
        else:
            MinArg      = 0
            Intersecs   = [Vector3(9999999, 9999999, 0)]
            Dists       = [9999999]
        return Intersecs[MinArg], Dists[MinArg]

def getVertices(centerpos, dimension, orientation):
    RotationMat         = DCM5DOF( Vector3(0,0,orientation) )
    vtx0                = centerpos + RotationMat.rotate(Vector3( dimension[0]/2,  dimension[1]/2, 0),'inv')
    vtx1                = centerpos + RotationMat.rotate(Vector3(-dimension[0]/2,  dimension[1]/2, 0),'inv')
    vtx2                = centerpos + RotationMat.rotate(Vector3(-dimension[0]/2, -dimension[1]/2, 0),'inv')
    vtx3                = centerpos + RotationMat.rotate(Vector3( dimension[0]/2, -dimension[1]/2, 0),'inv')
    return [vtx0, vtx1, vtx2, vtx3]

def isInside(theObject, vtx0, vtx1, vtx2, vtx3, areaREF):
    AreaT01             = TriAreaVtx(theObject, vtx0, vtx1)
    AreaT12             = TriAreaVtx(theObject, vtx1, vtx2)
    AreaT23             = TriAreaVtx(theObject, vtx2, vtx3)
    AreaT30             = TriAreaVtx(theObject, vtx3, vtx0)
    return ( (AreaT01 + AreaT12 + AreaT23 + AreaT30) <= (areaREF+1) )

def TriAreaVtx(p1, p2, p3):
    MAT     = np.array([ [p1.x, p1.y, 1], [p2.x, p2.y, 1], [p3.x, p3.y, 1]])
    return 0.5*abs(np.linalg.det(MAT))