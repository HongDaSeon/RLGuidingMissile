#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import math as m
import DaseonTypesNtf_V3 as Daseon
from DaseonTypesNtf_V3 import Vector3, DCM5DOF
import random as rd

class NFZ:

    def __init__(self, sizeRange, TargetPos, Margin):
        self.radius = sizeRange[0] + rd.random()*(sizeRange[1]-sizeRange[0])
        targetDist  = TargetPos.mag
        defined     = False
        while(not defined):
            DircCandi           = 2*m.pi*rd.random()
            DistCandi           = targetDist*rd.random()
            posCandi            = Vector3(DistCandi*m.cos(DircCandi), DistCandi*m.sin(DircCandi), 0)

            targetwiseDefined   = ( (TargetPos - posCandi).mag ) > (Margin + self.radius)
            missilewiseDefined  = ( (posCandi - Vector3(0,0,0)).mag ) > (Margin + self.radius)

            defined             = targetwiseDefined & missilewiseDefined

        self.pos    = posCandi

    def __repr__(self):
        return  "Center Pos : " + "{:.2f}".format(self.pos.x)+", "+"{:.2f}".format(self.pos.y)+\
                "\n Radius : "+"{:.2f}".format(self.radius)

    
            

