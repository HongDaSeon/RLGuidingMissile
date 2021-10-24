#!/usr/bin/env python
#-*- coding: utf-8 -*-

import pygame as pg
import numpy as np

C_Missile       = (50, 100, 200)
C_NoflyZone     = (100, 100, 100)
C_lidarSens     = (150, 150, 200)
C_Target        = (200, 50, 100)
C_lGrey         = (200, 200, 200)

class VisualizationPygame:
    
    def __init__(self, dispSize, lookScale, Title,joy=False):
        global LLSS, DDSS
        LLSS = lookScale
        DDSS = dispSize

        self.dSize  = dispSize
        self.LS     = lookScale
        self.controller = None
        self.Title      = Title
        pg.init()
        
        if joy:
            pg.joystick.init()
            try:
                self.controller = pg.joystick.Joystick(0)
                self.controller.init()
                print ("Joystick_Paired: {0}".format(self.controller.get_name()))
            except pg.error:
                print ("None of or Invalid joystick connected")
        
        self.Disp = pg.display.set_mode((self.dSize[0], self.dSize[1]))
        pg.display.set_caption(self.Title)
        self.centre = (self.dSize[0]/2, self.dSize[1]/2)

    def draw_NFZ(self, Battlefield):
        for nfzs in Battlefield.NoFlyZones:
            pg.draw.circle(self.Disp, C_NoflyZone, in2Dcenter(O3to2(nfzs.pos)), nfzs.radius*self.LS)

    def draw_STT(self, Battlefield):
        for stts in Battlefield.Structs:
            pg.draw.polygon(self.Disp, C_lGrey, [ in2Dcenter(stts.d2vertices[0]),\
                                                        in2Dcenter(stts.d2vertices[1]),\
                                                        in2Dcenter(stts.d2vertices[2]),
                                                        in2Dcenter(stts.d2vertices[3])], 2)

    def draw_lidar(self, startObject, Battlefield):
        d2Pos = O3to2(startObject.pos)
        for lidars in Battlefield.LidarInfo:
            pg.draw.aaline(self.Disp, C_lidarSens, in2Dcenter(d2Pos), in2Dcenter(O3to2(lidars[1])), 1)
    
    def draw_Spot(self, theObject, color, size=5):
        d2Pos = O3to2(theObject.pos)
        pg.draw.circle(self.Disp, color, in2Dcenter(d2Pos), 5)

    def update(self):
        pg.display.update()

    def wipeOut(self):
        pg.draw.rect(self.Disp, (0,0,0), [0, 0 , self.dSize[0], self.dSize[1]])

    def event_get(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                print('goodbye')
                sys.exit()
            if event.type == pg.JOYAXISMOTION:  # Joystick
                pass
            if event.type == pg.JOYBUTTONDOWN:  
                print("Joystick Button pressed")

def in2Dcenter(d2list):
    return [d2list[0]*LLSS+DDSS[0]/2, d2list[1]*LLSS+DDSS[1]/2 ]

def O3to2(vec3):
    return [vec3.x, vec3.y]