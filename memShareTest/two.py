#!/usr/bin/env python
#-*- coding: utf-8 -*-
from redis import Redis
from time import sleep
import numpy as np
import pickle as pkl

cli = Redis('localhost')
shared_var = 1

while True:
    array = pkl.loads( cli.get('share_place') )
    print( array[0,1] )
    
