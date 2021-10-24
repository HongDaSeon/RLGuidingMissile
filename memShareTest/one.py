#!/usr/bin/env python
#-*- coding: utf-8 -*-
from redis import Redis
from time import sleep
import numpy as np
import pickle as pkl


cli = Redis('localhost')
shared_var = pkl.dumps(np.array([[1,2,3,4,5],[6,7,8,9,0]],dtype='float32'))

while True:
    cli.set('share_place', shared_var)
    
    
