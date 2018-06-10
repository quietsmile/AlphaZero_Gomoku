# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

#--encoding=utf8
import socket  
#import mxnet as mx
import numpy as np
import time
import cPickle as pickle
addr = ('10.0.0.9',10003)
s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)  
s.connect(addr)  
while 1:  
    image = np.random.random((5,9,9))
    prob = np.random.random((81)) 
    prob /= np.sum(prob)
    value = np.random.random((1,))
    pkl = pickle.dumps([image, prob, value], protocol=2)
    #print('pkllen', len(pkl))
    s.sendto(pkl, addr)  
    #print('server send to')
    #label,addrg = s.recvfrom(2048)  
    #if label:  
    #    print("from:",addrg)  
    #    print("got recive :",cPickle.loads(label))  
    #time.sleep(1)
s.close()
