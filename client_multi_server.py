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
addr1 = ('10.0.0.9',10003)
s1 = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)  
s1.connect(addr1)  

addr2 = ('10.0.0.9',10004)
s2 = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)  
s2.connect(addr2)  
while 1:  
    image = np.random.random((5,9,9))
    prob = np.random.random((81)) 
    prob /= np.sum(prob)
    value = np.random.random((1,))
    pkl1 = pickle.dumps([image, prob, value], protocol=2)
    pkl2 = pickle.dumps(image.reshape((-1,5,9,9)), protocol=2)
    #print('pkllen', len(pkl))
    s1.sendto(pkl1, addr1)  
    s2.sendto(pkl2, addr2)  
    #print('server send to')
    #label,addrg = s.recvfrom(2048)  
    #if label:  
    #    print("from:",addrg)  
    #    print("got recive :",cPickle.loads(label))  
    time.sleep(1)
s.close()
