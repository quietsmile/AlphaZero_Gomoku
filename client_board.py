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
addr = ('10.0.0.9',10004)
s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)  
s.connect(addr)  
while 1:  
    board = np.random.random((1,5,9,9))
    start = time.time()
    pkl = pickle.dumps(board, protocol=2)
    s.sendto(pkl, addr)  
    label,addrg = s.recvfrom(20480)  
    if label:  
        print("got recive :",pickle.loads(label))  
    end = time.time()
    print('time: ', end - start)
    time.sleep(10)
s.close()
