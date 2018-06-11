# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""
import cPickle as pickle
import random
import numpy as np
from collections import deque
import time,sys,os

import socket  
import mxnet as mx
from policy_value_net_mxnet import PolicyValueNetPredict as PolicyValueNet # Keras
from multiprocessing import Queue, Process

address = ('10.0.0.9',10004)
s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)  
s.bind(address)  
print('server is ready')

batch_size = 28
learning_rate = 1e-3
epochs = 5


FileName = 'current_policy.caffemodel'

def producer(data_Q):
    print('waiting data')
    while True:  
        pkl,addr=s.recvfrom(20480)  
        board = pickle.loads(pkl) 
        data_Q.put([board, addr])

def consumer(data_Q):
    policy_value_net = PolicyValueNet(9,9)
    while True:
        if (open('model.log').readlines()[0].strip()=='1'):
            policy_value_net.update_predict(FileName)
            os.system('echo 0 > model.log')
            break
    print('consumer is ready')
    cnt = 0
    data_buffer = []
    addr_buffer = []
    while True:
        board, addr = data_Q.get(True)
        data_buffer.append(board)
        addr_buffer.append(addr)
        if len(data_buffer) == batch_size:
            start = time.time()
            if batch_size > 1:
                board = np.array(data_buffer, dtype=np.float32)
            else:
                board = data_buffer[0].reshape((-1, 5,9,9))
            assert(board.shape == (batch_size,5,9,9))
            lines = open('model.log').readlines() 
            try:
                if (len(lines) > 0 and lines[0].strip()=='1'):
                    policy_value_net.update_predict(FileName)
                    os.system('echo 0 > model.log')
                    print('load new model', cnt)
                    cnt += 1
                probs, vs = policy_value_net.predict_board(board)
            except:
                probs, vs = policy_value_net.predict_board(board)
            #print('probs.shape', probs.shape)
            #print('vs.shape', vs.shape)
            end = time.time()
            
            for i in range(batch_size):
                pkl = pickle.dumps([probs[i], vs[i]], protocol=2)
                s.sendto(pkl, addr_buffer[i])
            data_buffer = []
            addr_buffer = []
            
data_Q = Queue()
Q_producer = Process(target=producer, args=(data_Q, ))
Q_consumer = Process(target=consumer, args=(data_Q, ))

Q_producer.start()
Q_consumer.start()
Q_producer.join()
Q_consumer.join()
