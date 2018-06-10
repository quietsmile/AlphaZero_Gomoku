# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""
import cPickle as pickle
import random
import numpy as np
from collections import deque
import time,sys

import socket  
import mxnet as mx
from policy_value_net_mxnet import PolicyValueNet # Keras
from multiprocessing import Queue, Process

address = ('10.0.0.9',10004)
s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)  
s.bind(address)  
print('server is ready')

batch_size = 512
learning_rate = 1e-3
epochs = 5

def producer(data_Q):
    print('waiting data')
    while True:  
        pkl,addr=s.recvfrom(20480)  
        board = pickle.loads(pkl) 
        data_Q.put([board, addr])

def consumer(data_Q):
    policy_value_net = PolicyValueNet(9,9)
    while True:
        board, addr = data_Q.get(True)
        start = time.time()
        if len(board.shape) == 3:
            board = board.reshape((-1, board.shape[0], board.shape[1], board.shape[2]))
        probs, v = policy_value_net.predict_board(board)
        end = time.time()
        print('cal time:', end - start)
        pkl = pickle.dumps([probs, v], protocol=2)
        s.sendto(pkl, addr)

            
data_Q = Queue()
Q_producer = Process(target=producer, args=(data_Q, ))
Q_consumer = Process(target=consumer, args=(data_Q, ))

Q_producer.start()
Q_consumer.start()
Q_producer.join()
Q_consumer.join()
