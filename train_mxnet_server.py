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

address = ('10.0.0.9',10003)
s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)  
s.bind(address)  
print('server is ready')

batch_size = 512
learning_rate = 1e-3
epochs = 5

def policy_update(policy_value_net):
    mini_batch = random.sample(list(data_buffer), batch_size)
    state_batch = [data[0] for data in mini_batch]
    mcts_probs_batch = [data[1] for data in mini_batch]
    winner_batch = [data[2] for data in mini_batch]
    for i in range(epochs):
        loss, entropy = policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                learning_rate)
        new_probs, new_v = policy_value_net.policy_value(state_batch)
    return loss, entropy


data_buffer = deque(maxlen=100000)

def producer(data_Q):
    print('waiting data')
    while True:  
        pkl,addr=s.recvfrom(20480)  
        #print('rev pkl len', len(pkl))
        image_infos = pickle.loads(pkl) 
        data_Q.put(image_infos)

def consumer(data_Q):
    policy_value_net = PolicyValueNet(9,9)
    while True:
        for i in range(10):
            image_infos = data_Q.get(True)
            data_buffer.append(image_infos)
        if len(data_buffer) >= batch_size:
            loss, entropy = policy_update(policy_value_net)
            print('loss', loss, 'entropy', entropy)
        else:
            print('len(data_buffer)', len(data_buffer))

            
data_Q = Queue()
Q_producer = Process(target=producer, args=(data_Q, ))
Q_consumer = Process(target=consumer, args=(data_Q, ))

Q_producer.start()
Q_consumer.start()
Q_producer.join()
Q_consumer.join()
