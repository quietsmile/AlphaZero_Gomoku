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

def policy_update():
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

def train_producer(data_Q):
    print('waiting data')
    while True:  
        pkl,addr=s.recvfrom(20480)  
        #print('rev pkl len', len(pkl))
        image_infos = pickle.loads(pkl) 
        data_Q.put(image_infos)

def train_consumer(data_Q):
    policy_value_net = PolicyValueNet(9,9)
    while True:
        for i in range(10):
            image_infos = data_Q.get(True)
            data_buffer.append(image_infos)
        if len(data_buffer) >= batch_size:
            loss, entropy = policy_update()
            print('loss', loss, 'entropy', entropy)
        else:
            print('len(data_buffer)', len(data_buffer))

            

def predict_producer(data_Q):
    print('waiting data')
    while True:  
        pkl,addr=s.recvfrom(20480)  
        board = pickle.loads(pkl) 
        data_Q.put([board, addr])

def predict_consumer(data_Q):
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
            
train_data_Q = Queue()
Q_train_producer = Process(target=train_producer, args=(train_data_Q, ))
Q_train_consumer = Process(target=train_consumer, args=(train_data_Q, ))

Q_train_producer.start()
Q_train_consumer.start()


predict_data_Q = Queue()
Q_predict_producer = Process(target=predict_producer, args=(predict_data_Q, ))
Q_predict_consumer = Process(target=predict_consumer, args=(predict_data_Q, ))

Q_predict_producer.start()
Q_predict_consumer.start()



Q_train_producer.join()
Q_train_consumer.join()

Q_predict_producer.join()
Q_predict_consumer.join()


