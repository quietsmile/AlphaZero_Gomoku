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
from policy_value_net_mxnet import PolicyValueNetTrain as PolicyValueNet # Keras
from multiprocessing import Queue, Process

address = ('10.0.0.9',10003)
s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)  
s.bind(address)  
print('server is ready')

batch_size = 512
epochs = 5
board_height = 9
board_width = 9

def policy_update(policy_value_net, learning_rate):
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
        #new_probs, new_v = policy_value_net.policy_value(state_batch)

        #if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
        #    print('early stopping:', i, self.epochs)
        #    break
        ## adaptively adjust the learning rate
        #if kl > self.kl_targ * 2 and self.lr_multiplier > 0.05:
        #    self.lr_multiplier /= 1.5
        #elif kl < self.kl_targ / 2 and self.lr_multiplier < 20:
        #    self.lr_multiplier *= 1.5
    return loss, entropy


data_buffer = deque(maxlen=100000)

def producer(data_Q):
    print('waiting data')
    while True:  
        pkl,addr=s.recvfrom(20480)  
        #print('rev pkl len', len(pkl))
        image_infos = pickle.loads(pkl) 
        data_Q.put(image_infos)



def get_equi_data(play_data):
    """augment the data set by rotation and flipping
    play_data: [(state, mcts_prob, winner_z), ..., ...]
    """
    extend_data = []
    state, mcts_porb, winner = play_data
    for i in [1, 2, 3, 4]:
        # rotate counterclockwise
        equi_state = np.array([np.rot90(s, i) for s in state])
        equi_mcts_prob = np.rot90(np.flipud(
            mcts_porb.reshape(board_height, board_width)), i)
        extend_data.append((equi_state,
                            np.flipud(equi_mcts_prob).flatten(),
                            winner))
        # flip horizontally
        equi_state = np.array([np.fliplr(s) for s in equi_state])
        equi_mcts_prob = np.fliplr(equi_mcts_prob)
        extend_data.append((equi_state,
                            np.flipud(equi_mcts_prob).flatten(),
                            winner))
    return extend_data



def consumer(data_Q):
    policy_value_net = PolicyValueNet(board_height, board_width)
    print 'consumer is ready'
    new_add = 0
    cnt = 0
    learning_rates = [2e-4, 2e-5, 1e-5]
    start = time.time()
    while True:
        image_infos = data_Q.get(True)
        extend_data = get_equi_data(image_infos)
        data_buffer.extend(extend_data)
        new_add += len(extend_data)
        if new_add >= batch_size / 2. and len(data_buffer) >= batch_size:
            if cnt < 100000:
                lr = 2e-4
            elif cnt < 200000:
                lr = 2e-5
            elif cnt < 300000:
                lr = 1e-5
            loss, entropy = policy_update(policy_value_net, lr)
            end = time.time()
            print('mini-batch', cnt, 'loss', loss, 'entropy', entropy, 'time', end - start)
            start = end
            new_add = 0
            cnt += 1
        else:
            #print('len(data_buffer)', len(data_buffer))
            pass

            
data_Q = Queue()
Q_producer = Process(target=producer, args=(data_Q, ))
Q_consumer = Process(target=consumer, args=(data_Q, ))

Q_producer.start()
Q_consumer.start()
Q_producer.join()
Q_consumer.join()
