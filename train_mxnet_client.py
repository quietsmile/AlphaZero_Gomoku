# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import pickle
import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
#from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
#from policy_value_net_mxnet import PolicyValueNet # Keras
import time,sys

import socket  
import cPickle as pickle

addr1 = ('10.0.0.9',10003)
s1 = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)  
s1.connect(addr1)  

addr2 = ('10.0.0.9',10004)
s2 = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)  
s2.connect(addr2)  

class TrainPipeline():
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = 9
        self.board_height = 9
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-4
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 1000
        self.game_batch_num = 150000
        self.best_win_ratio = 0.55
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        #if init_model:
            # start training from an initial policy-value net
            #self.policy_value_net = PolicyValueNet(self.board_width,
        #                                           self.board_height,
        #                                           model_params=init_model)
        #else:
            # start training from a new policy-value net
            #self.policy_value_net = PolicyValueNet(self.board_width,
            #                                       self.board_height)
        self.policy_value_net = None
        #self.mcts_player = []                                           
        #for i in range(10):
        #    self.mcts_player.append(MCTSPlayer(self.policy_value_net.policy_value_fn,
        #                              c_puct=self.c_puct,
        #                              n_playout=self.n_playout,
        #                              is_selfplay=1))
        self.mcts_player = MCTSPlayer(None,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
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

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                      s2, addr2, temp=self.temp)
            play_data = list(play_data)[:]
            #self.episode_len = len(play_data)
            #play_data = self.get_equi_data(play_data)
            # s1 for training
            for data in play_data:
                pkl = pickle.dumps(data, protocol=2)
                s1.sendto(pkl, addr1)
            #s2.sendto(pkl2, addr2)  

    def run(self):
        while 1:
            self.collect_selfplay_data(1)

if __name__ == '__main__':
    training_pipeline = TrainPipeline(None)
    training_pipeline.run()

