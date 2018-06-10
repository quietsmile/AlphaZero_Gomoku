# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet with Keras
Tested under Keras 2.0.5 with tensorflow-gpu 1.2.1 as backend

@author: Mingxu Zhang
""" 
from __future__ import print_function
import sys
sys.path.insert(0, '/home/mingzhang/work/dmlc/python_mxnet/python')

import mxnet as mx
import numpy as np
import pickle


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height, model_params=None):
        self.train_context = [mx.gpu(i) for i in range(1,8)]
        self.predict_context = mx.gpu(0)
        self.batchsize = 512 #must same to the TrainPipeline's self.batch_size.
        self.predict_batchsize = 512 #must same to the TrainPipeline's self.batch_size.
        self.channelnum = 5
        self.board_width = board_width
        self.board_height = board_height 
        self.l2_const = 1e-4  # coef of l2 penalty 
        self.train_batch = self.create_policy_value_train(self.batchsize)   
        self.predict_batch = self.create_policy_value_predict(self.predict_batchsize)   
        self.predict_one = self.create_policy_value_predict(1)   
        self.num = 0

        if model_params:
           self.train_batch.set_params(*model_params)
           self.predict_batch.set_params(*model_params)
           self.predict_one.set_params(*model_params)
           pass

    def conv_act(self, data, num_filter=32, kernel=(3, 3), stride=(1, 1), act='relu', dobn=True, name=''):
        # self convolution activation
        assert(name!='' and name!=None)
        pad = (int(kernel[0]/2), int(kernel[1]/2))
        w = mx.sym.Variable(name+'_weight')
        b = mx.sym.Variable(name+'_bias')
        conv1 = mx.sym.Convolution(data=data, weight=w, bias=b, num_filter=num_filter, kernel=kernel, pad=pad, name=name)
        act1 = conv1
        if dobn:
            gamma = mx.sym.Variable(name+'_gamma')
            beta = mx.sym.Variable(name+'_beta')
            mean = mx.sym.Variable(name+'_mean')
            var = mx.sym.Variable(name+'_var')
            bn = mx.sym.BatchNorm(data=conv1, gamma=gamma, beta=beta, moving_mean=mean, moving_var=var, name=name+'_bn')
            act1 = bn
        if act is not None and act!='':
            #print('....', act)
            act1 = mx.sym.Activation(data=act1, act_type=act, name=name+'_act')

        return act1
    
    def fc_self(self, data, num_hidden, name=''):
        assert(name!='' and name!=None)
        w = mx.sym.Variable(name+'_weight')
        b = mx.sym.Variable(name+'_bias')
        fc_1 = mx.sym.FullyConnected(data, weight=w, bias=b, num_hidden=num_hidden, name=name)

        return fc_1

    def create_backbone(self, input_states):
        """create the policy value network """   
       
        conv1 = self.conv_act(input_states, 64, (3, 3), name='conv1')
        conv2 = self.conv_act(conv1, 64, (3, 3), name='conv2')
        conv3 = self.conv_act(conv2, 128, (3, 3), name='conv3')
        conv4 = self.conv_act(conv3, 128, (3, 3), name='conv4')
        conv5 = self.conv_act(conv4, 256, (3, 3), name='conv5')
        final = self.conv_act(conv5, 256, (3, 3), name='conv_final')

        # action policy layers
        conv3_1_1 = self.conv_act(final, 4, (1, 1), name='conv3_1_1')
        flatten_1 = mx.sym.Flatten(conv3_1_1)       
        flatten_1 = mx.sym.Dropout(flatten_1, p=0.5)
        fc_3_1_1 = self.fc_self(flatten_1, self.board_height*self.board_width, name='fc_3_1_1')
        action_1 = mx.sym.SoftmaxActivation(fc_3_1_1) 

        # state value layers
        conv3_2_1 = self.conv_act(final, 2, (1, 1), name='conv3_2_1')
        flatten_2 = mx.sym.Flatten(conv3_2_1)
        flatten_2 = mx.sym.Dropout(flatten_2, p=0.5)
        fc_3_2_1 = self.fc_self(flatten_2, 1, name='fc_3_2_1')
        evaluation = mx.sym.Activation(fc_3_2_1, act_type='tanh')

        return action_1, evaluation



    def residual_unit(self, data, num_filter, name, stride=(1,1), bn_mom=0.9, workspace=512):
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        return conv2 + data
        #return conv2


    def action_head(self, data, name, stride=(1,1), bn_mom=0.9, workspace=512):
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv_policy = mx.sym.Convolution(data=act1, num_filter=1, kernel=(1,1), stride=stride, pad=(0,0),
                                      no_bias=False, workspace=workspace, name=name + '_conv1')
        flatten = mx.sym.Flatten(conv_policy)       
        action = mx.sym.softmax(flatten, axis=1) 

        #conv3_1_1 = self.conv_act(data, 4, (1, 1), name='conv3_1_1')
        #flatten_1 = mx.sym.Flatten(conv3_1_1)       
        #flatten_1 = mx.sym.Dropout(flatten_1, p=0.5)
        #fc_3_1_1 = self.fc_self(flatten_1, self.board_height*self.board_width, name='fc_3_1_1')
        #action = mx.sym.SoftmaxActivation(fc_3_1_1) 

        return action

    def value_head(self, data, name, stride=(1,1), bn_mom=0.9, workspace=512):
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=1, kernel=(1,1), stride=stride, pad=(0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')

        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv_value = mx.sym.Convolution(data=act2, num_filter=1, kernel=(9,9), stride=stride, pad=(0,0),
                                      no_bias=False, workspace=workspace, name=name + '_conv_value')

        flatten = mx.sym.Flatten(conv_value)       
        value = mx.sym.Activation(flatten, act_type='tanh') 

        #conv3_2_1 = self.conv_act(data, 2, (1, 1), name='conv3_2_1')
        #flatten_2 = mx.sym.Flatten(conv3_2_1)
        #flatten_2 = mx.sym.Dropout(flatten_2, p=0.5)
        #fc_3_2_1 = self.fc_self(flatten_2, 1, name='fc_3_2_1')
        #value = mx.sym.Activation(fc_3_2_1, act_type='tanh')


        return value


    def create_backbone_resnet(self, input_states):
        """create the policy value network """   

        conv_base = mx.sym.Convolution(data=input_states, num_filter=128, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=512, name='conv_base')
        act = mx.sym.Activation(data=conv_base, act_type='relu', name='conv_relu1')
        #conv1 = self.conv_act(input_states, 32, (3, 3), name='conv1')
        body = act
        for i in range(10):
            body = self.residual_unit(data=body, num_filter=128, name='conv_%d' % i)

        action = self.action_head(body, 'action')
        value = self.value_head(body, 'value')
        return action, value



    def create_backbone2(self, input_states):
        """create the policy value network """   
       
        conv1 = self.conv_act(input_states, 32, (3, 3), name='conv1')
        conv2 = self.conv_act(conv1, 64, (3, 3), name='conv2')
        conv3 = self.conv_act(conv2, 128, (3, 3), name='conv3')
        conv4 = self.conv_act(conv3, 128, (3, 3), name='conv4')
        conv5 = self.conv_act(conv4, 128, (3, 3), name='conv5')
        final = self.conv_act(conv5, 128, (3, 3), name='conv_final')

        # action policy layers
        conv3_1_1 = self.conv_act(final, 1024, (1, 1), name='conv3_1_1')
        conv3_1_2 = self.conv_act(conv3_1_1, 1, (1, 1), act=None, dobn=False, name='conv3_1_2')
        flatten_1 = mx.sym.Flatten(conv3_1_2)       
        action_1 = mx.sym.SoftmaxActivation(flatten_1) 

        # state value layers
        conv3_2_1 = self.conv_act(final, 256, (1, 1), name='conv3_2_1')
        conv3_2_2 = self.conv_act(conv3_2_1, 1, (1, 1), act=None, dobn=False, name='conv3_2_2')
        flatten_2 = mx.sym.Flatten(conv3_2_2)
        mean_2 = mx.sym.mean(flatten_2, axis=1, keepdims=True)
        evaluation = mx.sym.Activation(mean_2, act_type='tanh')

        return action_1, evaluation

    def create_policy_value_train(self, batch_size):
        input_states_shape = (batch_size, self.channelnum, self.board_height, self.board_width)
        input_states = mx.sym.Variable(name='input_states', shape=input_states_shape)
        action_1, evaluation = self.create_backbone(input_states)
        #action_1, evaluation = self.create_backbone_resnet(input_states)
 
        mcts_probs_shape = (batch_size, self.board_height * self.board_width)
        mcts_probs = mx.sym.Variable(name='mcts_probs', shape=mcts_probs_shape)
        policy_loss = -mx.sym.sum(mx.sym.log(action_1) * mcts_probs, axis=1)
        policy_loss = mx.sym.mean(policy_loss)

        input_labels_shape = (batch_size, 1)
        input_labels = mx.sym.Variable(name='input_labels', shape=input_labels_shape)
        value_loss = mx.sym.mean(mx.sym.square(input_labels - evaluation))

        loss = value_loss + policy_loss
        loss = mx.sym.MakeLoss(loss) 

        entropy = mx.sym.sum(-action_1 * mx.sym.log(action_1), axis=1)
        entropy = mx.sym.mean(entropy)
        entropy = mx.sym.BlockGrad(entropy)
        entropy = mx.sym.MakeLoss(entropy) 
        policy_value_loss = mx.sym.Group([loss, entropy])
        policy_value_loss.save('policy_value_loss.json')

        pv_train = mx.mod.Module(symbol=policy_value_loss, 
                                 data_names=['input_states'],
                                 label_names=['input_labels', 'mcts_probs'],
                                 context=self.train_context) 
        pv_train.bind(data_shapes=[('input_states', input_states_shape)], 
                      label_shapes=[('input_labels', input_labels_shape), ('mcts_probs', mcts_probs_shape)],
                      for_training=True)
        #pv_train.init_params(initializer=mx.init.Xavier())
        pv_train.init_params(initializer=mx.init.MSRAPrelu(factor_type='avg', slope=0.0))
        pv_train.init_optimizer(optimizer='sgd',
                                optimizer_params={'learning_rate':0.001, 
                                                  #'clip_gradient':0.1, 
                                                  'momentum':0.9,
                                                  'wd':0.0001})

        return pv_train

    def create_policy_value_predict(self, batch_size): 
        input_states_shape = (batch_size, self.channelnum, self.board_height, self.board_width)
        input_states = mx.sym.Variable(name='input_states', shape=input_states_shape)
        action_1, evaluation = self.create_backbone(input_states)
        #action_1, evaluation = self.create_backbone_resnet(input_states)
        policy_value_output = mx.sym.Group([action_1, evaluation])
 
        pv_predict = mx.mod.Module(symbol=policy_value_output, 
                                   data_names=['input_states'],
                                   label_names=None,
                                   context=self.predict_context) 
        
        pv_predict.bind(data_shapes=[('input_states', input_states_shape)], for_training=False)
        args, auxs = self.train_batch.get_params()
        pv_predict.set_params(args, auxs)
        
        return pv_predict
        
    def policy_value(self, state_batch):
        states = np.asarray(state_batch)
        #print('policy_value:', states.shape)
        state_nd = mx.nd.array(states)
        self.predict_batch.forward(mx.io.DataBatch([state_nd]))
        acts, vals = self.predict_batch.get_outputs()
        acts = acts.asnumpy()
        vals = vals.asnumpy()
        #print(acts[0], vals[0])

        return acts, vals

    def policy_value2(self, state_batch):
        actsall = []
        valsall = []
        for state in state_batch:
            state = state.reshape(1, self.channelnum, self.board_height, self.board_width)
            #print(state.shape)
            state_nd = mx.nd.array(state)
            self.predict_one.forward(mx.io.DataBatch([state_nd]))
            act, val = self.predict_one.get_outputs()
            actsall.append(act[0].asnumpy())
            valsall.append(val[0].asnumpy())
        acts = np.asarray(actsall)
        vals = np.asarray(valsall)
        #print(acts.shape, vals.shape)

        return acts, vals
       
    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        #print('call network')
        legal_positions = board.availables
        current_state = board.current_state().reshape(1, self.channelnum, self.board_height, self.board_width)
        state_nd = mx.nd.array(current_state)
        self.predict_one.forward(mx.io.DataBatch([state_nd]))
        acts_probs, values = self.predict_one.get_outputs()
        acts_probs = acts_probs.asnumpy()
        values = values.asnumpy()
        #print(acts_probs[0, :4])
        legal_actprob = acts_probs[0][legal_positions] 
        act_probs = zip(legal_positions, legal_actprob)
       # print(len(legal_positions), legal_actprob.shape, acts_probs.shape)
       # if len(legal_positions)==0:
       #     exit()

        return act_probs, values[0]


    def predict_board(self, board_state):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        #print('call network')
        state_nd = mx.nd.array(board_state)
        self.predict_one.forward(mx.io.DataBatch([state_nd]))
        acts_probs, values = self.predict_one.get_outputs()
        acts_probs = acts_probs.asnumpy()
        values = values.asnumpy()

        return acts_probs[0], values[0]



    def train_step(self, state_batch, mcts_probs, winner_batch, learning_rate):
        #print('hello training....')
        #print(mcts_probs[0], winner_batch[0])
        self.train_batch._optimizer.lr = learning_rate
        state_batch = mx.nd.array(np.asarray(state_batch).reshape(-1, self.channelnum, self.board_height, self.board_width))
        mcts_probs = mx.nd.array(np.asarray(mcts_probs).reshape(-1, self.board_height*self.board_width))
        winner_batch = mx.nd.array(np.asarray(winner_batch).reshape(-1, 1))

        print('state_batch', state_batch.shape)
        print('probs', mcts_probs.shape)
        print('values', winner_batch.shape)
        self.train_batch.forward(mx.io.DataBatch([state_batch], [winner_batch, mcts_probs]))
        self.train_batch.backward()
        self.train_batch.update()
        loss, entropy = self.train_batch.get_outputs()
      
        args, auxs = self.train_batch.get_params()
        self.predict_batch.set_params(args, auxs)
        self.predict_one.set_params(args, auxs)

        return loss.asnumpy(), entropy.asnumpy()

    def get_policy_param(self):
        net_params = self.train_batch.get_params()        
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()
        print('>>>>>>>>>> saved into', model_file)
        pickle.dump(net_params, open(model_file, 'wb'), protocol=2)
