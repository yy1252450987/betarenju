from __future__ import print_function

import numpy as np
import random
import time
from keras.models import Model, load_model
from keras.layers import Activation, Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import add, Flatten
from keras.optimizers import SGD

class PolicyValueNetwork():
    def __init__(self, N=7, planes=4,input_neuron_num=64, input_kernal_size=(3,3),kernal_num=128, kernal_size=(3,3),block_num=3, batch_size=256, epoch=1, model_name=None):
        self.N = N
        self.planes = planes
        self.input_neuron_num = input_neuron_num
        self.input_kernal_size = input_kernal_size
        self.kernal_num = kernal_num
        self.kernal_size = kernal_size
        self.block_num = block_num
        self.batch_size = batch_size
        self.epoch = epoch
        self.create()
        if(model_name):
            self.model = load_model(model_name)

    def create(self):
        N = self.N
        planes = self.planes
        input_tensor = Input((N, N, planes))
        x = input_tensor
        x = Conv2D(self.input_neuron_num, self.input_kernal_size, padding="SAME", strides=(1,1), name="conv1")(x)
        x = BatchNormalization(axis=3, name="bn1")(x)
        x = Activation('relu')(x)
        for i in range(self.block_num):
            x = self.rediueBlock(x, stage=i, block="block")
        prob = self.policyHead(x)
        outcome = self.valueHead(x)
        self.model = Model(input_tensor, [prob, outcome])
        sgd = SGD(decay=0.01,momentum=0.9)
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'],\
                            optimizer=sgd,metrics=['accuracy'])

    def policyHead(self, x):
        x = Conv2D(2, (1,1))(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        x = Dense(self.N*self.N, activation='softmax', name='probability')(x)
        return x

    def valueHead(self, x):
        x = Conv2D(1, (1,1))(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        x = Dense(self.N, activation='relu')(x)
        x = Dense(1, activation='tanh', name='outcome')(x)
        return x

    def rediueBlock(self, input_tensor, stage, block):
        name_base = block+str(stage)
        x = input_tensor
        x = Conv2D(self.kernal_num, self.kernal_size, padding="SAME", strides=(1,1), name=name_base+"_conv_a")(x)
        x = BatchNormalization(axis=3, name=name_base+"_bn_a")(x)
        x = Activation('relu')(x)
        x = Conv2D(self.input_neuron_num, self.kernal_size, padding="SAME", strides=(1,1), name=name_base+"_conv_b")(x)
        x = BatchNormalization(axis=3, name=name_base+"_bn_b")(x)
        x = Activation('relu')(x)
        x = add([x, input_tensor])
        return x

    def plot(self, figname):
        plot_model(self.model, to_file=figname,show_shapes=True)


    def trainFit(self, dataset):
        batch_size = self.batch_size
        for i in range(self.epoch):
            bacth_data = random.sample(dataset, self.batch_size)
            X = np.asarray([data[0].reshape(self.N, self.N, self.planes) for data in bacth_data])
            y_probs = [data[1] for data in bacth_data]
            y_outcome = [data[2] for data in bacth_data]
            logs = self.model.train_on_batch(X, [y_probs, y_outcome])
        print(logs)

    def policy_value_fn(self, board):
        legal_positions = board.availables
        current_state = board.current_state()
        input_array = np.asarray(current_state).reshape(-1, self.N, self.N, self.planes)
        act_probs, value = self.model.predict_on_batch(np.asarray(input_array)) #self.policy_value(current_state.reshape(-1, self.board_width, self.board_height, 4))
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value[0][0]
    