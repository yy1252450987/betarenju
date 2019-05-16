# -*- coding; utf-8 -*-

import numpy as np
from game import *
from collections import deque
from policy_value_net_keras import PolicyValueNetwork
from mcts_alphaZero import MCTSPlayer
from utils import DataAugmentation

size = 9
board = Board(width=size, height=size, n_in_row=5)
game = Game(board)
temperature = 1.0
data_buffer = deque(maxlen=10000)
iteration = 10
number_selfplay_game = 1
policy_value_network = PolicyValueNetwork(N=size, planes=4)
mcts_player = MCTSPlayer(n_mcts=1000, policy_value_function=policy_value_network.policy_value_fn, is_selfplay=True)

policy_value_network.model.save("models/initial.model.h5")

for i in range(iteration):
    print("----------ITER %s----------" % i)
    for j in range(number_selfplay_game):
        winner, play_data = game.SelfPlayMode(mcts_player)
        play_data = DataAugmentation(play_data)
        data_buffer.extend(play_data)
    if(len(data_buffer) > 512):
        policy_value_network.trainFit(data_buffer)
        if(i > 0):
            policy_value_network.model.save("models/%s.model.h5" % i)