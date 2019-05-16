#!/usr/bin/python

from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_keras import PolicyValueNetwork
from human import Human

import pygame
import numpy as np
import os
import utils

def run():
    board = Board(width=7, height=7, n_in_row=4)
    game = Game(board)
    best_policy = PolicyValueNetwork(N=7, planes=4, model_name = 'models/558.model.h5')
    mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_mcts=500)
    human = Human()
    if(not game.PlayWithHumanMode(human, mcts_player, BOARD_RAOD=7, start_player=1, is_shown=1)):
        return False

run()