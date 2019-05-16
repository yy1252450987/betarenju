from __future__ import print_function
import numpy as np
import pygame
import os

class Board():
    def __init__(self, width, height, n_in_row):
        self.width = width
        self.height = height
        self.n_in_row = n_in_row
        self.states = {}
        self.players = [1, 2]
    
    def InitBoard(self, start_player=0):
        self.current_player = self.players[start_player]
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def DoMove(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (self.players[0] if self.current_player == self.players[1] else self.players[1])
        self.last_move = move

    def IsGameOver(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1
    
    def current_state(self):
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0
        return square_state[:, ::-1, :]


class Game():
    def __init__(self, board):
        self.board = board

    def DrawBoard(self, surf, HEIGHT, WIDTH, BOARD_RAOD, background_img):
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        surf.blit(background_img, (0, 0))
        grid = int(HEIGHT/(BOARD_RAOD+2-1))
        boundary_lines = [[(grid, grid), (grid, HEIGHT-grid)], [(HEIGHT-grid,grid), (HEIGHT-grid, HEIGHT-grid)],
                        [(grid, grid), (HEIGHT-grid, grid)], [(grid,HEIGHT-grid), (HEIGHT-grid, HEIGHT-grid)]]
        for bline in boundary_lines:
            pygame.draw.line(surf, BLACK, bline[0], bline[1], 2)
        for i in range(BOARD_RAOD-2):
            board_road_vline = [(grid, grid*(2+i)), (HEIGHT-grid, grid*(2+i))]
            board_road_hline = [(grid*(2+i), grid), (grid*(2+i), HEIGHT-grid)]
            pygame.draw.line(surf, BLACK, board_road_vline[0], board_road_vline[1])
            pygame.draw.line(surf, BLACK, board_road_hline[0], board_road_hline[1])
        centre_point = [(grid*(1+3), grid*(1+3)), (grid*(1+3), HEIGHT-grid*(1+3)),
                        (HEIGHT-grid*(1+3), HEIGHT-grid*(1+3)), (HEIGHT-grid*(1+3), grid*(1+3)),
                        (int(HEIGHT/2),int(HEIGHT/2))
        ]
        for point in centre_point:
            pygame.draw.circle(surf, BLACK, point, 5)

    def SelfPlayMode(self, player):
        self.board.InitBoard()
        states, mcts_probs, current_players = [], [], []
        game_end, winner = self.board.IsGameOver()
        while not game_end:
            move, move_probs = player.get_action(self.board, return_prob=True)
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            self.board.DoMove(move)
            game_end, winner = self.board.IsGameOver()
        winners_z = np.zeros(len(current_players))
        if(winner != -1):
            winners_z[np.array(current_players) == winner] = 1
            winners_z[np.array(current_players) != winner] = -1
        return winner, list(zip(states, mcts_probs, winners_z))[:]

    def PlayWithHumanMode(self, player1, player2, BOARD_RAOD=7, start_player=0, is_shown=1):
        pygame.init()
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        textFont = pygame.font.SysFont("arial", 20)

        WIDTH = 720
        HEIGHT = 720
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("五子棋")
        FPS = 30
        clock = pygame.time.Clock()

        base_folder = "/home/ys/Desktop/GoBang"
        img_folder = os.path.join(base_folder, 'images')
        background_img = pygame.image.load(os.path.join(img_folder, 'back.png')).convert()

        self.DrawBoard(screen, HEIGHT, WIDTH, BOARD_RAOD, background_img)
        BOARD = np.zeros((BOARD_RAOD, BOARD_RAOD))
        grid_axel = WIDTH//(BOARD_RAOD+1)
        pygame.display.flip()

        self.board.InitBoard(start_player)
        self.board.current_player = self.board.players[0]
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        grid = WIDTH//(BOARD_RAOD+1)
        clock.tick(FPS)
        while True:
            event = pygame.event.wait()
            if(event.type == pygame.MOUSEBUTTONDOWN):
                break
            if(event.type == pygame.QUIT):
                return False
        while True:
            current_player = self.board.current_player
            player_in_turn = players[current_player]
            if(current_player == 1):
                while True:
                    event = pygame.event.wait()
                    if(event.type == pygame.MOUSEBUTTONDOWN):
                        pressed_array = pygame.mouse.get_pressed()
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        locx = (mouse_y+grid//2)//grid
                        locy = (mouse_x+grid//2)//grid
                        locx, locy = BOARD_RAOD-locx, locy-1
                        move = locx * self.board.width + locy
                        if(move in self.board.availables):
                            loc_pos = (grid*(locy+1), grid*(BOARD_RAOD-locx))
                            pygame.draw.circle(screen, WHITE, loc_pos, 20)
                            break
                        else:
                            continue
                    if(event.type == pygame.QUIT):
                        return False
            if(current_player == 2):
                move = player_in_turn.get_action(self.board, None, None)
                locx, locy = move//self.board.width, move%self.board.width  
                loc_pos = (grid*(locy+1), grid*(BOARD_RAOD-locx))
                pygame.draw.circle(screen, BLACK, loc_pos, 20)
            pygame.display.flip()
            self.board.DoMove(move)
            end, winner = self.board.IsGameOver()
            if end:
                if is_shown:
                    if winner != -1:
                        textSurface = textFont.render("Game Over: %s VITORY" %(players[winner]), True, BLACK)
                        screen.blit(textSurface, (0, 0))
                        pygame.display.flip()
                    else:
                        textSurface = textFont.render("Game Over: Tie" , True, BLACK)
                        screen.blit(textSurface, (0, 0))
                        pygame.display.flip()
                while True:
                    event = pygame.event.wait()
                    if(event.type == pygame.MOUSEBUTTONDOWN):
                        break
                    if(event.type == pygame.QUIT):
                        return False
                self.PlayWithHumanMode(player1, player2, start_player=0, is_shown=1)