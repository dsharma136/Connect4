#python main.py -p1 alphaBetaAI -p2 stupidAI -limit_players 1,2 -verbose True -seed 0
#python3 main.py -p1 alphaBetaAI -p2 monteCarloAI -limit_players 1,2 -verbose True -seed 0
import random
import time
import pygame
import math
from connect4 import connect4
from copy import deepcopy
import numpy as np

class connect4Player(object):
	def __init__(self, position, seed=0, CVDMode=False):
		self.position = position
		self.opponent = None
		self.seed = seed
		random.seed(seed)
		if CVDMode:
			global P1COLOR
			global P2COLOR
			P1COLOR = (227, 60, 239)
			P2COLOR = (0, 255, 0)

	def play(self, env: connect4, move: list) -> None:
		move = [-1]

class human(connect4Player):

	def play(self, env: connect4, move: list) -> None:
		move[:] = [int(input('Select next move: '))]
		while True:
			if int(move[0]) >= 0 and int(move[0]) <= 6 and env.topPosition[int(move[0])] >= 0:
				break
			move[:] = [int(input('Index invalid. Select next move: '))]

class human2(connect4Player):

	def play(self, env: connect4, move: list) -> None:
		done = False
		while(not done):
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					sys.exit()

				if event.type == pygame.MOUSEMOTION:
					pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
					posx = event.pos[0]
					if self.position == 1:
						pygame.draw.circle(screen, P1COLOR, (posx, int(SQUARESIZE/2)), RADIUS)
					else: 
						pygame.draw.circle(screen, P2COLOR, (posx, int(SQUARESIZE/2)), RADIUS)
				pygame.display.update()

				if event.type == pygame.MOUSEBUTTONDOWN:
					posx = event.pos[0]
					col = int(math.floor(posx/SQUARESIZE))
					move[:] = [col]
					done = True

class randomAI(connect4Player):

	def play(self, env: connect4, move: list) -> None:
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		move[:] = [random.choice(indices)]

class stupidAI(connect4Player):

	def play(self, env: connect4, move: list) -> None:
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		if 3 in indices:
			move[:] = [3]
		elif 2 in indices:
			move[:] = [2]
		elif 1 in indices:
			move[:] = [1]
		elif 5 in indices:
			move[:] = [5]
		elif 6 in indices:
			move[:] = [6]
		else:
			move[:] = [0]

############################################################################################################################

class minimaxAI(connect4Player):
	numMovesMade = 0
	def play(self, env: connect4, move: list) -> None:
		if self.numMovesMade == 0:
			move[:] = [3]
		else:
			possible_moves = [i for i, p in enumerate(env.topPosition >= 0) if p]
			moves = np.full(len(possible_moves), -float('inf'))
			for idx, i in enumerate(possible_moves):
				copy = deepcopy(env) 
				self.moveSimulation(copy, i, self.position)
				if copy.gameOver(i, self.position):
					move[:] = [i]
					return
				else:
					moves[idx] = self.minimax(copy, self.opponent.position, 0)
			best_move_index = possible_moves[np.argmax(moves)]
			move[:] = [best_move_index]

		self.numMovesMade += 1

	def moveSimulation(self, env: connect4, move: int, player: int):
		if 0 <= move < len(env.topPosition) and env.topPosition[move] >= 0:
			row = env.topPosition[move]
			env.board[row][move] = player
			env.topPosition[move] -= 1
			env.history[0].append(move)
		else:
			print("Invalid move:", move)

	#some similar logic is used but this is incomplete minimax function
	def minimax(self, env: connect4, player: int, depth: int):
		if depth == 0:
			return self.eval(env)
		
		if player == self.position:
			value = -float('inf')
			for i in range(env.width):
				if env.topPosition[i] >= 0:
					copy = deepcopy(env)
					self.moveSimulation(copy, i, player)
					if copy.gameOver(i, player):
						return 100000 

					value = max(value, self.minimax(copy, self.opponent.position, depth - 1))

			return value
		else:
			value = float('inf')
			for i in range(env.width):
				if env.topPosition[i] >= 0:
					copy = deepcopy(env)
					self.moveSimulation(copy, i, player)
					if copy.gameOver(i, player):
						return -100000  

					value = min(value, self.minimax(copy, self.position, depth - 1))

			return value

	#note: incorrect evaluation function
	# alphabeta function has correct evaluation function
	def eval(self, env: connect4):
		playerVertical, oppDL, playerDL, oppDR, playerDR, oppVert = [
			np.zeros(7) for _ in range(6)
		]
		player, opp = self.position, self.opponent.position
		openThreeOpp, endedPlayerThree = 0, 0
		board = env.board
		playc1, playc2, playc3 = 0, 0, 0
		oppc1, oppc2, oppc3 = 0, 0, 0
		wp2, wp3, wo2, wo3 = 2, 3, 2, 4

		for r in reversed(range(len(board))):
			playerInRow = False
			playerNumInRow = 0
			oppInRow = False
			oppNumInRow = 0

			for c in range(len(board[0])):
				if board[r][c] == player:
					playerInRow = True
					playerNumInRow += 1
					playerVertical[c] += 1

					if -3 < c - r < 4:
						playerDR[c - r + 2] += 1
					if 2 < r + c < 9:
						playerDR[r + c - 3] += 1

				if board[r][c] != player or r == 0 or c == 6:
					if 2 < r + c < 9 and (playerDR[r + c - 3] != 0 or r == 0 or c == 6):
						if playerDR[r + c - 3] == 2:
							playc2 += 1
						elif playerDR[r + c - 3] == 3:
							playc3 += 1
							if playerDR[r + c - 3] != opp:
								endedPlayerThree += 1
							if c - 4 > -1 and r + 4 < 6 and board[r + 4][c - 4] != opp:
								endedPlayerThree += 1
						playerDR[r + c - 3] = 0 
					if -3 < c - r < 4 and (playerDL[c - r + 2] != 0 or r == 0 or c == 6):
						if playerDL[c - r + 2] == 2:
							playc2 += 1
						elif playerDL[c - r + 2] == 3:
							playc3 += 1
							if playerDL[c - r + 2] != opp:
								endedPlayerThree += 1
							if c + 4 < 7 and r + 4 < 6 and board[r + 4][c + 4] != opp:
								endedPlayerThree += 1
						playerDL[c - r + 2] = 0  

					if playerVertical[c] != 0 or r == 0:
						if playerVertical[c] == 2:
							playc2 += 1
						elif playerVertical[c] == 3:
							playc3 += 1
						
							if board[r][c] != opp or (c - 4 > 0 and board[r][c - 4] != opp):
								endedPlayerThree += 1
						playerVertical[c] = 0 

					if playerInRow or c == 6:
						playerInRow = False
						if playerNumInRow == 2:
							playc2 += 1
							if c != 6 and c - 2 > -1 and board[r][c - 2] != opp and board[r][c] != opp:
								endedPlayerThree += 2
						elif playerNumInRow == 3:
							playc3 += 1
							if board[r][c] != opp:
								endedPlayerThree += 1
							if c - 4 > 0 and board[r][c - 4] != opp:
								endedPlayerThree += 1
						playerNumInRow = 0
						
				if board[r][c] == opp:
					oppInRow = True if oppInRow is False else oppInRow
					oppNumInRow += 1
					oppVert[c] += 1

					if 2 < r + c < 9:
						oppDR[r + c - 3] += 1

				if board[r][c] != opp or r == 0 or c == 6:
					if 2 < r + c < 9 and (oppDR[r + c - 3] != 0 or r == 0 or c == 6):
						if oppDR[r + c - 3] == 2:
							oppc2 += 1
						elif oppDR[r + c - 3] == 3:
							oppc3 += 1
							if oppDR[r + c - 3] != player:
								openThreeOpp += 1
							if c - 4 > -1 and r + 4 < 6 and board[r + 4][c - 4] != player:
								openThreeOpp += 1
						oppDR[r + c - 3] = 0  


					if c - r > -3 and c - r < 4 and (oppDL[c - r + 2] != 0 or r == 0 or c == 6):
						if oppDL[c - r + 2] == 2:
							oppc2 = oppc2 + 1
						elif oppDL[c - r + 2] == 3:
							oppc3 = oppc3 + 1
							if oppDL[c - r + 2] != player:
								openThreeOpp = openThreeOpp + 1
							if c + 4 < 7 and r + 4 < 6 and board[r + 4][c + 4] != player:
								openThreeOpp = openThreeOpp + 1
						oppDL[c - r + 2] = 0 

					if oppVert[c] != 0 or r == 0:
						if oppVert[c] == 1:
							oppc1 = oppc1 + 1
						elif oppVert[c] == 2:
							oppc2 = oppc2 + 1
						elif oppVert[c] == 3:
							oppc3 = oppc3 + 1
							if board[r][c] != player:
								wo3 = wo3 * 5
						oppVert[c] = 0

					if oppInRow or c == 6:
						oppInRow = False
						if oppNumInRow == 2:
							oppc2 += 1
							if c != 6 and c - 2 > -1 and board[r][c - 2] != player and board[r][c] != player:
								openThreeOpp += 2
						elif oppNumInRow == 3:
							oppc3 += 1
							if board[r][c] != player:
								openThreeOpp += 1
							if c - 4 > -1 and board[r][c - 4] != player:
								openThreeOpp += 1
						oppNumInRow = 0

		pScore =  wp2 * playc2 + wp3*playc3 + 2 * endedPlayerThree
		oScore =  wo2 * oppc2 + wo3 * oppc3 + 6 * openThreeOpp
		return pScore - oScore

##############################################################################################################################

class alphaBetaAI(connect4Player):

    def __init__(self, position, seed=0, depth=3, CVDMode=False):
        super().__init__(position, seed, CVDMode)
        self.depth = 5
    
    def play(self, env: connect4, move: list) -> None:
        col, score = self.minimax(env.board, self.depth, float('-inf'), float('inf'), True)
        if col is not None and self.is_valid(env.board, col):
            move[0] = col
        else:
            move[0] = -1

    def minimax(self, board, depth, alpha, beta, maximizingPlayer):
        valid_locations = self.locations(board)
        is_terminal = self.node(board)
        
        if depth == 0 or is_terminal:
            return (None, self.evaluate_terminal_state(board, is_terminal))
        
        if maximizingPlayer:
            return self.maximize_play(board, depth, alpha, beta, valid_locations)
        else:
            return self.minimize_play(board, depth, alpha, beta, valid_locations)

    def evaluate_terminal_state(self, board, is_terminal):
        if is_terminal:
            if self.victory(board, self.position):
                return float('inf')
            elif self.victory(board, 3 - self.position):
                return float('-inf')
            else:
                return 0
        else:
            return self.score(board, self.position)

    def maximize_play(self, board, depth, alpha, beta, valid_locations):
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = self.opening(board, col)
            b_copy = board.copy()
            self.piece(b_copy, row, col, self.position)
            new_score = self.minimax(b_copy, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value

    def minimize_play(self, board, depth, alpha, beta, valid_locations):
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = self.opening(board, col)
            b_copy = board.copy()
            self.piece(b_copy, row, col, 3 - self.position)
            new_score = self.minimax(b_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value
        
    def score(self, board, piece):
        score = 0
        ROW_COUNT, COLUMN_COUNT = board.shape
        
        center_array = board[:, COLUMN_COUNT // 2]
        score += np.sum(center_array == piece) * 3
    
        def score_window(window):
            return self.myEval([int(i) for i in window], piece)
        
        for r in range(ROW_COUNT):
            for c in range(COLUMN_COUNT):
                # Horizontal
                if c <= COLUMN_COUNT - 4:
                    score += score_window(board[r, c:c+4])
                # Vertical
                if r <= ROW_COUNT - 4:
                    score += score_window(board[r:r+4, c])
                # Diagonals
                if r <= ROW_COUNT - 4 and c <= COLUMN_COUNT - 4:
                    score += score_window([board[r+i, c+i] for i in range(4)])
                # Diagonals2
                if r >= 3 and c <= COLUMN_COUNT - 4:
                    score += score_window([board[r-i, c+i] for i in range(4)])  
        return score
	
    def board_duplicate(self, env: connect4):
        return deepcopy(env.board)

    def piece(self, board, row, col, piece):
        board[row][col] = piece

    def myEval(self, window, piece):
        score = 0
        opp_piece = 3 - piece
        score_map = {(3, 1): 5, (2, 2): 2, (4, 0): 8000}

        ai_score = score_map.get((window.count(piece), window.count(0)), 0)
        opp_score = score_map.get((window.count(opp_piece), window.count(0)), 0)

        if window.count(opp_piece) == 4:
            opp_score = 600

        return ai_score - opp_score
    
    def locations(self, board):
        return [col for col in [3, 2, 4, 1, 5, 0, 6] if self.is_valid(board, col)]

    def node(self, board):
        return self.victory(board, self.position) or self.victory(board, 3 - self.position) or len(self.locations(board)) == 0

    def victory(self, board, piece):
        ROW_COUNT, COLUMN_COUNT = board.shape

        for r in range(ROW_COUNT):
            for c in range(COLUMN_COUNT):
                if c <= COLUMN_COUNT - 4 and all(board[r][c + i] == piece for i in range(4)):
                    return True
                if r <= ROW_COUNT - 4 and all(board[r + i][c] == piece for i in range(4)):
                    return True

        for r in range(ROW_COUNT):
            for c in range(COLUMN_COUNT):
                if r <= ROW_COUNT - 4 and c <= COLUMN_COUNT - 4 and all(board[r + i][c + i] == piece for i in range(4)):
                    return True
                if r >= 3 and c <= COLUMN_COUNT - 4 and all(board[r - i][c + i] == piece for i in range(4)):
                    return True
        return False

    def is_valid(self, board, col):
        return board[0][col] == 0

    def opening(self, board, col):
        for r in range(board.shape[0]-1, -1, -1):
            if board[r][col] == 0:
                return r
        return None





SQUARESIZE = 100
BLUE = (0,0,255)
BLACK = (0,0,0)
P1COLOR = (255,0,0)
P2COLOR = (255,255,0)

ROW_COUNT = 6
COLUMN_COUNT = 7

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE/2 - 5)

screen = pygame.display.set_mode(size)




