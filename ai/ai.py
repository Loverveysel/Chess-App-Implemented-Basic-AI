import chess.pgn
import chess.engine
import numpy as np
import tensorflow as tf
from keras import models
import sys

from os.path import dirname, abspath 

currentDir = dirname(abspath(__file__))
parentDir = dirname(currentDir)
sys.path.append(parentDir)

class Computer:
    def __init__(self):
        self.model = models.load_model("saved_model/model.h5")
        print("model loaded...")
        self.xCoordinates = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.yCoordinates = ["1", "2", "3", "4", "5", "6", "7", "8"]
        fileName = "ai/train_data/data2.pgn"

        file = open(fileName)

        self.game = chess.pgn.read_game(file)
        self.board = self.game.board()

       

    def pieceToInt(self, piece):
        
            
        if piece.type == "Pawn":
            if piece.color[0] == 'w':
                return 1
            else:
                return -1
        elif piece.type == "Knight":
            if piece.color[0] == 'w':
                return 2
            else:
                return -2
        
        elif piece.type == "Bishop":
            if piece.color[0] == 'w':
                return 3
            else:
                return -3
        
        elif piece.type == "Rook":
            if piece.color[0] == 'w':
                return 4
            else:
                return -4
        
        elif piece.type == "Queen":
            if piece.color[0] == 'w':
                return 5
            else:
                return -5
        
        elif piece.type == "King":
            if piece.color[0] == 'w':
                return 6
            else:
                return -6
        else:
            return 0

    
    def symbolToInt(self, symbol):
        if symbol == "P":
            return 1
        elif symbol == "N":
            return 2
        elif symbol == "B":
            return 3
        elif symbol == "Q":
            return 5
        elif symbol == "K":
            return 6
        elif symbol == "R":
            return 4
        if symbol == "p":
            return -1
        elif symbol == "n":
            return -2
        elif symbol == "b":
            return -3
        elif symbol == "q":
            return -5
        elif symbol == "k":
            return -6
        elif symbol == "r":
            return -4

    def pushMove(self, move):
        move = chess.Move.from_uci(move)
        self.board.push(move)

    def position_to_tensor(self):
        positions = []

        position = [self.symbolToInt(self.board.piece_at(square).symbol()) if self.board.piece_at(square) else 0 for square in chess.SQUARES]
        positions.append(position)

        positions = np.array(positions)
        positions = positions.astype("float32")
        print(self.board)

        return positions

    # Make a prediction with using the model
    def predict_best_move(self):
        # Choose the highest probability move
        movesProbabilities = []
        legal_moves = []
        for move in self.board.legal_moves:
            legal_moves.append(move)    
        for move in legal_moves:
            self.board.push(move)
            tensor = self.position_to_tensor()
            probabilty = self.model.predict(tensor)
            movesProbabilities.append((move, probabilty))
            self.board.pop()

        print("movesProbability : ") 
        print(movesProbabilities)
        pro = 0
        absMove = ""
        for move in movesProbabilities:
            if move[1] > pro:
                pro = move[1][0][0]
                absMove = move[0]
        
        print(absMove)
        return absMove

"""
fileName = "ai/train_data/data2.pgn"

file = open(fileName)

game = chess.pgn.read_game(file)
board = game.board()
i = 0
for move in game.mainline_moves():
    board.push(move)
    if i == 6:
        break
    i += 1

computer = Computer()
print(board)

tensor = computer.position_to_tensor()
print(tensor)
print(computer.predict_best_move())"""