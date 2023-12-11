import chess
import chess.pgn
import numpy as np
from keras import layers
from keras import models

fileName = "ai/train_data/data2.pgn"

def createDataset():
    count = 0
    totalGame = 121332
    determinedGame = 2

    file = open(fileName)
    game = chess.pgn.read_game(file)

    positions = []
    score = []

    engine = chess.engine.SimpleEngine.popen_uci(r"C:\Users\alper\Downloads\stockfish\stockfish-windows-x86-64-avx2.exe")


    while game is not None:
        board = game.board()

        for move in game.mainline_moves():
            position = [symbolToInt(board.piece_at(square).symbol()) if board.piece_at(square) else 0 for square in chess.SQUARES]
            positions.append(position)
            result = engine.analyse(board, chess.engine.Limit(time = 1.0))
            score.append(result["score"].relative.score())

            board.push(move)  # Make the move on the board

        game = chess.pgn.read_game(file)
        count += 1
        progress =  count/ determinedGame * 100
        strProgress = f"{progress:.2f}"
        print("Progress : %" + strProgress)
        if count == determinedGame:
            break
    return np.array(positions), np.array(score)

#positions, labels = createDataset()

def symbolToInt(symbol):
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

positions , score = createDataset()
positions = positions.astype('float32')
score = score.astype('float32')

# Describe the moedel and fit
model = models.Sequential([
    layers.InputLayer(input_shape=(64, )),  # 8x8 board
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(positions, score, epochs=10, validation_split=0.2)

# Evaluate the model
evaluationResult = model.evaluate(positions, score)

print("Loss:", evaluationResult[0])
print("Accuracy:", evaluationResult[1])

model.save("saved_model/model.h5")