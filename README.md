# Chess AI Project

## Overview

This project focuses on creating a chess artificial intelligence model that utilizes machine learning. The model is trained using positions as `x_train` and the corresponding Stockfish scores as `y_train`. In the training process, pieces such as pawn, knight, bishop, rook, queen, and king are represented as 1, 2, 3, 4, 5, 6, respectively, with negative values for black pieces. The computer plays with white pieces, as is standard in chess.

The training data is sourced from the Lichess database, and it's important to note that analyzing a large number of games may lead to errors. To address this, the `modelCompiler.py` file includes a variable named `determinedGameCount` that can be adjusted to control the number of analyzed games. It is recommended to analyze a moderate number of games for optimal performance.

## How to Run

1. Run `modelCompiler.py` to compile and train the model.
2. After compiling the model, run `main_window.py` to start the application.

## Requirements

Ensure you have the following dependencies installed:

- TensorFlow
- Keras
- python-chess
- PyQt5

## Contributing

Contributions and suggestions are welcome. Feel free to reach out if you are interested in contributing to the project.

