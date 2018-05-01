# CS 181 Practical 4: Reinforcement Learning.

Implementation of reinforcement learning (RL) for the game Swingy Monkey.

## Usage

There are 4 Python scripts: `SwingyMonkey.py` contains the implementation of the game, and the other 3 files (`vanillaQ.py`, `linearQ.py` and `deepQ.py`) contains different RL methods for the game. The folder `res` contains files needed for the execution of the file. The Python module `pygame` must be installed to run the game. The game can be played manually with the command `python SwingyMonkey.py`.

`vanillaQ.py` implementations traditional Q-learning, and can be run with the command `python vanillaQ.py`. Hyper-parameters as well as the number of training iterations can be modified within the file itself.

`linearQ.py` implementations approximate Q-learning using a linear approximation function, and can be run with the command `python linearQ.py`. Hyper-parameters as well as the number of training iterations can be modified within the file itself.

`deepQ.py` implementations approximate Q-learning using deep neural networks, and can be run with the command `python deepQ.py`. Hyper-parameters as well as the number of training iterations can be modified within the file itself. The Python module `pytorch` must be installed to run the neural network.

Refer to `report.pdf` for further details of implementation.

## Authors

* Sam Lurye

* Wanqian Yang

* William Gao Yuan

Harvard College Class of 2020. Canvas Group: P4 Group 24.
