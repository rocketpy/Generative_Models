# tinyzero - Easily train AlphaZero-like agents on any environment

# https://github.com/s-casci/tinyzero

"""
Usage
Make sure you have Python >= 3.8 intalled. After that, run pip install -r requirements.txt to install the necessary dependencies.

Then, to train an agent on one of the existing environments, run:

python3 tictactoe/two_dim/train.py
where tictactoe/two_dim is the name of the environment you want to train on.

Inside the train script, you can change parameters such as the number of episodes, the number of simulations and enable wandb logging.

Similarly, to evaluate the trained agent run:

python3 tictactoe/two_dim/eval.py
"""


# Add an environment
"""
To add a new environment, you can follow the game.py files in every existing examples.

The environment you add should implement the following methods:

reset(): resets the environment to its initial state
step(action): takes an action and modifies the state of the environment accordingly
get_legal_actions(): returns a list of legal actions
undo_last_action(): cancels the last action taken
to_observation(): returns the current state of the environment as an observation (a numpy array) to be used as input to the model
get_result(): returns the result of the game (for example, it might be 1 if the first player won, -1 if the second player won, 0 if it's a draw, and None if the game is not over yet)
get_first_person_result(): returns the result of the game from the perspective of the current player (for example, it might be 1 if the current player won, -1 if the opponent won, 0 if it's a draw, and None if the game is not over yet)
swap_result(result): swaps the result of the game (for example, if the result is 1, it should become -1, and vice versa). It's needed to cover all of the possible game types (single player, two players, zero-sum, non-zero-sum, etc.)
"""
