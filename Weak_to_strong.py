# Weak-to-strong generalization

# https://github.com/openai/weak-to-strong

# Installation
"""
You need to have Python installed on your machine. The project uses pyproject.toml to manage dependencies.
To install the dependencies, you can use a package manager like pip:

pip install .

Running the Script

The main script of the project is train_weak_to_strong.py. It can be run from the command line using the following command:

python train_weak_to_strong.py

The script accepts several command-line arguments to customize the training process. Here are some examples:

python train_weak_to_strong.py --batch_size 32 --max_ctx 512 --ds_name "sciq" --loss "logconf" --n_docs 1000
--n_test_docs 100 --weak_model_size "gpt2-medium" --strong_model_size "gpt2-large" --seed 42
"""

# Plotting
RESULTS_PATH = "../../your_sweep_results_path"

PLOT_ALL_SEEDS = False
# Full sweep
MODELS_TO_PLOT = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "Qwen/Qwen-1_8B", "Qwen/Qwen-7B", "Qwen/Qwen-14B"]
# Minimal sweep
# MODELS_TO_PLOT = ["gpt2", "gpt2-medium", "gpt2-large"]
