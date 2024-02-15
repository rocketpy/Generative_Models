# GeneGPT - code and data for GeneGPT, a tool-augmented LLM for improved access to biomedical information.

# https://github.com/ncbi/GeneGPT


# Requirements
"""
The code has been tested with Python 3.9.13. Please first install the required packages by:

pip install -r requirements.txt
You also need an OpenAI API key to run GeneGPT with Codex. Replace the placeholder with your key in config.py:

$ cat config.py 
API_KEY = 'YOUR_OPENAI_API_KEY'
"""

# Using GeneGPT
"""
After setting up the environment, one can run GeneGPT on GeneTuring by:

python main.py 111111
where 111111 denotes that all Documentations (Dc.1-2) and Demonstrations (Dm.1-4) are used.

To run GeneGPT-slim, simply use:

python main.py 001001
which will only use the Dm.1 and Dm.4 for in-context learning.
"""

# Evaluating GeneGPT
"""
One can evaluate the results by running:

python evaluate.py ${RESULT_DIRECTORY}
For example, we also put our experimental results in geneturing_results and geneturing_results. By running:

python evaluate.py geneturing_results/001001/
The user can get the evaluation results of GeneGPT-slim:

Evaluating geneturing_results/001001/Gene alias.json
0.84
"""
