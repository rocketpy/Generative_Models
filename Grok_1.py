# Grok-1 - This repository contains JAX example code for loading and running the Grok-1 open-weights model.

# https://github.com/xai-org/grok-1


"""
Then, run

pip install -r requirements.txt
python run.py

to test the code.


Downloading the weights

You can download the weights using a torrent client and this magnet link:

magnet:?xt=urn:btih:5f96d43576e3d386c9ba65b883210a393b68210e&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce

or directly using HuggingFace 🤗 Hub:

git clone https://github.com/xai-org/grok-1.git && cd grok-1
pip install huggingface_hub[hf_transfer]
huggingface-cli download xai-org/grok-1 --repo-type model --include ckpt-0/* --local-dir checkpoints --local-dir-use-symlinks False

"""
