# Llama 3 - the power of large language models

# https://github.com/meta-llama/llama3
# https://llama.meta.com/llama-downloads/


# To use with transformers, the following pipeline snippet will download and cache the weights:

import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
  "text-generation",
  model="meta-llama/Meta-Llama-3-8B-Instruct",
  model_kwargs={"torch_dtype": torch.bfloat16},
  device="cuda",
)


# Quick Start
"""
You can follow the steps below to quickly get up and running with Llama 3 models.
These steps will let you run quick inference locally. For more examples, see the Llama recipes repository.

    In a conda env with PyTorch / CUDA available clone and download this repository.

    In the top-level directory run:

    pip install -e .

    Visit the Meta Llama website and register to download the model/s.

    Once registered, you will get an email with a URL to download the models.
    You will need this URL when you run the download.sh script.

    Once you get the email, navigate to your downloaded llama repository and run the download.sh script.
        Make sure to grant execution permissions to the download.sh script
        During this process, you will be prompted to enter the URL from the email.
        Do not use the “Copy Link” option but rather make sure to manually copy the link from the email.

    Once the model/s you want have been downloaded, you can run the model locally using the command below:

torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 512 --max_batch_size 6

Note

    Replace Meta-Llama-3-8B-Instruct/ with the path to your checkpoint directory and
    Meta-Llama-3-8B-Instruct/tokenizer.model with the path to your tokenizer model.
    The –nproc_per_node should be set to the MP value for the model you are using.
    Adjust the max_seq_len and max_batch_size parameters as needed.
    This example runs the example_chat_completion.py found in this repository but
    you can change that to a different .py file.
"""
