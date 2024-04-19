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

