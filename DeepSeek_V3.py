# DeepSeek-V3 a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. 

# https://github.com/deepseek-ai/DeepSeek-V3

# Chat Website & API Platform
# You can chat with DeepSeek-V3 on DeepSeek's official website: chat.deepseek.com
# We also provide OpenAI-Compatible API at DeepSeek Platform: platform.deepseek.com


# How to Run Locally
"""
DeepSeek-V3 can be deployed locally using the following hardware and open-source community software:

DeepSeek-Infer Demo: We provide a simple and lightweight demo for FP8 and BF16 inference.
SGLang: Fully support the DeepSeek-V3 model in both BF16 and FP8 inference modes, with Multi-Token Prediction coming soon.
LMDeploy: Enables efficient FP8 and BF16 inference for local and cloud deployment.
TensorRT-LLM: Currently supports BF16 inference and INT4/8 quantization, with FP8 support coming soon.
vLLM: Support DeepSeek-V3 model with FP8 and BF16 modes for tensor parallelism and pipeline parallelism.
AMD GPU: Enables running the DeepSeek-V3 model on AMD GPUs via SGLang in both BF16 and FP8 modes.
Huawei Ascend NPU: Supports running DeepSeek-V3 on Huawei Ascend devices.
Since FP8 training is natively adopted in our framework, we only provide FP8 weights. 
If you require BF16 weights for experimentation, you can use the provided conversion script to perform the transformation.

Here is an example of converting FP8 weights to BF16:

cd inference
python fp8_cast_bf16.py --input-fp8-hf-path /path/to/fp8_weights --output-bf16-hf-path /path/to/bf16_weights
"""

# System Requirements

# Linux with Python 3.10 only. Mac and Windows are not supported.

# Dependencies:
"""
torch==2.4.1
triton==3.0.0
transformers==4.46.3
safetensors==0.4.5


Model Weights & Demo Code Preparation
First, clone our DeepSeek-V3 GitHub repository:

git clone https://github.com/deepseek-ai/DeepSeek-V3.git

Navigate to the inference folder and install dependencies listed in requirements.txt. 
Easiest way is to use a package manager like conda or uv to create a new virtual environment and install the dependencies.

cd DeepSeek-V3/inference
pip install -r requirements.txt

Model Weights Conversion
Convert Hugging Face model weights to a specific format:

python convert.py --hf-ckpt-path /path/to/DeepSeek-V3 --save-path /path/to/DeepSeek-V3-Demo --n-experts 256 --model-parallel 16

Run
Then you can chat with DeepSeek-V3:

torchrun --nnodes 2 --nproc-per-node 8 --node-rank $RANK --master-addr $ADDR generate.py --ckpt-path /path/to/DeepSeek-V3-Demo --config configs/config_671B.json --interactive --temperature 0.7 --max-new-tokens 200

Or batch inference on a given file:

torchrun --nnodes 2 --nproc-per-node 8 --node-rank $RANK --master-addr $ADDR generate.py --ckpt-path /path/to/DeepSeek-V3-Demo --config configs/config_671B.json --input-file $FILE
"""
