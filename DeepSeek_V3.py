# DeepSeek-V3 a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. 

# https://github.com/deepseek-ai/DeepSeek-V3


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
