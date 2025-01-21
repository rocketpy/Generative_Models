# MiniCPM-o - A GPT-4o Level MLLM for Vision, Speech and Multimodal Live Streaming on Your Phone

# https://github.com/OpenBMB/MiniCPM-o

# Features:
"""
MiniCPM-o is the latest series of end-side multimodal LLMs (MLLMs) ungraded from MiniCPM-V. 
The models can now take image, video, text, and audio as inputs and provide high-quality text and speech outputs in an end-to-end fashion. 
Since February 2024, we have released 6 versions of the model, aiming to achieve strong performance and efficient deployment. 
"""

# Local WebUI Demo
"""
Please ensure that transformers==4.44.2 is installed, as other versions may have compatibility issues.

If you are using an older version of PyTorch, you might encounter this issue "weight_norm_fwd_first_dim_kernel" not implemented for 'BFloat16', 
Please add self.minicpmo_model.tts.float() during the model initialization.

For real-time voice/video call demo:

Launch model server:
pip install -r requirements_o2.6.txt

python web_demos/minicpm-o_2.6/model_server.py
"""
