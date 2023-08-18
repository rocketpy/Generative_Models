# Fooocus is an image generating software.

# https://github.com/lllyasviel/Fooocus


# Windows, directly download Fooocus with:
# https://github.com/lllyasviel/Fooocus/releases/download/release/Fooocus_win64_1-1-10.7z

# To run: After download the file, please uncompress it, and then run the "run.bat".
"""
In the first time you launch the software, it will automatically download models:
It will download sd_xl_base_1.0_0.9vae.safetensors from here as the file "Fooocus\models\checkpoints\sd_xl_base_1.0_0.9vae.safetensors".
It will download sd_xl_refiner_1.0_0.9vae.safetensors from here as the file "Fooocus\models\checkpoints\sd_xl_refiner_1.0_0.9vae.safetensors".
"""

# Requirements !!!
# relatively low-end laptop with 16GB System RAM and 6GB VRAM (Nvidia 3060 laptop).
# The speed on this machine is about 1.35 seconds per iteration. 
# Pretty impressive – nowadays laptops with 3060 are usually at very acceptable price.

# Linux
"""
The command lines are:
git clone https://github.com/lllyasviel/Fooocus.git
cd Fooocus
conda env create -f environment.yaml
conda activate fooocus
pip install -r requirements_versions.txt
"""
