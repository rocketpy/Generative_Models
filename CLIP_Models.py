# CLIP Models - TiC-CLIP: Continual Training of CLIP Models

# https://github.com/apple/ml-tic-clip


# Installing dependencies
"""
We use OpenCLIP and DataComp for training and evaluation. We have made minor modifications to OpenCLIP/DataComp for support of TiC evaluations and training.
  To checkout the specific version of each library and apply our corresponding patch run the following commands in order:

# Clone TiC-CLIP repository
git clone git@github.com:apple/ml-tic-clip.git
cd ml-tic-clip/

# Clone DataComp repository and apply patch
git clone https://github.com/mlfoundations/datacomp.git
cd datacomp
git checkout 7fdb5c653e70d9c6fcc63ac7c8c66843e7c6f3e8
git apply ../datacomp.patch  # Minor changes to support TiC training/evaluation
bash create_env.sh
cd ..

# Clone OpenCILP repository, apply patch, and install
git clone https://github.com/mlfoundations/open_clip.git
cd open_clip
git checkout 73fa7f03a33da53653f61841eb6d69aef161e521
git apply ../open_clip.patch  # Support for sampling without replacement
pip install --ignore-installed .
cd ..
To activate the environment:

conda activate datacomp
If using cloud storage services (e.g., AWS S3), you'll need to install additional dependencies (e.g. pip install 'cloudpathlib[s3]').
"""
