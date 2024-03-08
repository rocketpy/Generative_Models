# Home Robot - open-source robotic mobile manipulation stack!

# https://github.com/facebookresearch/home-robot

# git clone https://github.com/facebookresearch/home-robot --branch home-robot-ovmm-challenge-2023-v0.1.2


# Installation
"""
Preliminary
HomeRobot requires Python 3.9. Installation on a workstation requires conda and mamba.
Installation on a robot assumes Ubuntu 20.04 and ROS Noetic.

To set up the hardware stack on a Hello Robot Stretch, see the ROS installation instructions in home_robot_hw.

You may need a calibrated URDF for our inverse kinematics code to work well; see calibration notes.

Spot installation instructions are experimental but are also available.

Network Setup
Follow the network setup guide to set up your robot to use the network,
and make sure that it can communicate between workstation and robot via ROS. On the robot side,
start up the controllers with:

roslaunch home_robot_hw startup_stretch_hector_slam.launch
"""


# Workstation Instructions
"""
To set up your workstation, follow these instructions.
HomeRobot requires Python 3.9. These instructions assume that your system supports CUDA 11.7 or better for pytorch; earlier versions should be fine,
but may require some changes to the conda environment.

If on Ubuntu, ensure some basic packages are installed:

sudo apt update
sudo apt install build-essential zip unzip
Then clone home-robot locally:

git clone https://github.com/facebookresearch/home-robot.git
cd ./home-robot
"""


# Create Your Environment
"""
If necessary, install mamba in your base conda environment. Optionally: install ROS noetic on your workstation.

# If using ROS - make sure you don't have PYTHONPATH set
unset PYTHONPATH

# Otherwise, use the version in src/home_robot
mamba env create -n home-robot -f src/home_robot/environment.yml

# Activate the environment
conda activate home-robot

# Optionally, update this environment to install ROS
mamba env update -f src/home_robot_hw/environment.yml
"""


# Run Install Script
"""
Make sure you have the correct environment variables set: CUDA_HOME should point to your cuda install, matching the one used by your python environment.
We recommend 11.7, and it's what will be automatically installed above as a part of the conda environment.

To build some third-party dependencies, you also need the full cuda toolkit with its compiler, nvcc. You can download it from nvidia's downloads page.
Download the runfile, and make sure to check the box NOT to install your drivers or update your system cuda version. It will be installed at a separate location.

Then make sure the environment variables are set to something reasonable, for example:

export HOME_ROBOT_ROOT=$USER/home-robot
export CUDA_HOME=/usr/local/cuda-11.7
Finally, you can run the install script to download submodules, model checkpoints, and build Detic for open-vocabulary object detection:

conda activate home-robot
cd $HOME_ROBOT_ROOT
./install_deps.sh
"""
  
