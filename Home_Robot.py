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
