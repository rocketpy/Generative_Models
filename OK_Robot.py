# OK-Robot -  is a zero-shot modular framework that effectively combines the state-of-art navigation and manipulation models to perform pick and place tasks in real homes. 

# https://github.com/ok-robot/ok-robot


# Hardware and software requirements
"""
Hardware required:

An iPhone Pro with Lidar sensors
Hello Robot Stretch with Dex Wrist installed
A workstation with GPU to run pretrained models
Software required:

Python 3.9
Record3D (>1.18.0)
CloudComapre
"""


# Run Experiments
"""
First set up the environment with the tapes, position the robot properly and scan the environment to get a r3d file from Record3D.
Place it in /navigation/r3d/ run following commands.

On Workstation:
In one terminal run the Navigation Module.

mamba activate ok-robot-env

cd ok-robot-navigation
python path_planning.py debug=False min_height={z coordinates of the ground tapes + 0.1} dataset_path='r3d/{your_r3d_filename}.r3d' cache_path='{your_r3d_filename}.pt' pointcloud_path='{your_r3d_filename}.ply'
In another terminal run the Manipulation module

mamba activate ok-robot-env

cd ok-robot-manipulation/src
python demo.py --open_communication --debug
"""


# On Robot:
# Home-robot: https://github.com/facebookresearch/home-robot
"""
Before running anything on the robot, you need to calibrate it by

stretch_robot_home.py

Our robot codes rely on robot controllers provided by home-robot.
Just like running other home-robot based codes, you need to run two processes synchronously in two terminals.

In one terminal start the home-robot

roslaunch home_robot_hw startup_stretch_hector_slam.launch
In another terminal run the robot control. More details in ok-robot-hw

cd ok-robot-hw

python run.py -x1 [x1] -y1 [y1] -x2 [x2] -y2 [y2] -ip [your workstation ip]
"""
