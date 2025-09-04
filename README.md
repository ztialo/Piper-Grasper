# Isaac Lab Piper-Grasper Project

## Overview

This project/repository is a project based on Isaac Lab. Using differential inverse kinematic and OpenCV, the end goal of this project is to pick up objects using vision sensors.

**Key Files:**

Three script files show how the project built:

- `arm-ariticulation.py` Applying articulation to robot arms, setting up scene and sensors.
- `piper_differential_ik.py` Differential inverse kinematic applied to the robot articulation, accuracy and different testing end effect coordinate can be tested here.
- `piper_pick_and_place.py` Using vision sensor's RGB and depth output to estimate the coordinate of target and goal to achieve pick and place task.

**Keywords:** OpenCV, Inverse Kinematic, isaaclab

## Install python package

after cloning and cd into the repository, install the python package

    python install -e piper-grasper

## Script Executable Commands

Inside the IsaacLab file, run the following commands to simulate the relative file.

- `arm-ariticulation.py`
    ```bash
    ./isaaclab.sh -p ../Piper-Grasper/scripts/arm-ariticulation.py

- `piper_differential_ik.py`
     ```bash
    ./isaaclab.sh -p ../Piper-Grasper/scripts/piper_differential_ik.py --enable_cameras
- `piper_pick_and_place.py`
    ```bash
    ./isaaclab.sh -p ../Piper-Grasper/scripts/piper_pick_and_place.py --enable_cameras
