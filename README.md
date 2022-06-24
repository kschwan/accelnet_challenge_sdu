# 2021-2022 AccelNet Surgical Robotics Challenge, SDU

This is the submission to the 2021-2022 AccelNet Surgical Robotics Challenge from SDU Robotics University of Southern Denmark, Odense. I am submitting a solution to challenge #1 only. Although we originally planned to also submit solutions to tasks #2 and #3 I have not had the resources to finish them.

Our needle pose estimation pipeline uses a U-net-like CNN for pixel segmentation and runs very slow on a CPU. With a GeForce RTX 3070 Ti I see segmentation rates of 4-5 FPS on 1920x1080 images. Note that the first image in the pipeline is processed significantly slower than the subsequent ones.

*We use the suture thread for determining the needle's orientation and request that you run the simulated environment that includes the thread.*

## Run procedure:

Our submission is set up as a catkin package. It can be run using the following procedure:

- Put the `accelnet_challenge_sdu` package in a catkin workspace's `src` dir.
- Build the workspace with `catkin build` or possibly `catkin_make`.
- Source the `devel/setup.bash`.
- Run the AMBF simulator `ambf_simulator ambf_simulator --launch_file surgical_robotics_challenge/launch.yaml -l 0,1,3,4,14,15 -p 120 -t 1 --override_max_comm_freq 120`
- Run CRTK interface `python3 surgical_robotics_challenge/scripts/surgical_robotics_challenge/launch_crtk_interface.py`
- Run evaluation script `python3 evaluation.py -t sdu -e 1`
- Run our solution to task #1 with `roslaunch accelnet_challange_sdu task1.roslaunch`.

## Dependencies:

We use the following non-standard Python packages (installable via pip)

- ~~open3d (tested with 0.15.2)~~
- tensorflow (tested with 2.8.0 and 2.9.1)
- scipy (tested with 1.8.1)
- numpy-quaternion (tested with 2022.4.2)

Apart from these we use packages that should be installed with ROS.

## Testing hardware
We've tested our work a desktop with Ryzen 5 5600X, GeForce RTX 3070 Ti (with CUDA), 32 GB RAM and on a laptop with Intel i7-1165G7, Integrated Graphics, 32 GB RAM. Both machines were running Ubuntu 20.04 with ROS Noetic.

## Contact

Kim Lindberg Schwaner <kils@mmmi.sdu.dk>
