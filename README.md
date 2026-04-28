## Description
ROS2 workspace to run simulation pipeline including Stonefish, ROS2, Turtlebot packages. This workspace is created to run assignments in Hands-on Localization and Hands-on Planning courses, taught by Universtiy of Girona in AY25-26.

<!-- ## Install dependencies
```bash
pip3 install bresenham --break-system-packages
sudo apt install ros-jazzy-tf-transformations
```

Copy the missing packages to the workspace.

## Run simulaton
In the first terminal:
```bash
ros2 launch online_motion_planning turtlebot_simulation_hoi_circuit_1_rrt.launch.py
``` -->

### NOTE - I am using my own localisation package which we developed for Hands-on-Localisation first lab which provides me the odometry

Set up the RTAB mapping package first through the word file which I have provided.

## Running the RTAB mapping package
Start the simulation in one terminal. In another terminal, start RTAB mapping with the following launch file after sourcing the workspace
```bash
ros2 launch rtabmap_examples realsense_d435i_color.launch.py 
```

## Inflated map
In another terminal, after building the packages, start the node the generates inflated map
```bash
ros2 run grid_mapping occupancy_grid_original 
```

## Bi-directional RRT for Global Planning with Frontier Based Planning
In another terminal, start the node that runs the frontier based planning using bidirectional RRT*
```bash
ros2 run online_motion_planning frontier_birrt_tb 
```

## Rviz visualisation
Add visualisation in Rviz as per your need.

## Workspace Build
colcon build --parallel-workers 2 --executor sequential --symlink-install