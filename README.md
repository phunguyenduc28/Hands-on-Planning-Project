## Description
ROS2 workspace to run simulation pipeline including Stonefish, ROS2, Turtlebot packages. This workspace is created to run assignments in Hands-on Localization and Hands-on Planning courses, taught by Universtiy of Girona in AY25-26.

## Install dependencies
```bash
pip3 install bresenham --break-system-packages
sudo apt install ros-jazzy-tf-transformations
```

Copy the missing packages to the workspace.

## Run simulaton
In the first terminal:
```bash
ros2 launch online_motion_planning turtlebot_simulation_hoi_circuit_1_rrt.launch.py
```