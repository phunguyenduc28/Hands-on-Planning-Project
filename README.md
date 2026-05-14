## Simulationn
```bash
ros2 launch online_motion_planning exploration_sim.launch.py
```

## Real testing
```bash
ros2 launch online_motion_planning turtlebot_real_test.launch.py
```
Useful commands:
Teleoperate Turtlebot2

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r __ns:=/turtlebot
```
Turning off Turtlebot2
```bash
ros2 service call /turtlebot/turtlebot_shutdown std_srvs/srv/Trigger {}
```
View tf frames for debugging
```bash
ros2 run tf2_tools view_frames
```