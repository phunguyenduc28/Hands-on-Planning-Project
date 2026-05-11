#!/usr/bin/env python3
"""
=============================================================================
COMPLETE NAVIGATION PIPELINE SETUP AND USAGE GUIDE
=============================================================================

This guide explains the complete navigation system:
1. Map Saving (from occupancy grid)
2. Path Planning (RRT* algorithm)
3. Waypoint Execution (robot controller)

=============================================================================
SYSTEM ARCHITECTURE
=============================================================================

[Bag File Playback]
       ↓
[occupancy_grid_node.py] → publishes /map topic
       ↓
[map_saver.py] → saves map as PGM + YAML (~/maps/)
       ↓
[path_planner_rrt_star.py] → loads map, subscribes to /goal_pose
       ↓
       publishes /plan (nav_msgs::Path with waypoints)
       ↓
[node_waypoint_controller.py] → executes waypoints sequentially
       ↓
publishes /turtlebot/cmd_vel to move robot

=============================================================================
STEP 1: GENERATE AND SAVE THE OCCUPANCY MAP
=============================================================================

Terminal 1 - Play bag file:
  ros2 bag play path/to/your/bag.mcap --clock

Terminal 2 - Run occupancy grid node:
  ros2 run lab1_1 occupancy_grid_node

Terminal 3 - Visualize in RViz:
  rviz2
  
  In RViz:
  - Set Fixed Frame to "odom"
  - Add visualization from:
    * /map topic (OccupancyGrid)
    * /tf topic (TF tree)

Terminal 4 - Save the map when satisfied:
  ros2 service call /save_map std_srvs/srv/Empty
  
  Map will be saved to: ~/maps/map_YYYYMMDD_HHMMSS.pgm
                        ~/maps/map_YYYYMMDD_HHMMSS.yaml

YAML file format:
  image: map_YYYYMMDD_HHMMSS.pgm
  resolution: 0.05  # meters per cell
  origin: [x, y, 0.0]  # map origin in world frame
  occupied_thresh: 0.65
  free_thresh: 0.196
  negate: 0

=============================================================================
STEP 2: PATH PLANNING WITH RRT*
=============================================================================

Terminal 5 - Run path planner node:
  ros2 run lab1_1 path_planner_rrt_star

Parameters (can be set via ros2 run or launch file):
  - robot_radius: 0.2 (meters, for collision checking)
  - map_file: ~/maps/map_latest.yaml (or use latest map)
  - rrt_max_iterations: 2000
  - rrt_step_size: 0.3

In RViz:
  - Subscribe to /path_markers topic to see path visualization
  - Add "Path" display and select /plan topic
  
When you click "Publish Point" in RViz and select a goal position:
  - RViz publishes PoseStamped to /goal_pose topic
  - Path planner receives it and computes path
  - Path is published to /plan topic
  - Markers show waypoints and path line

=============================================================================
STEP 3: EXECUTE WAYPOINTS WITH CONTROLLER
=============================================================================

Terminal 6 - Run waypoint controller:
  ros2 run lab1_1 node_waypoint_controller

Parameters (adjustable in launch file):
  - k_v: 0.5 (linear velocity gain)
  - k_w: 2.0 (angular velocity gain)
  - dist_tolerance: 0.1 (position tolerance in meters)
  - v_max: 0.5 (max linear velocity)
  - w_max: 1.0 (max angular velocity)

The controller:
  1. Receives /plan topic (path with waypoints)
  2. Executes waypoints sequentially
  3. Publishes /turtlebot/cmd_vel for robot motion
  4. Publishes /controller_status for feedback

Two modes of operation:
  a) Single goal: Publish a single goal to /goal_pose → robot moves there
  b) Path planning: Click multiple points in RViz → planner creates path → 
                     controller executes all waypoints

=============================================================================
COMPLETE WORKFLOW EXAMPLE
=============================================================================

Terminal 1: ros2 bag play bag.mcap --clock

Terminal 2: ros2 run lab1_1 occupancy_grid_node

Terminal 3: rviz2
  - Set Fixed Frame to "odom"
  - Add /map (OccupancyGrid)
  - Configure for interaction

Terminal 4: ros2 service call /save_map std_srvs/srv/Empty

Terminal 5: ros2 run lab1_1 path_planner_rrt_star

Terminal 6: ros2 run lab1_1 node_waypoint_controller

In RViz:
  1. Click "Publish Point" button
  2. Click somewhere on the map as goal
  3. Watch path being computed and displayed
  4. Robot starts moving following the waypoints
  5. Watch /controller_status topic for progress

Monitor progress:
  ros2 topic echo /controller_status

=============================================================================
KEY PARAMETERS AND TUNING
=============================================================================

RRT* Planning:
  - rrt_step_size: Smaller = more granular, slower planning
                   Larger = faster planning, might miss narrow paths
  - rrt_max_iterations: More = better paths, longer planning time
  - rewire_radius: Larger = better paths, slower computation

Robot Controller:
  - k_v: Linear velocity proportional gain
         Larger = faster, more overshoot
  - k_w: Angular velocity proportional gain
         Larger = sharper turns, more oscillations
  - dist_tolerance: Distance to consider waypoint "reached"
         Smaller = more accurate, slower
  - v_max, w_max: Command saturation limits

Collision Checking:
  - robot_radius: Must match actual robot footprint
                  Larger = more conservative (safer)
                  Smaller = tighter paths

=============================================================================
TROUBLESHOOTING
=============================================================================

1. Map not saving:
   - Check ~/maps/ directory exists (created automatically)
   - Check map is being published to /map topic
   - Use: ros2 topic echo /map

2. Path planning fails:
   - Start and/or goal in occupied space
   - RRT* can't find path (increase max_iterations)
   - Robot radius too large for map resolution
   - Check grid alignment with goal position

3. Robot moving incorrectly:
   - Verify controller is subscribed to /plan topic
   - Check waypoints are reasonable positions
   - Tune k_v and k_w gains
   - Verify /turtlebot/cmd_vel is being published

4. Transform issues:
   - Check transform buffer: ros2 run tf2_tools view_frames
   - Ensure laser_frame and map_frame are correct
   - Increase TF buffer time if needed

=============================================================================
VISUALIZING IN RVIZ
=============================================================================

Recommended RViz display configuration:

1. Add displays:
   - Type: OccupancyGrid → Topic: /map
   - Type: Path → Topic: /plan
   - Type: MarkerArray → Topic: /path_markers
   - Type: TF → Shows coordinate frames
   - Type: RobotModel → Shows TurtleBot links

2. Use Publish Point tool:
   - Select from Tools > Publish Point
   - Click on map to set goal
   - Watch path computation and execution

3. Monitor topics real-time:
   ros2 topic echo /controller_status
   ros2 topic echo /map (just for frequency)
   ros2 topic echo /plan

=============================================================================
"""

if __name__ == '__main__':
    print(__doc__)
